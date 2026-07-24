#!/usr/bin/env python3
"""Export legacy and high-resolution streaming Sortformer checkpoints for Riva.

Both variants use the recovered six-input/four-output streaming contract. New
high-resolution checkpoints execute their learned subpixel upsampler and then
average-pool predictions back to the 80 ms cadence consumed by Riva.

This script intentionally exports the acoustic feature input, not raw audio.  The
log-Mel preprocessor remains outside the graph, matching the recovered artifact.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import open_dict

from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel


INPUT_NAMES = [
    "chunk",
    "chunk_lengths",
    "spkcache",
    "spkcache_lengths",
    "fifo",
    "fifo_lengths",
]

OUTPUT_NAMES = [
    "predictions",
    "pred_lengths",
    "chunk_embs",
    "chunk_emb_lengths",
]

DYNAMIC_AXES = {
    "chunk": {0: "batch_size", 1: "chunk_frames"},
    "chunk_lengths": {0: "batch_size"},
    "spkcache": {0: "batch_size", 1: "spkcache_len"},
    "spkcache_lengths": {0: "batch_size"},
    "fifo": {0: "batch_size", 1: "fifo_len"},
    "fifo_lengths": {0: "batch_size"},
    "predictions": {0: "batch_size", 1: "output_frames"},
    "pred_lengths": {0: "batch_size"},
    "chunk_embs": {0: "batch_size", 1: "emb_frames"},
    "chunk_emb_lengths": {0: "batch_size"},
}


class RivaStreamingExportModel(SortformerEncLabelModel):
    """TensorRT-friendly export behavior shared by old and new checkpoints."""

    @property
    def input_names(self):
        return INPUT_NAMES

    @property
    def output_names(self):
        return OUTPUT_NAMES

    @staticmethod
    def concat_and_pad(embs, lengths):
        """TRT-friendly, fully vectorized replacement recovered from the graph.

        The output time dimension is the sum of the three allocated input time
        dimensions. ``total_lengths`` marks the valid prefix for each batch item.
        This avoids data-dependent allocation and Python slice-assignment loops.
        """
        spkcache, fifo, chunk_embs = embs
        spkcache_lengths, fifo_lengths, chunk_emb_lengths = lengths

        total_lengths = spkcache_lengths + fifo_lengths + chunk_emb_lengths
        allocated_frames = spkcache.shape[1] + fifo.shape[1] + chunk_embs.shape[1]
        positions = torch.arange(allocated_frames, device=spkcache.device).unsqueeze(0)
        positions = positions.expand(spkcache.shape[0], -1)

        fifo_start = spkcache_lengths.unsqueeze(1)
        chunk_start = (spkcache_lengths + fifo_lengths).unsqueeze(1)
        valid_end = total_lengths.unsqueeze(1)

        spkcache_mask = positions < fifo_start
        fifo_mask = (positions >= fifo_start) & (positions < chunk_start)
        chunk_mask = (positions >= chunk_start) & (positions < valid_end)

        emb_dim = spkcache.shape[2]

        def gather_local(source, local_positions):
            local_positions = local_positions.clamp(min=0, max=source.shape[1] - 1)
            indices = local_positions.unsqueeze(2).expand(-1, -1, emb_dim)
            return torch.gather(source, dim=1, index=indices)

        spkcache_values = gather_local(spkcache, positions)
        fifo_values = gather_local(fifo, positions - fifo_start)
        chunk_values = gather_local(chunk_embs, positions - chunk_start)

        output = spkcache_values * spkcache_mask.unsqueeze(2)
        output = output + fifo_values * fifo_mask.unsqueeze(2)
        output = output + chunk_values * chunk_mask.unsqueeze(2)
        return output, total_lengths

    @staticmethod
    def export_rope_attention(attention, hidden_states, lengths):
        """ONNX-friendly equivalent of the new encoder's FlexAttention path."""
        batch_size, num_frames, _ = hidden_states.shape
        num_heads, head_dim = attention.n_heads, attention.head_dim
        qkv = (
            attention.w_qkv(hidden_states)
            .view(batch_size, num_frames, 3, num_heads, head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.unbind(0)
        if attention.qk_norm:
            query = attention.q_norm(query).to(value.dtype)
            key = attention.k_norm(key).to(value.dtype)
        query, key = attention.rope(query, key)
        valid_keys = torch.arange(num_frames, device=hidden_states.device).view(1, 1, 1, -1)
        valid_keys = valid_keys < lengths.view(-1, 1, 1, 1)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=valid_keys,
            dropout_p=0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, num_frames, attention.d_model
        )
        return attention.out_proj(attended)

    def export_flex_frontend_encoder(self, combined_embs, combined_lengths):
        """Lower the new RoPE/FlexAttention encoder to standard ONNX operators."""
        encoder = self.encoder
        if getattr(encoder, "self_attention_model", None) != "rope":
            raise ValueError(
                "The export-only FlexAttention lowering currently supports the "
                "new TransformerEncoder with self_attention_model='rope'."
            )
        hidden_states = encoder.dropout_pre_encoder(combined_embs)
        hidden_states = encoder.embed_norm(hidden_states)
        for layer in encoder.layers:
            attended = self.export_rope_attention(
                layer.attn,
                layer.norm1(hidden_states),
                combined_lengths,
            )
            hidden_states = hidden_states + layer.drop(attended)
            hidden_states = hidden_states + layer.drop(
                layer.ffn(layer.norm2(hidden_states))
            )
        hidden_states = encoder.final_norm(hidden_states)
        if encoder.out_proj is not None:
            hidden_states = encoder.out_proj(hidden_states)
        return hidden_states, combined_lengths

    def uses_flex_frontend_encoder(self):
        return (
            self.encoder.__class__.__module__
            == "nemo.collections.asr.modules.transformer_encoder"
        )

    def forward_for_export(
        self,
        chunk,
        chunk_lengths,
        spkcache,
        spkcache_lengths,
        fifo,
        fifo_lengths,
    ):
        # The recovered graph begins at encoder.pre_encode.  Audio-to-Mel feature
        # extraction is performed by the surrounding Riva pipeline.
        chunk_embs, chunk_emb_lengths = self._call_pre_encode(chunk, chunk_lengths)
        chunk_emb_lengths = chunk_emb_lengths.to(torch.int64)

        combined_embs, combined_lengths = self.concat_and_pad(
            [spkcache, fifo, chunk_embs],
            [spkcache_lengths, fifo_lengths, chunk_emb_lengths],
        )

        if self.uses_flex_frontend_encoder():
            encoded, pred_lengths = self.export_flex_frontend_encoder(
                combined_embs, combined_lengths
            )
            if self.sortformer_modules.encoder_proj is not None:
                encoded = self.sortformer_modules.encoder_proj(encoded)
        else:
            encoded, pred_lengths = self.frontend_encoder(
                processed_signal=combined_embs,
                processed_signal_length=combined_lengths,
                bypass_pre_encode=True,
            )
        # The recovered graph deliberately omits the final
        # ``predictions * length_mask`` operation. Consumers truncate with the
        # explicit ``pred_lengths`` output instead, and the graph ends directly
        # at Sigmoid.
        encoder_mask = self.sortformer_modules.length_to_mask(pred_lengths, encoded.shape[1])
        transformed = self.transformer_encoder(encoder_states=encoded, encoder_mask=encoder_mask)
        if self.high_resolution:
            # Riva pads cache/FIFO allocations beyond their valid lengths. The
            # kernel-3 upsampler would otherwise observe non-zero padded query
            # states at the right boundary of the valid prefix.
            transformed = transformed * encoder_mask.unsqueeze(-1)
            transformed = self.sortformer_modules.upsample_hidden(transformed)
            predictions = self.sortformer_modules.forward_speaker_sigmoids(transformed)
            predictions = self.sortformer_modules.downsample_preds(
                predictions, self.upsample_factor
            )
        else:
            predictions = self.sortformer_modules.forward_speaker_sigmoids(transformed)

        return predictions, pred_lengths, chunk_embs, chunk_emb_lengths


def round_model_tensors_through_bf16(model: torch.nn.Module) -> None:
    """Round every floating parameter and buffer through BF16, retaining FP32 storage."""
    with torch.no_grad():
        for tensor in list(model.parameters()) + list(model.buffers()):
            if tensor.is_floating_point():
                tensor.copy_(tensor.to(torch.bfloat16).to(torch.float32))


def promote_constant_nodes_to_initializers(onnx_path: Path) -> None:
    """Replace tensor-valued Constant nodes with graph initializers.

    The recovered graph has no Constant operators and stores all of their values
    as initializers. This transformation preserves graph semantics while matching
    the layout consumed by the legacy TensorRT conversion path.
    """
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=True)
    retained_nodes = []
    for node in model.graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            retained_nodes.append(node)
            continue
        value_attributes = [attribute for attribute in node.attribute if attribute.name == "value"]
        if len(value_attributes) != 1 or not value_attributes[0].HasField("t"):
            retained_nodes.append(node)
            continue
        tensor = value_attributes[0].t
        tensor.name = node.output[0]
        model.graph.initializer.append(tensor)

    del model.graph.node[:]
    model.graph.node.extend(retained_nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))


def make_input_example(model: SortformerEncLabelModel):
    batch_size = 1
    chunk_frames = 16
    feature_dim = int(model.cfg.preprocessor.features)
    embedding_dim = int(model.cfg.model_defaults.fc_d_model)
    cache_frames = min(int(model.cfg.sortformer_modules.spkcache_len), 8)
    fifo_frames = 8

    # Trace non-empty cache/FIFO tensors and make their time dimensions dynamic.
    # Small allocated sizes keep the 31-layer export fast; TensorRT profiles set
    # the actual runtime maxima.
    chunk = torch.rand(batch_size, chunk_frames, feature_dim, device=model.device)
    chunk_lengths = torch.tensor(
        [chunk_frames],
        dtype=torch.int64,
        device=model.device,
    )
    spkcache = torch.randn(batch_size, cache_frames, embedding_dim, device=model.device)
    spkcache_lengths = torch.tensor(
        [max(1, cache_frames // 2)],
        dtype=torch.int64,
        device=model.device,
    )
    fifo = torch.randn(batch_size, fifo_frames, embedding_dim, device=model.device)
    fifo_lengths = torch.tensor(
        [3],
        dtype=torch.int64,
        device=model.device,
    )
    return chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("nemo_model", type=Path, help="Source streaming Sortformer .nemo model")
    parser.add_argument("output_onnx", type=Path, help="Destination ONNX path")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device used while exporting (default: cpu)",
    )
    parser.add_argument(
        "--no-bf16-roundtrip",
        action="store_true",
        help="Keep original FP32 checkpoint values instead of matching the recovered artifact",
    )
    parser.add_argument(
        "--keep-constant-nodes",
        action="store_true",
        help="Skip the recovered Constant-node to initializer post-processing pass",
    )
    parser.add_argument(
        "--skip-native-validation",
        action="store_true",
        help="Skip the new-checkpoint native versus export-lowered numerical check",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = SortformerEncLabelModel.restore_from(
        restore_path=str(args.nemo_model),
        map_location=args.device,
    )
    model.eval().float()

    if not args.no_bf16_roundtrip:
        round_model_tensors_through_bf16(model)

    input_example = make_input_example(model)
    native_predictions = None
    if model.high_resolution and not args.skip_native_validation:
        with torch.no_grad():
            native_predictions = model.forward_for_export(*input_example)[0]

    # NeMo's exporter temporarily replaces the class-level forward method with
    # forward_for_export.  Changing the instance class preserves that mechanism
    # and the standard _prepare_for_export() transformations, including Conv-BN
    # fusion, while supplying the recovered Riva four-output contract.
    model.__class__ = RivaStreamingExportModel
    with open_dict(model.cfg):
        model.cfg.precision = "bf16_mixed"

    if native_predictions is not None:
        with torch.no_grad():
            exported_predictions = model.forward_for_export(*input_example)[0]
        exported_predictions = exported_predictions[:, : native_predictions.shape[1]]
        max_abs_error = (native_predictions - exported_predictions).abs().max().item()
        print(f"native_vs_export_max_abs={max_abs_error:.9g}")
        if max_abs_error > 1.0e-5:
            raise RuntimeError(
                "Export-only attention lowering failed numerical validation: "
                f"max absolute error {max_abs_error} exceeds 1e-5"
            )

    args.output_onnx.parent.mkdir(parents=True, exist_ok=True)
    model.export(
        output=str(args.output_onnx),
        input_example=input_example,
        onnx_opset_version=16,
        do_constant_folding=True,
        dynamic_axes=DYNAMIC_AXES,
        check_trace=False,
        use_dynamo=False,
    )
    if not args.keep_constant_nodes:
        promote_constant_nodes_to_initializers(args.output_onnx)


if __name__ == "__main__":
    main()
