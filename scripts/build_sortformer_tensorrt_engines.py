# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ruff: noqa: INP001

"""Export and build high-resolution Sortformer TensorRT engines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import tensorrt as trt
import torch
from nemo.collections.asr.models import SortformerEncLabelModel


def _explicit_full_attention(
    attention: torch.nn.Module,
    hidden_states: torch.Tensor,
    key_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, frames, _ = hidden_states.shape
    heads, head_dim = attention.n_heads, attention.head_dim
    qkv = attention.w_qkv(hidden_states).view(batch_size, frames, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
    query, key, value = qkv.unbind(0)
    if attention.qk_norm:
        query = attention.q_norm(query).to(value.dtype)
        key = attention.k_norm(key).to(value.dtype)
    query, key = attention.rope(query, key)
    scores = torch.matmul(query, key.transpose(-2, -1)) * (head_dim**-0.5)
    scores = scores.masked_fill(~key_mask[:, None, None, :], torch.finfo(scores.dtype).min)
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.matmul(probabilities, value)
    output = output.transpose(1, 2).contiguous().view(batch_size, frames, attention.d_model)
    return attention.out_proj(output)


class _StreamingCore(torch.nn.Module):
    def __init__(self, model: SortformerEncLabelModel, phase: str) -> None:
        super().__init__()
        if phase not in {"cold", "steady"}:
            raise ValueError(phase)
        if model.encoder.attn_mode != "full" or model.encoder.self_attention_model != "rope":
            msg = "Sortformer TensorRT export requires full RoPE attention"
            raise RuntimeError(msg)
        self.model = model
        self.phase = phase

    def _encode(self, embeddings: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoder = self.model.encoder
        hidden = embeddings * encoder.xscale if encoder.xscale else embeddings
        hidden = encoder.embed_norm(encoder.dropout_pre_encoder(hidden))
        key_mask = torch.arange(hidden.shape[1], device=hidden.device)[None, :] < lengths[:, None]
        for layer in encoder.layers:
            attention_input = layer.norm1(hidden)
            attention_output = _explicit_full_attention(layer.attn, attention_input, key_mask)
            hidden = hidden + layer.drop(attention_output)
            hidden = hidden + layer.drop(layer.ffn(layer.norm2(hidden)))
        hidden = encoder.final_norm(hidden)
        if encoder.out_proj is not None:
            hidden = encoder.out_proj(hidden)
        projection = self.model.sortformer_modules.encoder_proj
        if projection is not None:
            hidden = projection(hidden)
        return hidden, lengths.to(torch.int64)

    def forward(
        self,
        chunk: torch.Tensor,
        chunk_lengths: torch.Tensor,
        spkcache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        chunk_embeddings, embedding_lengths = self.model._call_pre_encode(chunk, chunk_lengths)
        embedding_lengths = embedding_lengths.to(torch.int64)
        if self.phase == "cold":
            packed, packed_lengths = chunk_embeddings, embedding_lengths
        else:
            packed = torch.cat((spkcache, chunk_embeddings), dim=1)
            packed_lengths = embedding_lengths + spkcache.shape[1]
        encoded, encoded_lengths = self._encode(packed, packed_lengths)
        high_resolution = self.model.forward_infer(encoded, encoded_lengths)
        cache_resolution = self.model.sortformer_modules.downsample_preds(
            high_resolution,
            self.model.upsample_factor,
        )
        return cache_resolution, high_resolution, chunk_embeddings, embedding_lengths


def _native_outputs(
    model: SortformerEncLabelModel,
    phase: str,
    inputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    chunk, chunk_lengths, *state = inputs
    chunk_embeddings, embedding_lengths = model._call_pre_encode(chunk, chunk_lengths)
    embedding_lengths = embedding_lengths.to(torch.int64)
    if phase == "steady":
        packed = torch.cat((state[0], chunk_embeddings), dim=1)
        packed_lengths = embedding_lengths + state[0].shape[1]
    else:
        packed, packed_lengths = chunk_embeddings, embedding_lengths
    encoded, encoded_lengths = model.frontend_encoder(packed, packed_lengths, bypass_pre_encode=True)
    high_resolution = model.forward_infer(encoded, encoded_lengths)
    cache_resolution = model.sortformer_modules.downsample_preds(
        high_resolution,
        model.upsample_factor,
    )
    return cache_resolution, high_resolution, chunk_embeddings, embedding_lengths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workspace-gb", type=int, default=16)
    return parser.parse_args()


def _export(
    model: SortformerEncLabelModel,
    phase: str,
    output_dir: Path,
    batch_size: int,
    dimensions: dict[str, int],
) -> Path:
    chunk = torch.randn(
        batch_size,
        dimensions["feature_frames"],
        dimensions["feature_bins"],
        device="cuda",
    )
    lengths = torch.full(
        (batch_size,),
        dimensions["feature_frames"],
        dtype=torch.int64,
        device="cuda",
    )
    inputs = (chunk, lengths)
    input_names = ["chunk", "chunk_lengths"]
    if phase == "steady":
        cache = torch.randn(
            batch_size,
            dimensions["cache_frames"],
            dimensions["embedding_dim"],
            device="cuda",
        )
        inputs += (cache,)
        input_names.append("spkcache")

    wrapper = _StreamingCore(model, phase).eval()
    output_names = [
        "cache_resolution_preds",
        "high_resolution_preds",
        "chunk_embeddings",
        "chunk_embedding_lengths",
    ]
    with torch.inference_mode():
        expected = _native_outputs(model, phase, inputs)
        actual = wrapper(*inputs)
    for name, reference, candidate in zip(output_names, expected, actual, strict=True):
        if not torch.allclose(reference, candidate, atol=2e-3, rtol=5e-4):
            maximum = float((reference.float() - candidate.float()).abs().max())
            msg = f"{phase} explicit-attention parity failed for {name}: max_abs={maximum}"
            raise RuntimeError(msg)

    path = output_dir / f"sortformer_{phase}.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            inputs,
            path,
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            do_constant_folding=True,
        )
    graph = onnx.load(path, load_external_data=True)
    onnx.checker.check_model(graph, full_check=True)
    return path


def _build_engine(
    onnx_path: Path,
    engine_path: Path,
    phase: str,
    args: argparse.Namespace,
) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError(errors)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gb << 30)
    config.builder_optimization_level = 5
    if args.precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        for index in range(network.num_layers):
            layer = network.get_layer(index)
            name = layer.name.lower() if layer.name else ""
            if layer.type == trt.LayerType.SOFTMAX or any(key in name for key in ("layernorm", "norm", "ln")):
                layer.precision = trt.float32
                for output_index in range(layer.num_outputs):
                    layer.set_output_type(output_index, trt.float32)
    else:
        config.clear_flag(trt.BuilderFlag.TF32)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        msg = f"TensorRT failed to build {phase} engine"
        raise RuntimeError(msg)
    engine_path.write_bytes(serialized)


def main() -> None:
    args = _parse_args()
    if args.batch_size != 1:
        msg = "High-resolution Sortformer TensorRT export currently supports batch_size=1"
        raise ValueError(msg)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = SortformerEncLabelModel.restore_from(args.model, map_location="cpu")
    model.eval()
    model.freeze()
    model = model.cuda()
    if not model.high_resolution:
        msg = "The Sortformer checkpoint must have high-resolution output enabled"
        raise RuntimeError(msg)
    if int(model.sortformer_modules.fifo_len) != 0:
        msg = "The high-resolution TensorRT runtime currently requires fifo_len=0"
        raise RuntimeError(msg)

    dimensions = {
        "feature_frames": (
            int(model.sortformer_modules.chunk_len)
            + int(model.sortformer_modules.chunk_left_context)
            + int(model.sortformer_modules.chunk_right_context)
        )
        * int(model.encoder.subsampling_factor),
        "feature_bins": int(model.cfg.encoder.feat_in),
        "cache_frames": int(model.sortformer_modules.spkcache_len),
        "embedding_dim": int(model.sortformer_modules.fc_d_model),
    }
    for phase in ("cold", "steady"):
        onnx_path = _export(model, phase, output_dir, args.batch_size, dimensions)
        engine_path = output_dir / f"sortformer_{phase}_{args.precision}_bs{args.batch_size}.plan"
        _build_engine(onnx_path, engine_path, phase, args)
        print(f"Wrote {engine_path}", flush=True)

    metadata = {
        "model": str(Path(args.model).resolve()),
        "precision": args.precision,
        "batch_size": args.batch_size,
        **dimensions,
    }
    (output_dir / "sortformer_tensorrt.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
