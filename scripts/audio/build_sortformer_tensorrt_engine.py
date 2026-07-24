#!/usr/bin/env python3
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

"""Export an exact Sortformer checkpoint and build its streaming TensorRT engine."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _profile_shape(  # noqa: PLR0913
    name: str,
    shape: Sequence[int],
    batch: int,
    chunk_frames: int,
    spkcache_frames: int,
    fifo_frames: int,
) -> tuple[int, ...]:
    dimensions = list(shape)
    if name in {"chunk", "spkcache", "fifo", "chunk_lengths", "spkcache_lengths", "fifo_lengths"}:
        dimensions[0] = batch

    dynamic_length = {
        "chunk": chunk_frames,
        "spkcache": spkcache_frames,
        "fifo": fifo_frames,
    }.get(name)
    for index, dimension in enumerate(dimensions):
        if dimension != -1:
            continue
        if index == 0:
            dimensions[index] = batch
        elif index == 1 and dynamic_length is not None:
            dimensions[index] = dynamic_length
        else:
            message = f"No profile rule for dynamic dimension {index} of {name!r}: {tuple(shape)}"
            raise ValueError(message)
    return tuple(dimensions)


def _export_onnx(model_path: Path, onnx_path: Path):  # noqa: ANN202
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel

    if not torch.cuda.is_available():
        message = "Sortformer ONNX export requires CUDA in the supported NeMo container"
        raise RuntimeError(message)
    model = SortformerEncLabelModel.restore_from(
        restore_path=str(model_path),
        map_location="cuda",
        strict=False,
    )
    model.eval()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    class StreamingGraph(torch.nn.Module):
        """Exportable neural graph without NeMo's list-based concat_embs()."""

        def __init__(self, sortformer: SortformerEncLabelModel) -> None:
            super().__init__()
            self.sortformer = sortformer

        @staticmethod
        def _gather_frames(source: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            safe_positions = positions.clamp(min=0, max=source.shape[1] - 1)
            gather_index = safe_positions.unsqueeze(-1).expand(-1, -1, source.shape[2])
            return torch.gather(source, dim=1, index=gather_index)

        def _compact_states(  # noqa: PLR0913
            self,
            spkcache: torch.Tensor,
            spkcache_lengths: torch.Tensor,
            fifo: torch.Tensor,
            fifo_lengths: torch.Tensor,
            chunk: torch.Tensor,
            chunk_lengths: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            capacity = spkcache.shape[1] + fifo.shape[1] + chunk.shape[1]
            positions = torch.arange(capacity, dtype=torch.int64, device=chunk.device)
            positions = positions.unsqueeze(0).expand(chunk.shape[0], -1)

            spk_end = spkcache_lengths.unsqueeze(1)
            fifo_end = spk_end + fifo_lengths.unsqueeze(1)
            chunk_end = fifo_end + chunk_lengths.unsqueeze(1)

            spk_values = self._gather_frames(spkcache, positions)
            fifo_values = self._gather_frames(fifo, positions - spk_end)
            chunk_values = self._gather_frames(chunk, positions - fifo_end)
            zeros = torch.zeros_like(spk_values)

            compact = torch.where(
                (positions < spk_end).unsqueeze(-1),
                spk_values,
                torch.where(
                    (positions < fifo_end).unsqueeze(-1),
                    fifo_values,
                    torch.where((positions < chunk_end).unsqueeze(-1), chunk_values, zeros),
                ),
            )
            return compact, chunk_end.squeeze(1)

        def forward(  # noqa: PLR0913
            self,
            chunk: torch.Tensor,
            chunk_lengths: torch.Tensor,
            spkcache: torch.Tensor,
            spkcache_lengths: torch.Tensor,
            fifo: torch.Tensor,
            fifo_lengths: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            chunk_embs, chunk_emb_lengths = self.sortformer.encoder.pre_encode(
                x=chunk,
                lengths=chunk_lengths,
            )
            combined, combined_lengths = self._compact_states(
                spkcache,
                spkcache_lengths,
                fifo,
                fifo_lengths,
                chunk_embs,
                chunk_emb_lengths,
            )
            encoded, encoded_lengths = self.sortformer.frontend_encoder(
                processed_signal=combined,
                processed_signal_length=combined_lengths,
                bypass_pre_encode=True,
            )
            predictions = self.sortformer.forward_infer(
                emb_seq=encoded,
                emb_seq_length=encoded_lengths,
            )
            return predictions, encoded_lengths, chunk_embs, chunk_emb_lengths

    # NeMo 2.7.2 hard-codes 80 mel bins in streaming_input_examples(), while
    # diar_streaming_sortformer_4spk-v2 uses 128. Build the example from the
    # restored checkpoint and export a vectorized replacement for concat_embs.
    modules = model.sortformer_modules
    feature_dim = int(model.cfg.preprocessor.features)
    batch_size = 4
    chunk_frames = 120
    spkcache_frames = max(1, int(modules.spkcache_len))
    fifo_frames = max(1, int(modules.fifo_len))
    input_example = (
        torch.rand((batch_size, chunk_frames, feature_dim), device=model.device),
        torch.full((batch_size,), chunk_frames, dtype=torch.int64, device=model.device),
        torch.randn((batch_size, spkcache_frames, modules.fc_d_model), device=model.device),
        torch.full((batch_size,), spkcache_frames, dtype=torch.int64, device=model.device),
        torch.randn((batch_size, fifo_frames, modules.fc_d_model), device=model.device),
        torch.full((batch_size,), fifo_frames, dtype=torch.int64, device=model.device),
    )
    graph = StreamingGraph(model).eval()
    torch.onnx.export(
        graph,
        input_example,
        str(onnx_path),
        input_names=["chunk", "chunk_lengths", "spkcache", "spkcache_lengths", "fifo", "fifo_lengths"],
        output_names=["predictions", "pred_lengths", "chunk_embs", "chunk_emb_lengths"],
        dynamic_axes={
            "chunk": {0: "batch_size", 1: "chunk_frames"},
            "chunk_lengths": {0: "batch_size"},
            "spkcache": {0: "batch_size", 1: "spkcache_frames"},
            "spkcache_lengths": {0: "batch_size"},
            "fifo": {0: "batch_size", 1: "fifo_frames"},
            "fifo_lengths": {0: "batch_size"},
            "predictions": {0: "batch_size", 1: "output_frames"},
            "pred_lengths": {0: "batch_size"},
            "chunk_embs": {0: "batch_size", 1: "chunk_embedding_frames"},
            "chunk_emb_lengths": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    if not onnx_path.is_file():
        message = f"NeMo did not create the expected ONNX graph: {onnx_path}"
        raise RuntimeError(message)

    import onnx

    exported = onnx.load(str(onnx_path))
    onnx.checker.check_model(exported)
    print(f"SORTFORMER_ONNX_EXPORT_PASSED path={onnx_path}")
    return graph, input_example


def _validate_tensorrt_engine(engine_path: Path, graph, input_example, *, precision: str) -> None:  # noqa: ANN001
    """Execute the built engine and compare all neural graph outputs."""
    import torch

    from nemo_curator.stages.audio.segmentation.silero_tensorrt import TensorRTSession

    names = ["chunk", "chunk_lengths", "spkcache", "spkcache_lengths", "fifo", "fifo_lengths"]
    output_names = ["predictions", "pred_lengths", "chunk_embs", "chunk_emb_lengths"]
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        # Match the builder's true-FP32 configuration. PyTorch leaves cuDNN
        # TF32 enabled independently of matmul precision on H100.
        torch.backends.cudnn.allow_tf32 = precision != "fp32"
        with torch.inference_mode():
            expected = graph(*input_example)
    finally:
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
    session = TensorRTSession(engine_path)
    try:
        actual = session.infer(dict(zip(names, input_example, strict=True)))
        if precision == "fp32":
            rtol, atol = 2e-3, 2e-4
        else:
            rtol, atol = 5e-2, 1e-2
        for name, expected_tensor in zip(output_names, expected, strict=True):
            if expected_tensor.dtype.is_floating_point:
                difference = (actual[name] - expected_tensor).abs()
                print(
                    f"SORTFORMER_TENSORRT_PARITY name={name} "
                    f"max_abs={difference.max().item():.8g} mean_abs={difference.mean().item():.8g}"
                )
                torch.testing.assert_close(actual[name], expected_tensor, rtol=rtol, atol=atol)
            else:
                torch.testing.assert_close(actual[name], expected_tensor)
    finally:
        session.close()
    print(f"SORTFORMER_TENSORRT_PREFLIGHT_PASSED precision={precision} batch=4")


def _build_engine(  # noqa: C901
    onnx_path: Path,
    engine_path: Path,
    args: argparse.Namespace,
) -> tuple[str, list[dict[str, object]]]:
    try:
        import tensorrt as trt
    except ImportError as error:
        message = "TensorRT Python bindings are required to build the Sortformer engine"
        raise RuntimeError(message) from error

    logger = trt.Logger(trt.Logger.INFO if args.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        message = f"Failed to parse {onnx_path}:\n{errors}"
        raise RuntimeError(message)

    expected_inputs = {"chunk", "chunk_lengths", "spkcache", "spkcache_lengths", "fifo", "fifo_lengths"}
    actual_inputs = {network.get_input(index).name for index in range(network.num_inputs)}
    if actual_inputs != expected_inputs:
        message = f"Unexpected Sortformer streaming inputs: {sorted(actual_inputs)}"
        raise RuntimeError(message)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gb * (1 << 30))
    if args.precision == "fp32":
        # TensorRT enables TF32 by default on H100. Diarization predictions are
        # thresholded downstream, so the production FP32 engine must not
        # silently use reduced-mantissa matmuls.
        config.clear_flag(trt.BuilderFlag.TF32)
    elif args.precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif args.precision == "bf16":
        config.set_flag(trt.BuilderFlag.BF16)

    profile = builder.create_optimization_profile()
    profile_rows = []
    profile_args = (
        (args.min_batch, args.min_chunk_frames, 1, 1),
        (args.opt_batch, args.opt_chunk_frames, args.opt_spkcache_frames, args.opt_fifo_frames),
        (args.max_batch, args.max_chunk_frames, args.max_spkcache_frames, args.max_fifo_frames),
    )
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        original_shape = tuple(tensor.shape)
        shapes = []
        for batch, chunk_frames, spkcache_frames, fifo_frames in profile_args:
            shapes.append(
                _profile_shape(
                    tensor.name,
                    original_shape,
                    batch,
                    chunk_frames,
                    spkcache_frames,
                    fifo_frames,
                )
            )
        minimum, optimum, maximum = shapes
        shape_status = profile.set_shape(tensor.name, minimum, optimum, maximum)
        if shape_status is False:
            message = f"Could not set profile for {tensor.name}: {minimum}/{optimum}/{maximum}"
            raise RuntimeError(message)
        profile_rows.append({"name": tensor.name, "min": minimum, "opt": optimum, "max": maximum})
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        message = "TensorRT failed to build the Sortformer engine"
        raise RuntimeError(message)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized_engine)
    return trt.__version__, profile_rows


def build(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    onnx_path = Path(args.onnx)
    engine_path = Path(args.output)
    if not model_path.is_file():
        message = f"Sortformer checkpoint not found: {model_path}"
        raise FileNotFoundError(message)
    if not args.skip_export:
        graph_and_inputs = _export_onnx(model_path, onnx_path)
    elif not onnx_path.is_file():
        message = f"--skip-export was set but ONNX graph does not exist: {onnx_path}"
        raise FileNotFoundError(message)
    else:
        graph_and_inputs = None

    trt_version, profiles = _build_engine(onnx_path, engine_path, args)
    if graph_and_inputs is not None:
        _validate_tensorrt_engine(
            engine_path,
            graph_and_inputs[0],
            graph_and_inputs[1],
            precision=args.precision,
        )
    metadata = {
        "model_path": str(model_path.resolve()),
        "model_sha256": _sha256(model_path),
        "onnx_path": str(onnx_path.resolve()),
        "onnx_sha256": _sha256(onnx_path),
        "engine_path": str(engine_path.resolve()),
        "engine_sha256": _sha256(engine_path),
        "precision": args.precision,
        "tensorrt_version": trt_version,
        "profiles": profiles,
    }
    metadata_path = engine_path.with_suffix(f"{engine_path.suffix}.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Wrote Sortformer TensorRT engine to {engine_path}")
    print(f"Wrote reproducibility metadata to {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Exact .nemo checkpoint used by the Curator stage")
    parser.add_argument("--onnx", required=True, help="Intermediate streaming ONNX graph")
    parser.add_argument("--output", required=True, help="Destination .plan/.engine file")
    parser.add_argument("--skip-export", action="store_true", help="Reuse an existing ONNX graph")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=8)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--min-chunk-frames", type=int, default=1)
    parser.add_argument("--opt-chunk-frames", type=int, default=3048)
    parser.add_argument("--max-chunk-frames", type=int, default=4096)
    parser.add_argument("--opt-spkcache-frames", type=int, default=188)
    parser.add_argument("--max-spkcache-frames", type=int, default=188)
    parser.add_argument("--opt-fifo-frames", type=int, default=40)
    parser.add_argument("--max-fifo-frames", type=int, default=188)
    parser.add_argument("--workspace-gb", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    parsed = parser.parse_args()
    if not 1 <= parsed.min_batch <= parsed.opt_batch <= parsed.max_batch:
        parser.error("batch profile must satisfy 1 <= min-batch <= opt-batch <= max-batch")
    return parsed


if __name__ == "__main__":
    build(parse_args())
