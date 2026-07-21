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

"""Build an FP16 TensorRT encoder bundle for an Indic Parakeet RNN-T model."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _load_model(model_path: Path) -> torch.nn.Module:
    import nemo.collections.asr as nemo_asr
    import torch

    if not torch.cuda.is_available():
        msg = "Building an Indic Parakeet TensorRT engine requires CUDA"
        raise RuntimeError(msg)
    model = nemo_asr.models.ASRModel.restore_from(
        restore_path=str(model_path),
        map_location=torch.device("cuda"),
    )
    if not hasattr(model, "encoder") or not hasattr(model, "decoder") or not hasattr(model, "joint"):
        msg = "The supplied NeMo checkpoint is not an RNN-T encoder-decoder model"
        raise TypeError(msg)
    model.eval().to(dtype=torch.float16)
    return model


def _export_encoder(
    model: torch.nn.Module,
    onnx_path: Path,
    *,
    feature_count: int,
    example_frames: int,
) -> None:
    import torch

    audio_signal = torch.randn(
        (1, feature_count, example_frames),
        device="cuda",
        dtype=torch.float16,
    )
    length = torch.full((1,), example_frames, device="cuda", dtype=torch.int64)
    model.encoder.export(
        str(onnx_path),
        input_example=(audio_signal, length),
        do_constant_folding=False,
        onnx_opset_version=17,
        check_trace=False,
        dynamic_axes={
            "audio_signal": {0: "batch", 2: "feature_frames"},
            "length": {0: "batch"},
            "outputs": {0: "batch", 2: "encoded_frames"},
            "encoded_lengths": {0: "batch"},
        },
        use_dynamo=False,
    )


def _build_engine(
    onnx_path: Path,
    engine_path: Path,
    *,
    feature_count: int,
    args: argparse.Namespace,
) -> str:
    try:
        import tensorrt as trt
    except ImportError as error:
        msg = "TensorRT Python bindings are required to build the Indic Parakeet engine"
        raise RuntimeError(msg) from error

    logger = trt.Logger(trt.Logger.INFO if args.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        msg = f"Failed to parse {onnx_path}:\n{errors}"
        raise RuntimeError(msg)

    input_names = {network.get_input(index).name for index in range(network.num_inputs)}
    if input_names != {"audio_signal", "length"}:
        msg = f"Unexpected exported encoder inputs: {sorted(input_names)}"
        raise RuntimeError(msg)
    output_names = {network.get_output(index).name for index in range(network.num_outputs)}
    if output_names != {"outputs", "encoded_lengths"}:
        msg = f"Unexpected exported encoder outputs: {sorted(output_names)}"
        raise RuntimeError(msg)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gb * (1 << 30))
    if not builder.platform_has_fast_fp16:
        msg = "This GPU does not provide fast FP16 TensorRT kernels"
        raise RuntimeError(msg)
    config.set_flag(trt.BuilderFlag.FP16)
    config.builder_optimization_level = 5

    profile = builder.create_optimization_profile()
    audio_profile_status = profile.set_shape(
        "audio_signal",
        (args.min_batch, feature_count, args.min_frames),
        (args.opt_batch, feature_count, args.opt_frames),
        (args.max_batch, feature_count, args.max_frames),
    )
    length_profile_status = profile.set_shape(
        "length",
        (args.min_batch,),
        (args.opt_batch,),
        (args.max_batch,),
    )
    if audio_profile_status is False or length_profile_status is False:
        msg = "Could not set the TensorRT optimization profile"
        raise RuntimeError(msg)
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        msg = "TensorRT failed to build the Indic Parakeet encoder engine"
        raise RuntimeError(msg)
    engine_path.write_bytes(serialized_engine)
    return trt.__version__


def _validate_engine(
    model: torch.nn.Module,
    engine_path: Path,
    *,
    feature_count: int,
    min_frames: int,
    opt_frames: int,
) -> None:
    import torch

    from nemo_curator.stages.audio.inference.indic_parakeet_rnnt_tensorrt import (
        TensorRTParakeetEncoderSession,
    )

    validation_frames = max(min_frames, min(opt_frames, 256))
    generator = torch.Generator(device="cuda").manual_seed(0)
    audio_signal = torch.randn(
        (1, feature_count, validation_frames),
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    length = torch.full((1,), validation_frames, device="cuda", dtype=torch.int64)
    with torch.inference_mode():
        expected_outputs, expected_lengths = model.encoder(audio_signal=audio_signal, length=length)

    session = TensorRTParakeetEncoderSession(engine_path)
    try:
        actual = session.infer({"audio_signal": audio_signal, "length": length})
        torch.testing.assert_close(actual["outputs"], expected_outputs, rtol=5e-2, atol=5e-2)
        torch.testing.assert_close(actual["encoded_lengths"], expected_lengths)
    finally:
        session.close()
    print("INDIC_PARAKEET_RNNT_TENSORRT_ENCODER_PARITY_PASSED")


def build_bundle(args: argparse.Namespace) -> None:
    model_path = Path(args.model).resolve()
    if not model_path.is_file() or model_path.suffix != ".nemo":
        msg = f"Local NeMo checkpoint not found: {model_path}"
        raise FileNotFoundError(msg)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = [output_dir / name for name in ("encoder.plan", "model.nemo", "metadata.json")]
    existing = [str(path) for path in destinations if path.exists()]
    if existing:
        msg = f"Refusing to overwrite existing bundle artifacts: {existing}"
        raise FileExistsError(msg)

    model = _load_model(model_path)
    feature_count = int(getattr(model.encoder, "_feat_in", model.cfg.encoder.feat_in))
    subsampling_factor = int(model.encoder.subsampling_factor)
    sample_rate = int(model.cfg.preprocessor.sample_rate)
    vocabulary_size = int(getattr(model.joint, "_vocab_size", model.cfg.joint.num_classes))

    engine_path = output_dir / "encoder.plan"
    with tempfile.TemporaryDirectory(prefix=".indic-parakeet-rnnt-", dir=output_dir) as temporary_dir:
        onnx_path = Path(temporary_dir) / "encoder.onnx"
        _export_encoder(
            model,
            onnx_path,
            feature_count=feature_count,
            example_frames=args.min_frames,
        )
        tensorrt_version = _build_engine(
            onnx_path,
            engine_path,
            feature_count=feature_count,
            args=args,
        )

    _validate_engine(
        model,
        engine_path,
        feature_count=feature_count,
        min_frames=args.min_frames,
        opt_frames=args.opt_frames,
    )
    shutil.copy2(model_path, output_dir / "model.nemo")
    metadata = {
        "schema_version": 1,
        "model_type": "indic_parakeet_rnnt",
        "precision": "fp16",
        "engine_file": "encoder.plan",
        "model_file": "model.nemo",
        "source_model": model_path.name,
        "sample_rate": sample_rate,
        "feature_count": feature_count,
        "subsampling_factor": subsampling_factor,
        "vocabulary_size": vocabulary_size,
        "max_symbols_per_step": args.max_symbols_per_step,
        "input_names": ["audio_signal", "length"],
        "output_names": ["outputs", "encoded_lengths"],
        "profile": {
            "min": {"batch": args.min_batch, "feature_frames": args.min_frames},
            "opt": {"batch": args.opt_batch, "feature_frames": args.opt_frames},
            "max": {"batch": args.max_batch, "feature_frames": args.max_frames},
        },
        "onnx_opset": 17,
        "tensorrt_version": tensorrt_version,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote Indic Parakeet RNN-T TensorRT bundle to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to the Indic Parakeet RNN-T .nemo checkpoint")
    parser.add_argument("--output-dir", required=True, help="Destination engine bundle directory")
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=8)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--min-frames", type=int, default=8, help="Minimum input feature frames")
    parser.add_argument("--opt-frames", type=int, default=800, help="Optimization input feature frames")
    parser.add_argument("--max-frames", type=int, default=3000, help="Maximum input feature frames")
    parser.add_argument("--max-symbols-per-step", type=int, default=10)
    parser.add_argument("--workspace-gb", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.min_batch <= args.opt_batch <= args.max_batch:
        parser.error("batch profile must satisfy 1 <= min-batch <= opt-batch <= max-batch")
    if not 1 <= args.min_frames <= args.opt_frames <= args.max_frames:
        parser.error("frame profile must satisfy 1 <= min-frames <= opt-frames <= max-frames")
    if args.max_symbols_per_step < 1:
        parser.error("max-symbols-per-step must be at least 1")
    if args.workspace_gb < 1:
        parser.error("workspace-gb must be at least 1")
    return args


if __name__ == "__main__":
    build_bundle(parse_args())
