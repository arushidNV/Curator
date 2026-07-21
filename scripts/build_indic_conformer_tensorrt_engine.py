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

"""Build an FP16 TensorRT encoder bundle for an AI4Bharat IndicConformer model."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from tensorrt_encoder_utils import build_engine, export_encoder, validate_engine

if TYPE_CHECKING:
    import torch


def _load_model(model_path: Path) -> torch.nn.Module:
    import nemo.collections.asr as nemo_asr
    import torch

    from nemo_curator.stages.audio.inference.indic_conformer_hybrid import _apply_multisoftmax_patches

    if not torch.cuda.is_available():
        msg = "Building an IndicConformer TensorRT engine requires CUDA"
        raise RuntimeError(msg)
    _apply_multisoftmax_patches()
    model = nemo_asr.models.ASRModel.restore_from(
        restore_path=str(model_path),
        map_location=torch.device("cuda"),
    )
    required_components = ("encoder", "decoder", "joint", "ctc_decoder")
    if any(not hasattr(model, component) for component in required_components):
        msg = "The supplied NeMo checkpoint is not an IndicConformer hybrid model"
        raise TypeError(msg)
    model.eval().to(dtype=torch.float16)
    return model


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
    encoder_dim = int(model.cfg.encoder.d_model)

    engine_path = output_dir / "encoder.plan"
    with tempfile.TemporaryDirectory(prefix=".indic-conformer-", dir=output_dir) as temporary_dir:
        onnx_path = Path(temporary_dir) / "encoder.onnx"
        export_encoder(
            model,
            onnx_path,
            feature_count=feature_count,
            example_frames=args.min_frames,
        )
        tensorrt_version = build_engine(
            onnx_path,
            engine_path,
            feature_count=feature_count,
            args=args,
        )

    validate_engine(
        model,
        engine_path,
        feature_count=feature_count,
        min_frames=args.min_frames,
        opt_frames=args.opt_frames,
    )
    print("INDIC_CONFORMER_TENSORRT_ENCODER_PARITY_PASSED")
    shutil.copy2(model_path, output_dir / "model.nemo")
    metadata = {
        "schema_version": 1,
        "model_type": "indic_conformer_hybrid",
        "precision": "fp16",
        "engine_file": "encoder.plan",
        "model_file": "model.nemo",
        "source_model": model_path.name,
        "sample_rate": sample_rate,
        "feature_count": feature_count,
        "subsampling_factor": subsampling_factor,
        "encoder_dim": encoder_dim,
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
    print(f"Wrote IndicConformer TensorRT bundle to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to the AI4Bharat IndicConformer .nemo checkpoint")
    parser.add_argument("--output-dir", required=True, help="Destination engine bundle directory")
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=8)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--min-frames", type=int, default=8, help="Minimum input feature frames")
    parser.add_argument("--opt-frames", type=int, default=800, help="Optimization input feature frames")
    parser.add_argument("--max-frames", type=int, default=3000, help="Maximum input feature frames")
    parser.add_argument("--workspace-gb", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.min_batch <= args.opt_batch <= args.max_batch:
        parser.error("batch profile must satisfy 1 <= min-batch <= opt-batch <= max-batch")
    if not 1 <= args.min_frames <= args.opt_frames <= args.max_frames:
        parser.error("frame profile must satisfy 1 <= min-frames <= opt-frames <= max-frames")
    if args.workspace_gb < 1:
        parser.error("workspace-gb must be at least 1")
    return args


if __name__ == "__main__":
    build_bundle(parse_args())
