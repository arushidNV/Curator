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

"""TensorRT bundle validation for AI4Bharat IndicConformer encoders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ENGINE_FILENAME = "encoder.plan"
METADATA_FILENAME = "metadata.json"
MODEL_FILENAME = "model.nemo"
_INPUT_NAMES = {"audio_signal", "length"}
_OUTPUT_NAMES = {"outputs", "encoded_lengths"}


def load_engine_metadata(engine_dir: str | Path) -> dict[str, Any]:  # noqa: C901
    """Load and validate an IndicConformer TensorRT bundle manifest."""
    path = Path(engine_dir) / METADATA_FILENAME
    if not path.is_file():
        msg = f"TensorRT engine metadata not found: {path}"
        raise FileNotFoundError(msg)
    try:
        metadata = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        msg = f"Could not read TensorRT engine metadata: {path}"
        raise ValueError(msg) from error

    if metadata.get("schema_version") != 1:
        msg = f"Unsupported TensorRT engine metadata schema: {metadata.get('schema_version')!r}"
        raise ValueError(msg)
    if metadata.get("model_type") != "indic_conformer_hybrid":
        msg = f"Unsupported TensorRT engine model type: {metadata.get('model_type')!r}"
        raise ValueError(msg)
    if metadata.get("precision") != "fp16":
        msg = f"Unsupported TensorRT engine precision: {metadata.get('precision')!r}"
        raise ValueError(msg)
    if metadata.get("engine_file") != ENGINE_FILENAME or metadata.get("model_file") != MODEL_FILENAME:
        msg = "TensorRT engine metadata does not describe the expected bundle filenames"
        raise ValueError(msg)
    if set(metadata.get("input_names", [])) != _INPUT_NAMES:
        msg = f"Unexpected TensorRT encoder inputs: {metadata.get('input_names')!r}"
        raise ValueError(msg)
    if set(metadata.get("output_names", [])) != _OUTPUT_NAMES:
        msg = f"Unexpected TensorRT encoder outputs: {metadata.get('output_names')!r}"
        raise ValueError(msg)
    for key in ("sample_rate", "feature_count", "subsampling_factor", "encoder_dim"):
        if not isinstance(metadata.get(key), int) or metadata[key] < 1:
            msg = f"Invalid TensorRT engine metadata value for {key}: {metadata.get(key)!r}"
            raise ValueError(msg)
    profile = metadata.get("profile")
    if not isinstance(profile, dict) or not isinstance(profile.get("max"), dict):
        msg = f"Invalid TensorRT engine profile: {profile!r}"
        raise ValueError(msg)  # noqa: TRY004
    max_batch = profile["max"].get("batch")
    min_frames = profile.get("min", {}).get("feature_frames")
    if not isinstance(max_batch, int) or max_batch < 1 or not isinstance(min_frames, int) or min_frames < 1:
        msg = f"Invalid TensorRT engine profile: {profile!r}"
        raise ValueError(msg)
    return metadata
