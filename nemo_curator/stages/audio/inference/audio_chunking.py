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

"""Zero-overlap waveform chunking for offline ASR inference."""

from __future__ import annotations

import math

import numpy as np

_MINIMUM_CHUNK_DURATION_SEC = 0.1


def model_training_max_duration(model: object) -> float:
    """Read the training audio upper bound from a loaded NeMo model."""
    train_ds = getattr(getattr(model, "cfg", None), "train_ds", None)
    max_duration = getattr(train_ds, "max_duration", None)
    try:
        duration = float(max_duration)
    except (TypeError, ValueError) as error:
        msg = "Loaded NeMo model does not define train_ds.max_duration"
        raise ValueError(msg) from error
    if not math.isfinite(duration) or duration <= 0:
        msg = f"Loaded NeMo model has invalid train_ds.max_duration: {max_duration!r}"
        raise ValueError(msg)
    return duration


def model_chunk_duration(model: object, max_feature_frames: int | None = None) -> float:
    """Return the model-trained window, capped by an optional encoder input shape."""
    training_duration = model_training_max_duration(model)
    if max_feature_frames is None:
        return training_duration

    preprocessor = getattr(getattr(model, "cfg", None), "preprocessor", None)
    try:
        sample_rate = int(preprocessor.sample_rate)
        window_stride = float(preprocessor.window_stride)
    except (AttributeError, TypeError, ValueError) as error:
        msg = "Loaded NeMo model does not define a valid preprocessor sample rate and window stride"
        raise ValueError(msg) from error
    hop_samples = round(sample_rate * window_stride)
    if sample_rate <= 0 or hop_samples <= 0 or max_feature_frames < 1:
        msg = "Cannot derive a safe audio window from the model preprocessor and encoder shape"
        raise ValueError(msg)
    engine_duration = (max_feature_frames * hop_samples - 1) / sample_rate
    return min(training_duration, engine_duration)


def split_waveforms(
    waveforms: list[np.ndarray],
    sample_rates: list[int],
    max_duration_sec: float,
) -> tuple[list[np.ndarray], list[int], list[int]]:
    """Split time-first waveforms into consecutive chunks and return their input owners."""
    chunks: list[np.ndarray] = []
    chunk_sample_rates: list[int] = []
    owners: list[int] = []
    for owner, (waveform, sample_rate) in enumerate(zip(waveforms, sample_rates, strict=True)):
        arr = np.asarray(waveform)
        if arr.size == 0:
            continue
        if arr.ndim == 0:
            msg = "Audio waveform must have a time dimension"
            raise ValueError(msg)
        rate = int(sample_rate)
        if rate <= 0:
            msg = f"Audio sample rate must be positive, got {sample_rate!r}"
            raise ValueError(msg)
        chunk_samples = max(1, int(max_duration_sec * rate))
        minimum_chunk_samples = min(chunk_samples, max(1, round(_MINIMUM_CHUNK_DURATION_SEC * rate)))
        for start in range(0, arr.shape[0], chunk_samples):
            chunk = arr[start : start + chunk_samples]
            if chunk.shape[0] < minimum_chunk_samples:
                chunk = np.pad(chunk, (0, minimum_chunk_samples - chunk.shape[0]))
            chunks.append(chunk)
            chunk_sample_rates.append(rate)
            owners.append(owner)
    return chunks, chunk_sample_rates, owners


def merge_chunk_texts(chunk_texts: list[str], owners: list[int], num_inputs: int) -> list[str]:
    """Join ordered, non-empty chunk transcripts for each original input."""
    grouped: list[list[str]] = [[] for _ in range(num_inputs)]
    for text, owner in zip(chunk_texts, owners, strict=True):
        normalized = text.strip()
        if normalized:
            grouped[owner].append(normalized)
    return [" ".join(parts) for parts in grouped]
