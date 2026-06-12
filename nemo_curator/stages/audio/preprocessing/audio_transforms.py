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

"""Lightweight in-memory audio transform stages.

These stages operate on waveforms already loaded in task.data (numpy arrays)
and are typically placed early in a pipeline after a reader stage.

- MonoDownsampleStage: convert to mono and resample to a target sample rate.
- SqueezeWaveformStage: squeeze (1, N) waveforms to (N,) for downstream compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


@dataclass
class MonoDownsampleStage(ProcessingStage[AudioTask, AudioTask]):
    """Convert in-memory waveform to mono and resample to target sample rate.

    Unlike ``MonoConversionStage`` (which reads from files and rejects
    mismatched sample rates), this stage operates on waveforms already
    loaded in ``task.data`` and actively resamples to the desired rate.
    Typically placed after a reader stage and before VAD.

    Args:
        target_sample_rate: Desired output sample rate in Hz.
        waveform_key: Task data key for the audio waveform array.
        sample_rate_key: Task data key for the integer sample rate.
    """

    name: str = "MonoDownsample"
    target_sample_rate: int = 16000
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.waveform_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.waveform_key, self.sample_rate_key]

    def process(self, task: AudioTask) -> AudioTask:
        wav = task.data.get(self.waveform_key)
        sr = task.data.get(self.sample_rate_key, self.target_sample_rate)

        if wav is None:
            return task

        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=0)

        if sr != self.target_sample_rate:
            import librosa

            task.data["original_sampling_rate"] = sr
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.target_sample_rate)

        task.data[self.waveform_key] = wav
        task.data[self.sample_rate_key] = self.target_sample_rate
        task.data["sampling_rate"] = self.target_sample_rate
        return task

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        return [self.process(task) for task in tasks]


@dataclass
class SqueezeWaveformStage(ProcessingStage[AudioTask, AudioTask]):
    """Squeeze (1, N) waveforms to (N,) for downstream compatibility.

    VAD outputs waveform with shape (1, N) from .unsqueeze(0), but
    downstream stages (SED, LangID) expect 1-D (N,) arrays.

    Args:
        waveform_key: Task data key for the audio waveform array.
    """

    name: str = "SqueezeWaveform"
    waveform_key: str = "waveform"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.waveform_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.waveform_key]

    def process(self, task: AudioTask) -> AudioTask:
        wav = task.data.get(self.waveform_key)
        if wav is not None:
            wav = np.asarray(wav)
            if wav.ndim > 1:
                wav = wav.squeeze()
            task.data[self.waveform_key] = wav
        return task

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        return [self.process(task) for task in tasks]
