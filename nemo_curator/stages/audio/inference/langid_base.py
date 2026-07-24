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

"""Shared base for spoken-language-identification stages.

Holds the waveform normalization / resampling logic common to the AmberNet and
SpeechBrain LangID stages so the two only differ in model loading and inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class BaseLangIDStage(ProcessingStage[AudioTask, AudioTask]):
    """Common config + waveform preprocessing for LangID stages.

    Subclasses provide model loading (``setup``/``teardown``) and a
    ``process_batch`` that runs inference on prepared waveforms.

    Args:
        target_sr: Target sample rate for the model (default 16000).
        waveform_key: Task data key for the audio waveform.
        sample_rate_key: Task data key for the sample rate.
        output_key: Task data key to write the predicted language.
        confidence_key: Task data key to write prediction confidence.
        min_duration_sec: Minimum segment duration for LangID (skip shorter).
        max_duration_sec: Max seconds fed to the model; longer segments are truncated.
            Language ID needs only a few seconds of speech (VoxLingua107 is trained on
            short clips), and batches are padded to their longest row, so a single 40 s
            VAD segment otherwise inflates the whole batch's conv activations -> OOM.
            Truncating to ~10 s cuts peak GPU memory ~3-4x with no accuracy cost.
        batch_size: Number of segments to process at once.
        max_workers: Hard cap on concurrent actors for this stage (None = executor
            autoscales). LangID is cheap per row but the autoscaler will happily spin up
            ~10 actors on one 80 GB GPU (gpu_memory_gb=8 -> 0.1 GPU each), each loading a
            full model + a padded-batch activation spike, which collectively OOMs a GPU
            shared with Canary/Sortformer. Capping the pool is the primary OOM guard.
    """

    target_sr: int = 16000
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    output_key: str = "language"
    confidence_key: str = "language_confidence"
    min_duration_sec: float = 1.0
    max_duration_sec: float = 10.0
    batch_size: int = 16
    max_workers: int | None = None
    resources: Resources = field(default_factory=lambda: Resources(gpu_memory_gb=4.0))

    _resamplers: dict[tuple[int, int], Any] = field(default_factory=dict, init=False, repr=False)

    def num_workers(self) -> int | None:
        """Cap concurrent actors when ``max_workers`` is set (else defer to the executor)."""
        return self.max_workers

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.waveform_key, self.sample_rate_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.output_key, self.confidence_key]

    def _set_empty(self, task: AudioTask) -> None:
        task.data[self.output_key] = ""
        task.data[self.confidence_key] = 0.0

    def _resample(self, waveform: np.ndarray, sr: int) -> torch.Tensor:
        """Resample a 1-D float32 array to ``target_sr`` using a cached transform."""
        audio = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))
        if sr != self.target_sr:
            import torchaudio

            key = (sr, self.target_sr)
            resampler = self._resamplers.get(key)
            if resampler is None:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
                self._resamplers[key] = resampler
            audio = resampler(audio)
        return audio

    def _prepare_audio(self, task: AudioTask) -> torch.Tensor | None:
        """Normalize a task's waveform to a 1-D 16kHz tensor, or None if it should be skipped.

        Skipped tasks (missing/empty/too-short waveform) get empty results written in place.
        """
        waveform = task.data.get(self.waveform_key)
        sr = task.data.get(self.sample_rate_key, self.target_sr)

        if waveform is None:
            self._set_empty(task)
            return None

        if isinstance(waveform, torch.Tensor):
            waveform = waveform.squeeze().cpu().numpy()
        else:
            waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.squeeze()
        if waveform.size == 0:
            self._set_empty(task)
            return None

        if len(waveform) / sr < self.min_duration_sec:
            self._set_empty(task)
            return None

        audio = self._resample(waveform, sr)

        # Truncate to max_duration_sec of audio at the model sample rate. LangID needs only
        # a few seconds; keeping full (up to 40 s) segments is what inflates the padded batch
        # and drives the SpeechBrain CNN OOM. Bounded here so every batch row is <= this cap.
        if self.max_duration_sec and self.max_duration_sec > 0:
            max_samples = int(self.max_duration_sec * self.target_sr)
            if audio.shape[-1] > max_samples:
                audio = audio[..., :max_samples]
        return audio

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]
