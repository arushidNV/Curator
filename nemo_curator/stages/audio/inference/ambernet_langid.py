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

"""AmberNet Language Identification stage using NeMo's EncDecSpeakerLabelModel.

Identifies the spoken language of each audio segment using the AmberNet model
from NeMo (langid_ambernet). Operates on in-memory waveforms, resamples to
16kHz internally if needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from loguru import logger

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class AmberNetLangIDStage(ProcessingStage[AudioTask, AudioTask]):
    """Language identification using NeMo's AmberNet model.

    Processes audio segments and adds a ``language`` field with the
    predicted language code (e.g., 'en', 'de', 'fr').

    The model resamples to 16kHz internally if the input is at a
    different sample rate.

    Args:
        model_name: NeMo model name for from_pretrained.
        waveform_key: Task data key for the audio waveform.
        sample_rate_key: Task data key for the sample rate.
        output_key: Task data key to write the predicted language.
        confidence_key: Task data key to write prediction confidence.
        min_duration_sec: Minimum segment duration for LangID (skip shorter).
        batch_size: Number of segments to process at once.
    """

    name: str = "AmberNetLangID"
    model_name: str = "langid_ambernet"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sampling_rate"
    output_key: str = "language"
    confidence_key: str = "language_confidence"
    min_duration_sec: float = 1.0
    batch_size: int = 32
    resources: Resources = field(default_factory=lambda: Resources(gpu_memory_gb=4.0))

    _model: Any = field(default=None, init=False, repr=False)
    _target_sr: int = field(default=16000, init=False, repr=False)

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        pass

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._model is not None:
            return
        import nemo.collections.asr as nemo_asr

        logger.info(f"AmberNetLangID: loading model {self.model_name}")
        self._model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(self.model_name)
        self._model.eval()
        if torch.cuda.is_available():
            self._model = self._model.cuda()
        logger.info("AmberNetLangID: model ready")

    def teardown(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.waveform_key, self.sample_rate_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.output_key, self.confidence_key]

    def _resample_if_needed(self, waveform: np.ndarray, sr: int) -> torch.Tensor:
        audio = torch.from_numpy(waveform).float()
        if sr != self._target_sr:
            import torchaudio

            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self._target_sr)
            audio = resampler(audio)
        return audio

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        if self._model is None:
            msg = "Model not initialised — setup() was not called"
            raise RuntimeError(msg)

        valid_indices: list[int] = []
        audio_signals: list[torch.Tensor] = []
        audio_lengths: list[int] = []

        for i, task in enumerate(tasks):
            waveform = task.data.get(self.waveform_key)
            sr = task.data.get(self.sample_rate_key, self._target_sr)

            if waveform is None or len(waveform) == 0:
                task.data[self.output_key] = ""
                task.data[self.confidence_key] = 0.0
                continue

            duration = len(waveform) / sr
            if duration < self.min_duration_sec:
                task.data[self.output_key] = ""
                task.data[self.confidence_key] = 0.0
                continue

            audio = self._resample_if_needed(waveform, sr)
            valid_indices.append(i)
            audio_signals.append(audio)
            audio_lengths.append(len(audio))

        if not audio_signals:
            return tasks

        max_len = max(audio_lengths)
        batch_tensor = torch.zeros(len(audio_signals), max_len)
        length_tensor = torch.tensor(audio_lengths, dtype=torch.long)

        for j, sig in enumerate(audio_signals):
            batch_tensor[j, : len(sig)] = sig

        device = next(self._model.parameters()).device
        batch_tensor = batch_tensor.to(device)
        length_tensor = length_tensor.to(device)

        with torch.no_grad():
            logits, _ = self._model.forward(
                input_signal=batch_tensor, input_signal_length=length_tensor
            )
            probs = torch.softmax(logits, dim=-1)
            confidences, pred_indices = probs.max(dim=-1)

        labels = self._model.cfg.train_ds.get("labels", None) or self._model.cfg.get("labels", None)
        if labels is None:
            labels = [str(i) for i in range(logits.shape[-1])]

        for j, task_idx in enumerate(valid_indices):
            task = tasks[task_idx]
            pred_label = labels[pred_indices[j].item()]
            task.data[self.output_key] = pred_label
            task.data[self.confidence_key] = confidences[j].item()

        return tasks
