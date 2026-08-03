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

import torch
from loguru import logger

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata
    from nemo_curator.tasks import AudioTask

from nemo_curator.stages.audio.inference.langid_base import BaseLangIDStage, LangIDResult


@dataclass
class AmberNetLangIDStage(BaseLangIDStage):
    """Language identification using NeMo's AmberNet model.

    Processes audio segments and adds a ``language`` field with the
    predicted language code (e.g., 'en', 'de', 'fr').

    The model requires 16kHz audio and resamples internally if the
    input is at a different sample rate.

    Args:
        model_name: NeMo model name for from_pretrained.

    See :class:`~nemo_curator.stages.audio.inference.langid_base.BaseLangIDStage`
    for the shared waveform/output arguments.
    """

    name: str = "AmberNetLangID"
    model_name: str = "langid_ambernet"

    model: Any = field(default=None, init=False, repr=False)

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        pass

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self.model is not None:
            return
        import nemo.collections.asr as nemo_asr

        logger.info(f"AmberNetLangID: loading model {self.model_name}")
        self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(self.model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        logger.info("AmberNetLangID: model ready")

    def teardown(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        if self.model is None:
            msg = "Model not initialised — setup() was not called"
            raise RuntimeError(msg)

        valid_indices: list[int] = []
        audio_signals: list[torch.Tensor] = []
        audio_lengths: list[int] = []

        for i, task in enumerate(tasks):
            audio = self._prepare_audio(task)
            if audio is None:
                continue
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

        device = next(self.model.parameters()).device
        batch_tensor = batch_tensor.to(device)
        length_tensor = length_tensor.to(device)

        with torch.no_grad():
            logits, _ = self.model.forward(input_signal=batch_tensor, input_signal_length=length_tensor)
            probs = torch.softmax(logits, dim=-1)
            confidences, pred_indices = probs.max(dim=-1)

        labels = self.model.cfg.train_ds.get("labels", None) or self.model.cfg.get("labels", None)
        if labels is None:
            labels = [str(i) for i in range(logits.shape[-1])]

        for j, task_idx in enumerate(valid_indices):
            task = tasks[task_idx]
            pred_label = labels[pred_indices[j].item()]
            lid_result = LangIDResult(language=pred_label, confidence=confidences[j].item(), tag=self.tag)
            task.data.setdefault(self.lid_key, []).append({self.name: lid_result})

        return tasks
