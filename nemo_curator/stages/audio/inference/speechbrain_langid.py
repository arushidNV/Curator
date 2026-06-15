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

"""SpeechBrain VoxLingua107 Language Identification stage.

Identifies the spoken language of each audio segment using the SpeechBrain
ECAPA-TDNN model trained on VoxLingua107 (107 languages).
See: https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa
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
class SpeechBrainLangIDStage(ProcessingStage[AudioTask, AudioTask]):
    """Language identification using SpeechBrain's VoxLingua107 ECAPA-TDNN model.

    Supports 107 languages with high accuracy. Operates on in-memory waveforms,
    resamples to 16kHz internally if needed.

    Args:
        source: HuggingFace model source (default: speechbrain/lang-id-voxlingua107-ecapa).
        savedir: Directory to cache the downloaded model.
        target_sr: Target sample rate for the model (default 16000).
        waveform_key: Task data key for the audio waveform.
        sample_rate_key: Task data key for the sample rate.
        output_key: Task data key to write the predicted language.
        confidence_key: Task data key to write prediction confidence.
        min_duration_sec: Minimum segment duration for LangID (skip shorter).
    """

    name: str = "SpeechBrainLangID"
    source: str = "speechbrain/lang-id-voxlingua107-ecapa"
    savedir: str = "/tmp/speechbrain_langid"
    target_sr: int = 16000
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    output_key: str = "language"
    confidence_key: str = "language_confidence"
    min_duration_sec: float = 1.0
    resources: Resources = field(default_factory=lambda: Resources(gpu_memory_gb=4.0))

    _classifier: Any = field(default=None, init=False, repr=False)

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        pass

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._classifier is not None:
            return
        from speechbrain.inference.classifiers import EncoderClassifier

        logger.info(f"SpeechBrainLangID: loading model from {self.source}")
        self._classifier = EncoderClassifier.from_hparams(
            source=self.source,
            savedir=self.savedir,
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )
        logger.info("SpeechBrainLangID: model ready")

    def teardown(self) -> None:
        self._classifier = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.waveform_key, self.sample_rate_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.output_key, self.confidence_key]

    def _resample_if_needed(self, waveform: np.ndarray, sr: int) -> torch.Tensor:
        audio = torch.from_numpy(waveform).float()
        if sr != self.target_sr:
            import torchaudio

            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            audio = resampler(audio)
        return audio

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        if self._classifier is None:
            msg = "Model not initialised — setup() was not called"
            raise RuntimeError(msg)

        for task in tasks:
            waveform = task.data.get(self.waveform_key)
            sr = task.data.get(self.sample_rate_key, self.target_sr)

            if waveform is None:
                task.data[self.output_key] = ""
                task.data[self.confidence_key] = 0.0
                continue

            if isinstance(waveform, torch.Tensor):
                waveform = waveform.squeeze().cpu().numpy()
            else:
                waveform = np.asarray(waveform, dtype=np.float32)
            if waveform.ndim > 1:
                waveform = waveform.squeeze()
            if waveform.size == 0:
                task.data[self.output_key] = ""
                task.data[self.confidence_key] = 0.0
                continue

            duration = len(waveform) / sr
            if duration < self.min_duration_sec:
                task.data[self.output_key] = ""
                task.data[self.confidence_key] = 0.0
                continue

            audio = self._resample_if_needed(waveform, sr).unsqueeze(0)
            out_prob, score, index, label = self._classifier.classify_batch(audio)
            task.data[self.output_key] = label[0]
            task.data[self.confidence_key] = score.squeeze().item()

        return tasks
