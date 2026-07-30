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

import os
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata
    from nemo_curator.tasks import AudioTask

from nemo_curator.stages.audio.inference.langid_base import BaseLangIDStage


@dataclass
class SpeechBrainLangIDStage(BaseLangIDStage):
    """Language identification using SpeechBrain's VoxLingua107 ECAPA-TDNN model.

    Supports 107 languages with high accuracy. Operates on in-memory waveforms,
    resamples to 16kHz internally if needed.

    Args:
        source: HuggingFace model source (default: speechbrain/lang-id-voxlingua107-ecapa).
        savedir: Directory to cache the downloaded model.

    See :class:`~nemo_curator.stages.audio.inference.langid_base.BaseLangIDStage`
    for the shared waveform/output arguments.
    """

    name: str = "SpeechBrainLangID"
    source: str = "speechbrain/lang-id-voxlingua107-ecapa"
    savedir: str = field(default_factory=lambda: os.path.join(tempfile.gettempdir(), "speechbrain_langid"))

    _classifier: Any = field(default=None, init=False, repr=False)

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        # Pre-fetch the model once per node so the per-actor setup() below is a HF
        # cache hit instead of N concurrent downloads. Best-effort: if it fails
        # (e.g. offline / non-HF source), actors still fetch individually in setup().
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=self.source)
        except Exception as exc:  # noqa: BLE001
            logger.info(f"SpeechBrainLangID: could not pre-cache {self.source} on node ({exc})")

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._classifier is not None:
            return
        from speechbrain.inference.classifiers import EncoderClassifier

        # Isolate savedir per actor. SpeechBrain's fetch into a shared savedir is NOT
        # concurrency-safe: co-located actors race on the same symlinks and one reads a
        # half-populated dir -> FileNotFoundError: <savedir>/hyperparams.yaml. A per-PID
        # savedir removes the race; the node-level pre-fetch keeps each copy a cache hit.
        savedir = os.path.join(self.savedir, f"actor_{os.getpid()}")
        logger.info(f"SpeechBrainLangID: loading model from {self.source} (savedir={savedir})")
        self._classifier = EncoderClassifier.from_hparams(
            source=self.source,
            savedir=savedir,
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )
        logger.info("SpeechBrainLangID: model ready")

    def teardown(self) -> None:
        self._classifier = None

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        if self._classifier is None:
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

        # Pad to a single [B, T] batch and classify in one forward pass. wav_lens is the
        # relative (0-1) valid length of each row so padding is ignored by the model.
        max_len = max(audio_lengths)
        batch_tensor = torch.zeros(len(audio_signals), max_len)
        for j, sig in enumerate(audio_signals):
            batch_tensor[j, : len(sig)] = sig
        wav_lens = torch.tensor([length / max_len for length in audio_lengths])

        # inference_mode is essential here: from_hparams only sets .eval() (dropout/BN),
        # which does NOT stop autograd. Without this, SpeechBrain's classify_batch builds
        # a full graph and keeps every activation alive for the whole ECAPA-TDNN forward,
        # inflating peak GPU memory ~2-3x — the usual cause of OOM on long (up to 40s) VAD
        # segments, especially with multiple actors packed per GPU.
        with torch.inference_mode():
            _out_prob, score, _index, label = self._classifier.classify_batch(batch_tensor, wav_lens)

        # The model's final layer is Softmax(apply_log=True), so `score` is a log-probability
        # (<= 0). Exponentiate to a linear 0-1 confidence.
        confidence = torch.exp(score)

        for j, task_idx in enumerate(valid_indices):
            task = tasks[task_idx]
            task.data[self.output_key] = label[j]
            task.data[self.confidence_key] = confidence[j].item()

        return tasks
