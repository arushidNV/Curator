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

"""Whisper language identification stage."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any

import torch
from loguru import logger
from torch.nn.utils.rnn import pad_sequence

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.tasks import AudioTask

from nemo_curator.stages.audio.inference.langid_base import BaseLangIDStage, LangIDResult


@dataclass
class WhisperLangIDStage(BaseLangIDStage):
    """Language identification using OpenAI Whisper.

    Audio is padded or trimmed to Whisper's 30-second input window, converted
    to a batched log-Mel spectrogram, and passed to ``detect_language`` without
    running transcription.

    Writes a ``LangIDResult`` tagged ``tertiary`` into ``task.data[lid_key]``,
    consistent with the SpeechBrain/AmberNet (primary) and Indic Canary (secondary)
    stages.

    Args:
        model_size: Whisper model name passed directly to ``whisper.load_model``
            (e.g. ``"medium"``, ``"large-v3"``).
        device: Torch device on which to run the model. ``"auto"`` selects
            CUDA when available and CPU otherwise.
        batch_size: Number of audio samples to process per forward pass.

    See :class:`~nemo_curator.stages.audio.inference.langid_base.BaseLangIDStage`
    for the shared waveform/output arguments.
    """

    tag: str = "tertiary"
    name: str = "WhisperLangID"
    model_size: str = "medium"
    device: str = "auto"
    batch_size: int = 8
    max_duration_sec: float = 30.0  # Whisper's input window is 30 s; base defaults to 10 s

    _model: Any = field(default=None, init=False, repr=False)
    _device: torch.device | None = field(default=None, init=False, repr=False)

    def _load_model(self, device: str | torch.device) -> Any:
        try:
            import whisper
        except ImportError as exc:
            msg = "OpenAI Whisper is required for WhisperLangIDStage. Install: pip install openai-whisper"
            raise ImportError(msg) from exc

        return whisper.load_model(self.model_size, device=device)

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        # Populate the checkpoint cache once before worker actors start. Loading
        # on CPU avoids reserving a GPU in this node-level hook.
        try:
            model = self._load_model("cpu")
            del model
            gc.collect()
        except Exception as exc:  # noqa: BLE001
            logger.info(f"WhisperLangID: could not pre-cache {self.model_size} on node ({exc})")

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._model is not None:
            return

        if self.device == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_name = self.device
        self._device = torch.device(device_name)

        logger.info(f"WhisperLangID: loading {self.model_size} on {self._device}")
        self._model = self._load_model(self._device)
        self._model.eval()
        logger.info("WhisperLangID: model ready")

    def teardown(self) -> None:
        self._model = None
        self._device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        try:
            import whisper
        except ImportError as exc:
            msg = "OpenAI Whisper is required for WhisperLangIDStage. Install: pip install openai-whisper"
            raise ImportError(msg) from exc
        if len(tasks) == 0:
            return []
        if self._model is None or self._device is None:
            msg = "Model not initialised — setup() was not called"
            raise RuntimeError(msg)

        valid_indices: list[int] = []
        audio_signals: list[torch.Tensor] = []
        for i, task in enumerate(tasks):
            audio = self._prepare_audio(task)
            if audio is None:
                continue
            valid_indices.append(i)
            audio_signals.append(audio)

        if not audio_signals:
            return tasks

        n_mels = int(self._model.dims.n_mels)
        for chunk_start in range(0, len(audio_signals), self.batch_size):
            chunk_signals = audio_signals[chunk_start : chunk_start + self.batch_size]
            chunk_indices = valid_indices[chunk_start : chunk_start + self.batch_size]

            # Pad variable-length rows into [B, T], then normalize to Whisper's fixed
            # 30-second window. log_mel_spectrogram preserves the leading batch
            # dimension and returns [B, n_mels, frames].
            audio_batch = pad_sequence(chunk_signals, batch_first=True, padding_value=0.0)
            audio_batch = whisper.pad_or_trim(audio_batch)
            mel_batch = whisper.log_mel_spectrogram(audio_batch, n_mels=n_mels).to(self._device)
            with torch.inference_mode():
                _language_tokens, probabilities = self._model.detect_language(mel_batch)

            # detect_language returns one probability dictionary per batch row.
            if isinstance(probabilities, dict):
                probabilities = [probabilities]
            if len(probabilities) != len(chunk_indices):
                msg = f"Whisper returned {len(probabilities)} predictions for {len(chunk_indices)} inputs"
                raise RuntimeError(msg)

            for task_idx, language_probabilities in zip(chunk_indices, probabilities, strict=True):
                language = max(language_probabilities, key=language_probabilities.get)
                lid_result = LangIDResult(
                    language=language,
                    confidence=float(language_probabilities[language]),
                    tag=self.tag,
                )
                tasks[task_idx].data.setdefault(self.lid_key, []).append({self.name: lid_result})

        return tasks
