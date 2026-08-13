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
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger
from torch.nn.utils.rnn import pad_sequence

from nemo_curator.stages.audio.inference.langid_base import BaseLangIDStage, LangIDResult

if TYPE_CHECKING:
    from types import ModuleType

    from nemo_curator.backends.base import NodeInfo, WorkerMetadata
    from nemo_curator.tasks import AudioTask


def _import_whisper() -> ModuleType:
    try:
        import whisper
    except ImportError as exc:
        msg = "OpenAI Whisper is required for WhisperLangIDStage. Install: pip install openai-whisper"
        raise ImportError(msg) from exc
    return whisper


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
        model_size: Whisper model name passed to ``whisper.load_model`` when no
            ``model_path`` is given (e.g. ``"medium"``, ``"large-v3"``).
        model_path: Path to a local Whisper checkpoint (``.pt`` file). When set,
            this takes precedence over ``model_size`` and no download occurs.
        device: Torch device on which to run the model. ``"auto"`` selects
            CUDA when available and CPU otherwise.
        fp16: Use FP16 Mel inputs on CUDA. Whisper keeps its parameters in FP32
            but is designed to cast weights to the input dtype during inference.
        batch_size: Number of audio samples to process per forward pass.

    See :class:`~nemo_curator.stages.audio.inference.langid_base.BaseLangIDStage`
    for the shared waveform/output arguments.
    """

    tag: str = "tertiary"
    name: str = "WhisperLangID"
    model_size: str = "medium"
    model_path: str | None = None
    device: str = "auto"
    fp16: bool = True
    batch_size: int = 8
    max_duration_sec: float = 30.0  # Whisper's input window is 30 s; base defaults to 10 s

    _model: Any = field(default=None, init=False, repr=False)
    _device: torch.device | None = field(default=None, init=False, repr=False)
    _mel_dtype: torch.dtype = field(default=torch.float32, init=False, repr=False)
    _hann_window: torch.Tensor | None = field(default=None, init=False, repr=False)

    def _load_model(self, device: str | torch.device) -> object:
        whisper = _import_whisper()
        name_or_path = self.model_path if self.model_path is not None else self.model_size
        return whisper.load_model(name_or_path, device=device)

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        # A local checkpoint needs no download/cache warm-up. Avoid deserializing
        # the full model on CPU immediately before each worker loads it again.
        if self.model_path is not None:
            return

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

        device_name = ("cuda" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device
        self._device = torch.device(device_name)

        model_desc = self.model_path if self.model_path is not None else self.model_size
        logger.info(f"WhisperLangID: loading {model_desc} on {self._device}")
        self._model = self._load_model(self._device)
        self._model.eval()
        self._mel_dtype = torch.float16 if self.fp16 and self._device.type == "cuda" else torch.float32
        logger.info("WhisperLangID: model ready")

    def teardown(self) -> None:
        self._model = None
        self._device = None
        self._mel_dtype = torch.float32
        self._hann_window = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _log_mel_spectrogram(self, audio_batch: torch.Tensor, n_mels: int, whisper: ModuleType) -> torch.Tensor:
        """Compute Whisper log-Mels as one GPU batch with per-sample normalization.

        OpenAI Whisper's helper supports batched STFTs, but its dynamic-range
        normalization uses a scalar maximum over the entire tensor. Calling it
        on a batch therefore makes each prediction depend on the loudest sample
        in that batch. This vectorized equivalent keeps the maximum per sample,
        caches the Hann window, and casts the result to the configured dtype.
        """
        if self._device is None:
            msg = "Model device is not initialized"
            raise RuntimeError(msg)

        n_fft = int(whisper.audio.N_FFT)
        hop_length = int(whisper.audio.HOP_LENGTH)
        if (
            self._hann_window is None
            or self._hann_window.device != self._device
            or self._hann_window.dtype != audio_batch.dtype
        ):
            self._hann_window = torch.hann_window(n_fft, device=self._device, dtype=audio_batch.dtype)

        stft = torch.stft(
            audio_batch,
            n_fft,
            hop_length,
            window=self._hann_window,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs().square()
        filters = whisper.audio.mel_filters(self._device, n_mels).to(dtype=magnitudes.dtype)
        mel_spec = filters @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        per_sample_max = log_spec.amax(dim=(-2, -1), keepdim=True)
        log_spec = torch.maximum(log_spec, per_sample_max - 8.0)
        return ((log_spec + 4.0) / 4.0).to(dtype=self._mel_dtype)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        whisper = _import_whisper()
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

            # Transfer only the longest real waveform in this chunk, then add the
            # fixed 30-second padding on-device. This avoids sending zero padding
            # over PCIe and keeps the batched STFT on the inference device.
            audio_batch = pad_sequence(chunk_signals, batch_first=True, padding_value=0.0)
            audio_batch = audio_batch.to(self._device, non_blocking=self._device.type == "cuda")
            audio_batch = whisper.pad_or_trim(audio_batch)
            mel_batch = self._log_mel_spectrogram(audio_batch, n_mels, whisper)
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
