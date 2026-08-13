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

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.nn import functional

from nemo_curator.stages.audio.inference.whisper_langid import WhisperLangIDStage
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    import pytest


class _FakeWhisperAudio:
    N_FFT = 400
    HOP_LENGTH = 160

    @staticmethod
    def mel_filters(device: torch.device, n_mels: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(7)
        return torch.rand(n_mels, _FakeWhisperAudio.N_FFT // 2 + 1, generator=generator).to(device)


def _pad_or_trim(audio: torch.Tensor, length: int = 3200) -> torch.Tensor:
    if audio.shape[-1] > length:
        return audio[..., :length]
    return functional.pad(audio, (0, length - audio.shape[-1]))


class _FakeWhisperModel:
    def __init__(self) -> None:
        self.dims = SimpleNamespace(n_mels=80)
        self.last_mel: torch.Tensor | None = None

    def detect_language(self, mel: torch.Tensor) -> tuple[torch.Tensor, list[dict[str, float]]]:
        self.last_mel = mel
        probabilities = [{"en": 0.9, "fr": 0.1}, {"de": 0.8, "en": 0.2}]
        return torch.tensor([1, 2]), probabilities[: mel.shape[0]]

    def eval(self) -> _FakeWhisperModel:
        return self


def _task(waveform: np.ndarray, task_id: str) -> AudioTask:
    return AudioTask(data={"waveform": waveform, "sample_rate": 16000}, task_id=task_id, dataset_name="d")


def test_process_batch_uses_model_dtype_and_preserves_task_order(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_whisper = SimpleNamespace(audio=_FakeWhisperAudio, pad_or_trim=_pad_or_trim)
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)

    stage = WhisperLangIDStage(tag="secondary", min_duration_sec=0.0, batch_size=2)
    stage._device = torch.device("cpu")
    stage._mel_dtype = torch.float16
    stage._model = _FakeWhisperModel()
    tasks = [
        _task(np.ones(1600, dtype=np.float32), "a"),
        _task(np.ones(2400, dtype=np.float32) * 0.25, "b"),
    ]

    output = stage.process_batch(tasks)

    assert stage._model.last_mel is not None
    assert stage._model.last_mel.dtype == torch.float16
    assert stage._model.last_mel.device.type == "cpu"
    assert output[0].data["lid"][0]["WhisperLangID"].language == "en"
    assert output[1].data["lid"][0]["WhisperLangID"].language == "de"


def test_batched_log_mel_normalizes_each_sample_independently() -> None:
    fake_whisper = SimpleNamespace(audio=_FakeWhisperAudio)
    stage = WhisperLangIDStage(tag="secondary")
    stage._device = torch.device("cpu")

    generator = torch.Generator().manual_seed(11)
    quiet = torch.randn(3200, generator=generator) * 0.01
    loud = torch.randn(3200, generator=generator)
    audio_batch = torch.stack([quiet, loud])

    batched = stage._log_mel_spectrogram(audio_batch, 80, fake_whisper)
    separate = torch.cat(
        [stage._log_mel_spectrogram(row.unsqueeze(0), 80, fake_whisper) for row in audio_batch],
        dim=0,
    )

    torch.testing.assert_close(batched, separate)


def test_local_checkpoint_skips_node_level_model_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    stage = WhisperLangIDStage(tag="secondary", model_path="/models/whisper.pt")

    def fail_if_called(_device: str) -> None:
        message = "local checkpoint should not be loaded during setup_on_node"
        raise AssertionError(message)

    monkeypatch.setattr(stage, "_load_model", fail_if_called)
    stage.setup_on_node()


def test_setup_uses_fp16_mels_only_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeWhisperModel()
    stage = WhisperLangIDStage(tag="secondary", device="cuda")
    monkeypatch.setattr(stage, "_load_model", lambda _device: model)

    stage.setup()

    assert stage._model is model
    assert stage._mel_dtype == torch.float16

    fp32_stage = WhisperLangIDStage(tag="secondary", device="cuda", fp16=False)
    monkeypatch.setattr(fp32_stage, "_load_model", lambda _device: _FakeWhisperModel())
    fp32_stage.setup()
    assert fp32_stage._mel_dtype == torch.float32
