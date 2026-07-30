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

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from nemo_curator.stages.audio.inference.speechbrain_langid import SpeechBrainLangIDStage
from nemo_curator.tasks import AudioTask


def _task(waveform: np.ndarray, sr: int = 16000, task_id: str = "t") -> AudioTask:
    return AudioTask(data={"waveform": waveform, "sample_rate": sr}, task_id=task_id, dataset_name="d")


class TestBaseLangIDPrep:
    """Shared waveform prep lives in BaseLangIDStage; exercise it via a subclass."""

    def test_missing_waveform_sets_empty(self) -> None:
        stage = SpeechBrainLangIDStage()
        task = AudioTask(data={}, task_id="t", dataset_name="d")
        assert stage._prepare_audio(task) is None
        assert task.data["language"] == ""
        assert task.data["language_confidence"] == 0.0

    def test_short_waveform_skipped(self) -> None:
        stage = SpeechBrainLangIDStage(min_duration_sec=2.0)
        task = _task(np.zeros(16000, dtype=np.float32))  # 1s < 2s
        assert stage._prepare_audio(task) is None
        assert task.data["language"] == ""

    def test_valid_waveform_returns_tensor(self) -> None:
        stage = SpeechBrainLangIDStage(min_duration_sec=0.5)
        out = stage._prepare_audio(_task(np.zeros(16000, dtype=np.float32)))
        assert isinstance(out, torch.Tensor)
        assert out.shape[0] == 16000

    def test_long_waveform_is_truncated(self) -> None:
        stage = SpeechBrainLangIDStage(min_duration_sec=0.5, max_duration_sec=2.0)
        out = stage._prepare_audio(_task(np.zeros(48000, dtype=np.float32)))
        assert out is not None
        assert out.shape[0] == 32000

    def test_zero_max_duration_disables_truncation(self) -> None:
        stage = SpeechBrainLangIDStage(min_duration_sec=0.5, max_duration_sec=0)
        out = stage._prepare_audio(_task(np.zeros(48000, dtype=np.float32)))
        assert out is not None
        assert out.shape[0] == 48000

    def test_worker_override_and_defaults(self) -> None:
        stage = SpeechBrainLangIDStage(max_workers=2)
        assert stage.batch_size == 16
        assert stage.max_duration_sec == 10.0
        assert stage.num_workers() == 2

    def test_resampler_is_built_once_and_cached(self) -> None:
        stage = SpeechBrainLangIDStage(min_duration_sec=0.1, target_sr=16000)
        wav = np.zeros(8000, dtype=np.float32)
        out1 = stage._resample(wav, 8000)
        first = stage._resamplers[(8000, 16000)]
        stage._resample(wav, 8000)
        assert stage._resamplers[(8000, 16000)] is first  # reused, not rebuilt
        assert out1.shape[0] == pytest.approx(16000, abs=64)  # 8k -> 16k upsample


class TestSpeechBrainBatching:
    def test_all_segments_run_in_one_forward_pass(self) -> None:
        stage = SpeechBrainLangIDStage(min_duration_sec=0.5)
        mock_clf = MagicMock()
        # classify_batch returns log-probabilities (Softmax(apply_log=True)); the stage
        # exponentiates them into a linear 0-1 confidence.
        mock_clf.classify_batch.return_value = (None, torch.log(torch.tensor([0.9, 0.8])), None, ["en", "de"])
        stage._classifier = mock_clf

        tasks = [
            _task(np.zeros(16000, dtype=np.float32), task_id="a"),
            _task(np.zeros(24000, dtype=np.float32), task_id="b"),
        ]
        out = stage.process_batch(tasks)

        # One batched call for the whole batch, not one per segment.
        assert mock_clf.classify_batch.call_count == 1
        batch_tensor, wav_lens = mock_clf.classify_batch.call_args[0]
        assert batch_tensor.shape[0] == 2
        assert wav_lens.shape[0] == 2
        assert out[0].data["language"] == "en"
        assert out[1].data["language"] == "de"
        assert out[0].data["language_confidence"] == pytest.approx(0.9)

    def test_short_segments_skipped_but_batch_still_runs(self) -> None:
        stage = SpeechBrainLangIDStage(min_duration_sec=2.0)
        mock_clf = MagicMock()
        mock_clf.classify_batch.return_value = (None, torch.tensor([0.7]), None, ["fr"])
        stage._classifier = mock_clf

        tasks = [
            _task(np.zeros(16000, dtype=np.float32), task_id="short"),  # 1s -> skipped
            _task(np.zeros(48000, dtype=np.float32), task_id="long"),   # 3s -> classified
        ]
        out = stage.process_batch(tasks)

        assert mock_clf.classify_batch.call_count == 1
        assert out[0].data["language"] == ""
        assert out[0].data["language_confidence"] == 0.0
        assert out[1].data["language"] == "fr"

    def test_all_short_skips_model_entirely(self) -> None:
        stage = SpeechBrainLangIDStage(min_duration_sec=5.0)
        mock_clf = MagicMock()
        stage._classifier = mock_clf

        out = stage.process_batch([_task(np.zeros(16000, dtype=np.float32), task_id="short")])

        mock_clf.classify_batch.assert_not_called()
        assert out[0].data["language"] == ""
