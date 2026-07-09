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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nemo_curator.stages.audio.inference.sortformer import (
    InferenceSortformerStage,
    _parse_sortformer_segments,
    _write_rttm,
)
from nemo_curator.tasks import AudioTask


class TestParseSortformerSegments:
    def test_parses_string_segments(self) -> None:
        raw = ["0.00 2.70 speaker_0", "0.80 13.60 speaker_1"]
        out = _parse_sortformer_segments(raw)
        assert len(out) == 2
        assert out[0] == {"start": 0.0, "end": 2.7, "speaker": "speaker_0"}
        assert out[1] == {"start": 0.8, "end": 13.6, "speaker": "speaker_1"}

    def test_parses_object_segments(self) -> None:
        seg1 = SimpleNamespace(start=1.0, end=3.5, speaker="speaker_0")
        seg2 = SimpleNamespace(start=4.0, end=7.2, speaker="speaker_1")
        out = _parse_sortformer_segments([seg1, seg2])
        assert out[0] == {"start": 1.0, "end": 3.5, "speaker": "speaker_0"}
        assert out[1] == {"start": 4.0, "end": 7.2, "speaker": "speaker_1"}

    def test_parses_object_with_label_attr(self) -> None:
        seg = SimpleNamespace(start=0.5, end=1.5, label="spk_2")
        out = _parse_sortformer_segments([seg])
        assert out[0]["speaker"] == "spk_2"

    def test_parses_tuple_segments(self) -> None:
        raw = [(0.0, 2.0, "speaker_0"), (3.0, 5.0, "speaker_1")]
        out = _parse_sortformer_segments(raw)
        assert len(out) == 2
        assert out[0] == {"start": 0.0, "end": 2.0, "speaker": "speaker_0"}

    def test_empty_list_returns_empty(self) -> None:
        assert _parse_sortformer_segments([]) == []

    def test_unrecognised_format_warns(self) -> None:
        out = _parse_sortformer_segments([42])
        assert out == []


class TestWriteRttm:
    def test_writes_rttm_file(self, tmp_path: Path) -> None:
        segments = [
            {"start": 0.0, "end": 2.5, "speaker": "speaker_0"},
            {"start": 3.0, "end": 5.0, "speaker": "speaker_1"},
        ]
        _write_rttm(segments, "test_session", str(tmp_path))
        rttm_path = tmp_path / "test_session.rttm"
        assert rttm_path.exists()
        lines = rttm_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("SPEAKER test_session 1 0.000 2.500")
        assert "speaker_0" in lines[0]
        assert lines[1].startswith("SPEAKER test_session 1 3.000 2.000")
        assert "speaker_1" in lines[1]

    def test_sanitizes_slashes_in_session_name(self, tmp_path: Path) -> None:
        segments = [{"start": 0.0, "end": 1.0, "speaker": "speaker_0"}]
        sess_name = "audio-riva-originals/nl/pilot_90files_5"
        _write_rttm(segments, sess_name, str(tmp_path))
        rttm_path = tmp_path / "audio-riva-originals_nl_pilot_90files_5.rttm"
        assert rttm_path.exists()
        assert "SPEAKER audio-riva-originals/nl/pilot_90files_5" in rttm_path.read_text()

    def test_setup_on_node_pre_caches_model(self) -> None:
        stage = InferenceSortformerStage(model_name="nvidia/diar_streaming_sortformer_4spk-v2")
        with patch("nemo_curator.stages.audio.inference.sortformer.snapshot_download") as mock_dl:
            stage.setup_on_node()
            mock_dl.assert_called_once_with(repo_id="nvidia/diar_streaming_sortformer_4spk-v2", cache_dir=None)

    def test_setup_on_node_skips_for_local_path(self) -> None:
        stage = InferenceSortformerStage(model_path="/local/model.nemo")
        with patch("nemo_curator.stages.audio.inference.sortformer.snapshot_download") as mock_dl:
            stage.setup_on_node()
            mock_dl.assert_not_called()

    def test_setup_skips_when_model_provided(self) -> None:
        mock_model = MagicMock()
        mock_model.sortformer_modules = MagicMock()
        stage = InferenceSortformerStage(diar_model=mock_model)
        stage.setup()
        assert mock_model.sortformer_modules.chunk_len == 340

    def test_rejects_unknown_precision(self) -> None:
        with pytest.raises(ValueError, match="Unsupported Sortformer precision"):
            InferenceSortformerStage(precision="int8")  # type: ignore[arg-type]

    def test_tensorrt_backend_requires_engine_path(self) -> None:
        with pytest.raises(ValueError, match="tensorrt_engine_path is required"):
            InferenceSortformerStage(backend="tensorrt")

    @patch("nemo_curator.stages.audio.inference.sortformer_tensorrt.TensorRTSortformerRunner")
    def test_tensorrt_backend_replaces_only_streaming_forward(self, mock_runner_cls: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.sortformer_modules = MagicMock()
        runner = mock_runner_cls.return_value
        stage = InferenceSortformerStage(
            diar_model=mock_model,
            model_path="/models/sortformer.nemo",
            backend="tensorrt",
            tensorrt_engine_path="/models/sortformer.plan",
        )

        stage.setup()

        mock_runner_cls.assert_called_once_with(
            mock_model,
            "/models/sortformer.plan",
            model_path="/models/sortformer.nemo",
            metadata_path=None,
            validate_metadata=True,
        )
        assert mock_model.forward_streaming == runner.forward_streaming
        stage.teardown()
        runner.close.assert_called_once_with()

    @patch("nemo_curator.stages.audio.inference.sortformer.torch.cuda.is_available", return_value=False)
    def test_mixed_precision_falls_back_to_fp32_without_cuda(self, mock_cuda: MagicMock) -> None:
        assert mock_cuda.return_value is False
        stage = InferenceSortformerStage(precision="bf16")
        with stage._autocast_context():
            pass

    def test_optimized_diarize_wrapper_keeps_cuda_cache_warm(self) -> None:
        mock_model = MagicMock()
        mock_model.sortformer_modules = MagicMock()
        predictions = MagicMock()
        cpu_predictions = MagicMock()
        mock_model.forward.return_value = predictions
        predictions.to.return_value = cpu_predictions
        stage = InferenceSortformerStage(diar_model=mock_model, avoid_cuda_cache_flush=True)

        stage.setup()
        result = mock_model._diarize_forward(["audio", "length"])

        mock_model.forward.assert_called_once_with(audio_signal="audio", audio_signal_length="length")
        predictions.to.assert_called_once_with("cpu")
        assert result is cpu_predictions

    def test_streaming_config_applied(self) -> None:
        mock_model = MagicMock()
        mock_model.sortformer_modules = MagicMock()
        stage = InferenceSortformerStage(
            diar_model=mock_model,
            chunk_len=124,
            chunk_right_context=1,
            fifo_len=124,
            spkcache_update_period=124,
            spkcache_len=200,
        )
        stage.setup()
        sm = mock_model.sortformer_modules
        assert sm.chunk_len == 124
        assert sm.chunk_right_context == 1
        assert sm.fifo_len == 124
        assert sm.spkcache_update_period == 124
        assert sm.spkcache_len == 200

    def _make_mock_model(self, fake_segments_per_file: list[list[str]]) -> MagicMock:
        mock_model = MagicMock()
        mock_model.sortformer_modules = MagicMock()
        mock_model.diarize.return_value = fake_segments_per_file
        return mock_model

    def test_process_audio_task(self) -> None:
        fake_output = [
            ["0.00 2.70 speaker_0", "0.80 13.60 speaker_1"],
        ]
        mock_model = self._make_mock_model(fake_output)
        stage = InferenceSortformerStage(diar_model=mock_model)

        task = AudioTask(
            data={"audio_filepath": "/test/audio1.wav"},
        )
        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert result.data["audio_filepath"] == "/test/audio1.wav"
        assert result.data["diar_segments"] == [
            {"start": 0.0, "end": 2.7, "speaker": "speaker_0"},
            {"start": 0.8, "end": 13.6, "speaker": "speaker_1"},
        ]
        assert result.task_id.endswith("_sortformer")
        mock_model.diarize.assert_called_once_with(
            audio=["/test/audio1.wav"],
            batch_size=8,
        )

    def test_process_batch_buckets_by_duration_and_restores_task_order(self) -> None:
        mock_model = MagicMock()
        mock_model.sortformer_modules = MagicMock()

        def fake_diarize(*, audio, batch_size, sample_rate):  # noqa: ANN001, ANN202
            assert batch_size == 2
            assert sample_rate == 16000
            return [[f"0 1 samples_{waveform.shape[-1]}"] for waveform in audio]

        mock_model.diarize.side_effect = fake_diarize
        stage = InferenceSortformerStage(diar_model=mock_model, inference_batch_size=2)
        tasks = [
            AudioTask(data={"waveform": np.zeros(size), "sample_rate": 16000}, task_id=f"task_{size}")
            for size in (32000, 8000, 16000)
        ]

        result = stage.process_batch(tasks)

        assert result == tasks
        called_audio = mock_model.diarize.call_args.kwargs["audio"]
        assert [waveform.shape[-1] for waveform in called_audio] == [8000, 16000, 32000]
        assert [task.data["diar_segments"][0]["speaker"] for task in result] == [
            "samples_32000",
            "samples_8000",
            "samples_16000",
        ]
        assert [task.task_id for task in result] == [
            "task_32000_sortformer",
            "task_8000_sortformer",
            "task_16000_sortformer",
        ]

    def test_process_batch_groups_waveforms_by_sample_rate(self) -> None:
        mock_model = MagicMock()
        mock_model.sortformer_modules = MagicMock()
        mock_model.diarize.side_effect = lambda **kwargs: [["0 1 speaker_0"] for _ in kwargs["audio"]]
        stage = InferenceSortformerStage(diar_model=mock_model)
        tasks = [
            AudioTask(data={"waveform": np.zeros(8000), "sample_rate": 8000}, task_id="8k"),
            AudioTask(data={"waveform": np.zeros(16000), "sample_rate": 16000}, task_id="16k"),
        ]

        stage.process_batch(tasks)

        assert mock_model.diarize.call_count == 2
        assert {call.kwargs["sample_rate"] for call in mock_model.diarize.call_args_list} == {8000, 16000}

    def test_process_batch_leaves_read_errors_untouched(self) -> None:
        mock_model = self._make_mock_model([["0 1 speaker_0"]])
        stage = InferenceSortformerStage(diar_model=mock_model)
        failed = AudioTask(data={"read_error": True}, task_id="failed")
        valid = AudioTask(data={"audio_filepath": "/test/valid.wav"}, task_id="valid")

        result = stage.process_batch([failed, valid])

        assert result[0].task_id == "failed"
        assert result[1].task_id == "valid_sortformer"

    def test_process_writes_rttm(self, tmp_path: Path) -> None:
        fake_output = [["0.00 2.50 speaker_0"]]
        mock_model = self._make_mock_model(fake_output)
        stage = InferenceSortformerStage(
            diar_model=mock_model,
            rttm_out_dir=str(tmp_path),
        )

        task = AudioTask(data={"audio_filepath": "/test/my_audio.wav"})
        stage.process(task)

        rttm_file = tmp_path / "my_audio.rttm"
        assert rttm_file.exists()
        content = rttm_file.read_text()
        assert "SPEAKER my_audio" in content

    def test_process_preserves_existing_data(self) -> None:
        fake_output = [["0.00 1.00 speaker_0"]]
        mock_model = self._make_mock_model(fake_output)
        stage = InferenceSortformerStage(diar_model=mock_model)

        task = AudioTask(
            data={"audio_filepath": "/test/audio1.wav", "extra_key": "extra_value"},
        )
        result = stage.process(task)
        assert result.data["extra_key"] == "extra_value"
        assert "diar_segments" in result.data

    def test_process_uses_session_name_from_data(self, tmp_path: Path) -> None:
        fake_output = [["0.00 1.00 speaker_0"]]
        mock_model = self._make_mock_model(fake_output)
        stage = InferenceSortformerStage(
            diar_model=mock_model,
            rttm_out_dir=str(tmp_path),
        )

        task = AudioTask(
            data={"audio_filepath": "/test/audio1.wav", "session_name": "sess_42"},
        )
        stage.process(task)
        assert (tmp_path / "sess_42.rttm").exists()
