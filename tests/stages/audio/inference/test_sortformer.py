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

import pytest
import torch

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

    def test_parses_dict_segments(self) -> None:
        segment = {"start": 0.0, "end": 2.0, "speaker": "speaker_0"}
        assert _parse_sortformer_segments([segment]) == [segment]

    def test_empty_list_returns_empty(self) -> None:
        assert _parse_sortformer_segments([]) == []

    def test_unrecognised_format_warns(self) -> None:
        out = _parse_sortformer_segments([42])
        assert out == []


class TestWriteRttm:
    def test_rejects_unknown_precision(self) -> None:
        with pytest.raises(ValueError, match="Unsupported Sortformer precision"):
            InferenceSortformerStage(precision="int8")  # type: ignore[arg-type]

    def test_tensorrt_requires_all_artifacts(self) -> None:
        with pytest.raises(ValueError, match="requires engine, config, and runtime module"):
            InferenceSortformerStage(backend="tensorrt", tensorrt_engine_path="model.plan")

    def test_writes_rttm_file(self, tmp_path: Path) -> None:
        segments = [
            {"start": 0.0, "end": 2.5, "speaker": "speaker_0"},
            {"start": 3.0, "end": 5.0, "speaker": "speaker_1"},
        ]
        _write_rttm(segments, "test_session", str(tmp_path))
        rttm_path = tmp_path / "rttm" / "test_session.rttm"
        assert rttm_path.exists()
        lines = rttm_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("SPEAKER test_session 1 0.000 2.500")
        assert "speaker_0" in lines[0]
        assert lines[1].startswith("SPEAKER test_session 1 3.000 2.000")
        assert "speaker_1" in lines[1]

    def test_writes_rttm_under_shard_key_subdir(self, tmp_path: Path) -> None:
        segments = [{"start": 0.0, "end": 1.0, "speaker": "speaker_0"}]
        shard_key = "yt_harvested/es/youtube/v1.1_wer10_whisper/yt_mixed/manifest_0"
        _write_rttm(segments, "recording_1", str(tmp_path), shard_key=shard_key)
        rttm_path = tmp_path / shard_key / "rttm" / "recording_1.rttm"
        assert rttm_path.exists()

    def test_sanitizes_slashes_in_session_name(self, tmp_path: Path) -> None:
        segments = [{"start": 0.0, "end": 1.0, "speaker": "speaker_0"}]
        sess_name = "audio-riva-originals/nl/pilot_90files_5"
        _write_rttm(segments, sess_name, str(tmp_path))
        rttm_path = tmp_path / "rttm" / "audio-riva-originals_nl_pilot_90files_5.rttm"
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

    def test_setup_uses_tensorrt_runtime(self) -> None:
        stage = InferenceSortformerStage(
            backend="tensorrt",
            tensorrt_engine_path="model.plan",
            tensorrt_config_path="model.json",
            tensorrt_runtime_module_path="sortformer_modules.py",
            inference_batch_size=2,
        )
        runtime = MagicMock()
        runtime.diarize.return_value = [[], []]
        with patch(
            "nemo_curator.stages.audio.inference.sortformer_tensorrt.TensorRTSortformer",
            return_value=runtime,
        ) as runtime_class:
            stage.setup()

        runtime_class.assert_called_once_with("model.plan", "model.json", "sortformer_modules.py")
        runtime.diarize.assert_not_called()
        stage.teardown()
        runtime.close.assert_called_once_with()

    def test_setup_preserves_model_streaming_config(self) -> None:
        mock_model = MagicMock()
        mock_model.sortformer_modules = SimpleNamespace(
            chunk_len=264,
            chunk_right_context=1,
            fifo_len=0,
            spkcache_update_period=188,
            spkcache_len=264,
        )
        stage = InferenceSortformerStage(diar_model=mock_model)
        stage.setup()
        assert mock_model.sortformer_modules.chunk_len == 264
        assert mock_model.sortformer_modules.chunk_right_context == 1
        assert mock_model.sortformer_modules.fifo_len == 0
        assert mock_model.sortformer_modules.spkcache_update_period == 188
        assert mock_model.sortformer_modules.spkcache_len == 264

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
        sm._check_streaming_parameters.assert_called_once_with()

    def test_setup_compiles_encoder(self) -> None:
        mock_model = MagicMock()
        mock_model.sortformer_modules = MagicMock()
        mock_model.diarize.return_value = [[]]
        stage = InferenceSortformerStage(diar_model=mock_model, compile_encoder=True)
        encoder = mock_model.encoder

        with patch("torch.compile", return_value=MagicMock()) as mock_compile:
            stage.setup()

        mock_compile.assert_called_once_with(encoder, dynamic=False)

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
            batch_size=1,
        )

    def test_process_writes_rttm_under_shard_key(self, tmp_path: Path) -> None:
        fake_output = [["0.00 2.50 speaker_0"]]
        mock_model = self._make_mock_model(fake_output)
        stage = InferenceSortformerStage(
            diar_model=mock_model,
            rttm_out_dir=str(tmp_path),
        )
        shard_key = "yt_harvested/es/manifest_0"
        task = AudioTask(
            data={"audio_filepath": "/test/my_audio.wav"},
            _metadata={"_shard_key": shard_key},
        )
        stage.process(task)

        rttm_file = tmp_path / shard_key / "rttm" / "my_audio.rttm"
        assert rttm_file.exists()
        assert task.data["rttm_filepath"] == f"{shard_key}/rttm/my_audio.rttm"

    def test_bfloat16_inference_uses_autocast(self) -> None:
        mock_model = self._make_mock_model([[]])
        stage = InferenceSortformerStage(diar_model=mock_model, precision="bf16")

        with patch("torch.autocast") as mock_autocast:
            stage.process(AudioTask(data={"audio_filepath": "/test/audio1.wav"}))

        mock_autocast.assert_called_once_with(device_type="cuda", dtype=torch.bfloat16)

    def test_process_batch_uses_inference_batch_size(self) -> None:
        fake_output = [[f"0.00 1.00 speaker_{index}"] for index in range(4)]
        mock_model = self._make_mock_model(fake_output)
        stage = InferenceSortformerStage(diar_model=mock_model, inference_batch_size=4)
        tasks = [AudioTask(data={"audio_filepath": f"/test/audio{index}.wav"}) for index in range(4)]

        stage.process_batch(tasks)

        mock_model.diarize.assert_called_once_with(
            audio=[f"/test/audio{index}.wav" for index in range(4)],
            batch_size=4,
        )

    def test_process_batch_orders_by_duration_and_restores_results(self) -> None:
        durations = [8.0, 2.0, 7.0, 1.0]
        sorted_indices = [3, 1, 2, 0]
        fake_output = [[f"0.00 1.00 speaker_{index}"] for index in sorted_indices]
        mock_model = self._make_mock_model(fake_output)
        stage = InferenceSortformerStage(diar_model=mock_model, inference_batch_size=2)
        tasks = [
            AudioTask(data={"audio_filepath": f"/test/audio{index}.wav", "duration": duration})
            for index, duration in enumerate(durations)
        ]

        results = stage.process_batch(tasks)

        mock_model.diarize.assert_called_once_with(
            audio=[f"/test/audio{index}.wav" for index in sorted_indices],
            batch_size=2,
        )
        assert results == tasks
        for index, task in enumerate(results):
            assert task.data["diar_segments"][0]["speaker"] == f"speaker_{index}"

    def test_process_writes_rttm(self, tmp_path: Path) -> None:
        fake_output = [["0.00 2.50 speaker_0"]]
        mock_model = self._make_mock_model(fake_output)
        stage = InferenceSortformerStage(
            diar_model=mock_model,
            rttm_out_dir=str(tmp_path),
        )

        task = AudioTask(data={"audio_filepath": "/test/my_audio.wav"})
        stage.process(task)

        rttm_file = tmp_path / "rttm" / "my_audio.rttm"
        assert rttm_file.exists()
        assert task.data["rttm_filepath"] == "rttm/my_audio.rttm"
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
        assert (tmp_path / "rttm" / "sess_42.rttm").exists()
        assert task.data["rttm_filepath"] == "rttm/sess_42.rttm"
