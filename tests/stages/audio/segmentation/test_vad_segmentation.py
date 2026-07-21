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

import pickle
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.audio.segmentation.silero_tensorrt import TensorRTSileroModel
from nemo_curator.stages.audio.segmentation.vad_segmentation import VADSegmentationStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@pytest.mark.gpu
class TestVADSegmentationStage:
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_onnx_backend_loads_official_silero_model(self, mock_load_vad: MagicMock) -> None:
        mock_model = MagicMock()
        mock_load_vad.return_value = mock_model

        stage = VADSegmentationStage(backend="onnx")
        stage.setup()

        mock_load_vad.assert_called_once_with(onnx=True, opset_version=16)
        assert stage._device == torch.device("cpu")
        mock_model.to.assert_not_called()

    def test_rejects_unknown_backend(self) -> None:
        with pytest.raises(ValueError, match="Unsupported Silero backend"):
            VADSegmentationStage(backend="openvino")  # type: ignore[arg-type]

    def test_tensorrt_backend_requires_engine_path(self) -> None:
        with pytest.raises(ValueError, match="tensorrt_engine_path is required"):
            VADSegmentationStage(backend="tensorrt")

    @patch("nemo_curator.stages.audio.segmentation.silero_tensorrt.TensorRTSileroModel")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.torch.cuda.is_available", return_value=True)
    def test_tensorrt_backend_loads_persistent_gpu_model(
        self,
        mock_cuda: MagicMock,
        mock_trt_model: MagicMock,
    ) -> None:
        assert mock_cuda.return_value is True
        model = MagicMock()
        mock_trt_model.return_value = model
        stage = VADSegmentationStage(
            backend="tensorrt",
            tensorrt_engine_path="/models/silero.plan",
            resources=Resources(cpus=1, gpus=1),
        )

        stage.setup()

        mock_trt_model.assert_called_once_with("/models/silero.plan")
        assert stage._vad_model is model
        assert stage._device == torch.device("cuda")

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_process_returns_segments(self, mock_load_vad: MagicMock, mock_get_ts: MagicMock) -> None:
        mock_model = MagicMock()
        mock_load_vad.return_value = mock_model

        sr = 48000
        mock_get_ts.return_value = [
            {"start": 0, "end": sr * 3},
            {"start": sr * 5, "end": sr * 8},
        ]

        waveform = torch.randn(1, sr * 10)
        task = AudioTask(
            data={"waveform": waveform, "sample_rate": sr},
            task_id="test",
            dataset_name="test",
        )

        stage = VADSegmentationStage(min_duration_sec=1.0, max_duration_sec=30.0)
        stage.setup()
        result = stage.process(task)

        assert isinstance(result, list)
        assert len(result) == 2
        for seg in result:
            assert isinstance(seg, AudioTask)
            assert "waveform" in seg.data
            assert "start_ms" in seg.data
            assert "end_ms" in seg.data
            assert "segment_num" in seg.data
            assert "duration_sec" in seg.data

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_process_output_keys(self, mock_load_vad: MagicMock, mock_get_ts: MagicMock) -> None:
        mock_load_vad.return_value = MagicMock()

        sr = 48000
        mock_get_ts.return_value = [{"start": 0, "end": sr * 5}]

        waveform = torch.randn(1, sr * 10)
        task = AudioTask(
            data={"waveform": waveform, "sample_rate": sr},
            task_id="test",
            dataset_name="test",
        )

        stage = VADSegmentationStage(min_duration_sec=1.0)
        stage.setup()
        result = stage.process(task)

        assert result[0].data["start_ms"] == 0
        assert result[0].data["segment_num"] == 0
        assert result[0].data["duration_sec"] > 0
        assert result[0].data["sample_rate"] == sr

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_empty_speech_returns_empty(self, mock_load_vad: MagicMock, mock_get_ts: MagicMock) -> None:
        mock_load_vad.return_value = MagicMock()
        mock_get_ts.return_value = []

        waveform = torch.randn(1, 48000 * 5)
        task = AudioTask(
            data={"waveform": waveform, "sample_rate": 48000},
            task_id="test",
            dataset_name="test",
        )

        stage = VADSegmentationStage()
        stage.setup()
        result = stage.process(task)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].data.get("vad_empty") is True
        # Full-file waveform must be dropped so downstream GPU stages skip it (avoids OOM).
        assert result[0].data.get("waveform") is None
        assert result[0].data.get("duration_sec") == pytest.approx(5.0, abs=0.1)

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_read_error_passthrough(self, mock_load_vad: MagicMock, mock_get_ts: MagicMock) -> None:
        mock_load_vad.return_value = MagicMock()

        task = AudioTask(
            data={
                "read_error": True,
                "audio_filepath": "s3://bucket/broken.m4a",
                "original_file": "s3://bucket/broken.m4a",
            },
            task_id="test",
            dataset_name="test",
        )

        stage = VADSegmentationStage()
        stage.setup()
        result = stage.process(task)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].data.get("read_error") is True
        assert result[0].data.get("waveform") is None
        mock_get_ts.assert_not_called()

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_vad_exception_emits_read_error_placeholder(
        self, mock_load_vad: MagicMock, mock_get_ts: MagicMock
    ) -> None:
        mock_load_vad.return_value = MagicMock()
        # A crash during segmentation must not silently drop the recording, or its
        # shard would never reach shard_total and .jsonl.done would never be written.
        mock_get_ts.side_effect = RuntimeError("boom")

        waveform = torch.randn(1, 48000 * 5)
        task = AudioTask(
            data={
                "waveform": waveform,
                "sample_rate": 48000,
                "audio_filepath": "s3://bucket/x.wav",
                "original_file": "s3://bucket/x.wav",
            },
            task_id="test",
            dataset_name="test",
        )

        stage = VADSegmentationStage(min_duration_sec=1.0)
        stage.setup()
        result = stage.process(task)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].data.get("read_error") is True
        assert result[0].data.get("waveform") is None

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_segment_numbering(self, mock_load_vad: MagicMock, mock_get_ts: MagicMock) -> None:
        mock_load_vad.return_value = MagicMock()

        sr = 48000
        mock_get_ts.return_value = [
            {"start": 0, "end": sr * 2},
            {"start": sr * 3, "end": sr * 5},
            {"start": sr * 6, "end": sr * 8},
        ]

        waveform = torch.randn(1, sr * 10)
        task = AudioTask(
            data={"waveform": waveform, "sample_rate": sr},
            task_id="test",
            dataset_name="test",
        )

        stage = VADSegmentationStage(min_duration_sec=0.5)
        stage.setup()
        result = stage.process(task)

        assert len(result) == 3
        for i, seg in enumerate(result):
            assert seg.data["segment_num"] == i

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_missing_waveform_and_filepath_skipped(self, mock_load_vad: MagicMock, mock_get_ts: MagicMock) -> None:
        mock_load_vad.return_value = MagicMock()

        task = AudioTask(
            data={"some_key": "value"},
            task_id="test",
            dataset_name="test",
        )

        stage = VADSegmentationStage()
        stage.setup()
        result = stage.process(task)

        assert isinstance(result, list)
        assert len(result) == 0

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_nested_mode_returns_single_task(self, mock_load_vad: MagicMock, mock_get_ts: MagicMock) -> None:
        mock_load_vad.return_value = MagicMock()

        sr = 48000
        mock_get_ts.return_value = [
            {"start": 0, "end": sr * 3},
            {"start": sr * 5, "end": sr * 8},
        ]

        waveform = torch.randn(1, sr * 10)
        task = AudioTask(
            data={"waveform": waveform, "sample_rate": sr},
            task_id="test",
            dataset_name="test",
        )

        stage = VADSegmentationStage(min_duration_sec=1.0, nested=True)
        stage.setup()
        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert "segments" in result.data
        assert len(result.data["segments"]) == 2
        for seg in result.data["segments"]:
            assert "waveform" in seg
            assert "start_ms" in seg
            assert "end_ms" in seg
            assert "segment_num" in seg
            assert "duration_sec" in seg
            assert "original_file" in seg

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_nested_mode_no_speech_returns_task_with_empty_segments(
        self, mock_load_vad: MagicMock, mock_get_ts: MagicMock
    ) -> None:
        mock_load_vad.return_value = MagicMock()
        mock_get_ts.return_value = []

        waveform = torch.randn(1, 48000 * 5)
        task = AudioTask(
            data={"waveform": waveform, "sample_rate": 48000},
            task_id="test",
            dataset_name="test",
        )

        stage = VADSegmentationStage(nested=True)
        stage.setup()
        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert result.data["segments"] == []

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_nested_mode_ray_stage_spec_no_fanout(self, mock_load_vad: MagicMock, mock_get_ts: MagicMock) -> None:
        stage = VADSegmentationStage(nested=True)
        spec = stage.ray_stage_spec()
        assert spec == {}

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_non_nested_mode_ray_stage_spec_has_fanout(self, mock_load_vad: MagicMock, mock_get_ts: MagicMock) -> None:
        stage = VADSegmentationStage(nested=False)
        spec = stage.ray_stage_spec()
        assert spec[RayStageSpecKeys.IS_FANOUT_STAGE] is True

    def test_pickling(self) -> None:
        stage = VADSegmentationStage(min_duration_sec=2.0, threshold=0.6, backend="onnx")
        pickled = pickle.dumps(stage)
        restored = pickle.loads(pickled)  # noqa: S301
        assert restored.min_duration_sec == 2.0
        assert restored.threshold == 0.6
        assert restored.backend == "onnx"
        assert restored._vad_model is None


class _FakeSession:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def infer(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        state = inputs["state"]
        self.batch_sizes.append(state.shape[0])
        return {"output": state[:, 0, :1] + 0.25, "stateN": state + 1}


def test_tensorrt_scheduler_compacts_finished_recordings() -> None:
    model = TensorRTSileroModel.__new__(TensorRTSileroModel)
    model.session = _FakeSession()

    probabilities = model.infer_probabilities([torch.zeros(512), torch.zeros(1024), torch.zeros(1536)])

    assert model.session.batch_sizes == [3, 2, 1]
    torch.testing.assert_close(probabilities[0], torch.tensor([0.25]))
    torch.testing.assert_close(probabilities[1], torch.tensor([0.25, 1.25]))
    torch.testing.assert_close(probabilities[2], torch.tensor([0.25, 1.25, 2.25]))


class _FakeBatchedModel:
    def infer_probabilities(self, waveforms: list[torch.Tensor]) -> list[torch.Tensor]:
        return [torch.tensor([0.0, 0.9, 0.9, 0.0]) for _ in waveforms]


def test_tensorrt_stage_batches_recordings_and_preserves_order() -> None:
    stage = VADSegmentationStage(
        backend="tensorrt",
        tensorrt_engine_path="silero.plan",
        min_duration_sec=0.01,
        min_interval_ms=1,
        speech_pad_ms=0,
        nested=True,
        resources=Resources(cpus=1, gpus=1),
    )
    stage._vad_model = _FakeBatchedModel()
    stage._device = torch.device("cpu")
    tasks = [
        AudioTask(
            data={"waveform": torch.zeros(1, 4 * 512), "sample_rate": 16000},
            task_id=f"task-{index}",
            dataset_name="test",
        )
        for index in range(3)
    ]

    results = stage.process_batch(tasks)

    assert [task.task_id for task in results] == ["task-0", "task-1", "task-2"]
    assert all(len(task.data["segments"]) == 1 for task in results)


def test_vad_backend_defaults_are_preserved() -> None:
    stage = VADSegmentationStage()
    assert stage.backend == "torch"
    assert stage.batch_size == 8
    assert stage.resources.gpu_memory_gb == 4.0
