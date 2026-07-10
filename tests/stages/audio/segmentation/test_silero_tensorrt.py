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

import pytest
import torch
from silero_vad import get_speech_timestamps

from nemo_curator.stages.audio.segmentation.silero_tensorrt import (
    TensorRTSileroModel,
    probabilities_to_speech_timestamps,
)
from nemo_curator.stages.audio.segmentation.vad_segmentation import VADSegmentationStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


class _ProbabilityModel:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = probabilities
        self.index = 0

    def reset_states(self) -> None:
        self.index = 0

    def __call__(self, _audio: torch.Tensor, _sampling_rate: int) -> torch.Tensor:
        probability = self.probabilities[self.index]
        self.index += 1
        return torch.tensor([[probability]], dtype=torch.float32)


@pytest.mark.parametrize(
    ("probabilities", "audio_length"),
    [
        ([0.0, 0.8, 0.9, 0.1, 0.0], 5 * 512),
        ([0.7] * 20, 20 * 512 - 73),
        ([0.0, 0.8, 0.2, 0.8, 0.2, 0.0, 0.9, 0.9], 8 * 512),
    ],
)
def test_probability_postprocessing_matches_official_silero(
    probabilities: list[float], audio_length: int
) -> None:
    kwargs = {
        "sampling_rate": 16000,
        "threshold": 0.5,
        "min_speech_duration_ms": 32,
        "max_speech_duration_s": 0.25,
        "min_silence_duration_ms": 32,
        "speech_pad_ms": 16,
    }
    expected = get_speech_timestamps(
        torch.zeros(audio_length),
        _ProbabilityModel(probabilities),
        **kwargs,
    )
    actual = probabilities_to_speech_timestamps(
        torch.tensor(probabilities),
        audio_length,
        **kwargs,
    )
    assert actual == expected


class _FakeSession:
    device = torch.device("cpu")
    input_names = ("input", "state")
    output_names = ("output", "stateN")

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def infer(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        state = inputs["state"]
        self.batch_sizes.append(state.shape[0])
        return {
            "output": state[:, 0, :1] + 0.25,
            "stateN": state + 1,
        }


def test_infer_probabilities_compacts_finished_recordings() -> None:
    model = TensorRTSileroModel.__new__(TensorRTSileroModel)
    model.session = _FakeSession()
    model.sample_rate = 16000
    model.context_size = 64
    model.state_size = 128
    model.input_name = "input"
    model.state_input_name = "state"
    model.output_name = "output"
    model.state_output_name = "stateN"

    probabilities = model.infer_probabilities(
        [
            torch.zeros(512),
            torch.zeros(1024),
            torch.zeros(1536),
        ]
    )

    assert model.session.batch_sizes == [3, 2, 1]
    torch.testing.assert_close(probabilities[0], torch.tensor([0.25]))
    torch.testing.assert_close(probabilities[1], torch.tensor([0.25, 1.25]))
    torch.testing.assert_close(probabilities[2], torch.tensor([0.25, 1.25, 2.25]))


class _FakeBatchedModel:
    def __init__(self) -> None:
        self.batch_size = 0
        self.inference_count = 0

    def infer_probabilities(self, waveforms: list[torch.Tensor]) -> list[torch.Tensor]:
        self.batch_size = len(waveforms)
        self.inference_count += 4
        return [torch.tensor([0.0, 0.9, 0.9, 0.0]) for _ in waveforms]


def test_vad_stage_batches_recordings_and_preserves_task_order() -> None:
    stage = VADSegmentationStage(
        backend="tensorrt",
        tensorrt_engine_path="silero.plan",
        min_duration_sec=0.01,
        min_interval_ms=1,
        speech_pad_ms=0,
        nested=True,
        resources=Resources(cpus=1, gpus=1),
    )
    model = _FakeBatchedModel()
    stage._vad_model = model
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

    assert model.batch_size == 3
    assert [task.task_id for task in results] == ["task-0", "task-1", "task-2"]
    assert all(len(task.data["segments"]) == 1 for task in results)


def test_tensorrt_vad_uses_one_persistent_worker() -> None:
    stage = VADSegmentationStage(backend="tensorrt", tensorrt_engine_path="silero.plan")
    assert stage.num_workers() == 1
