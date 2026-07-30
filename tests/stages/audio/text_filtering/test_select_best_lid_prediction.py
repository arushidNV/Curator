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

from nemo_curator.stages.audio.text_filtering.select_best_lid_prediction import SelectBestLIDPredictionStage
from nemo_curator.tasks import AudioTask


def _task(**data: object) -> AudioTask:
    return AudioTask(data=data, task_id="test", dataset_name="test")


def test_selects_indic_canary_when_confidence_is_higher() -> None:
    stage = SelectBestLIDPredictionStage()
    task = _task(
        speechbrain_language="hi",
        speechbrain_language_confidence=0.7,
        indic_canary_language="mr",
        indic_canary_language_confidence=1.0,
    )

    result = stage.process(task)

    assert result.data["language"] == "mr"
    assert result.data["language_confidence"] == 1.0
    assert result.data["language_source"] == "indic_canary"


def test_falls_back_to_primary_when_canary_has_no_language() -> None:
    stage = SelectBestLIDPredictionStage()
    task = _task(
        speechbrain_language="en",
        speechbrain_language_confidence=0.9,
        indic_canary_language="",
        indic_canary_language_confidence=0.0,
    )

    result = stage.process(task)

    assert result.data["language"] == "en"
    assert result.data["language_confidence"] == 0.9
    assert result.data["language_source"] == "primary"


def test_primary_wins_when_its_confidence_is_higher() -> None:
    stage = SelectBestLIDPredictionStage()
    task = _task(
        speechbrain_language="en",
        speechbrain_language_confidence=0.95,
        indic_canary_language="hi",
        indic_canary_language_confidence=0.8,
    )

    result = stage.process(task)

    assert result.data["language"] == "en"
    assert result.data["language_source"] == "primary"
