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

from dataclasses import dataclass, field

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class SelectBestLIDPredictionStage(ProcessingStage[AudioTask, AudioTask]):
    """Select the higher-confidence result from primary and Indic Canary LID."""

    speechbrain_language_key: str = "speechbrain_language"
    speechbrain_confidence_key: str = "speechbrain_language_confidence"
    indic_canary_language_key: str = "indic_canary_language"
    indic_canary_confidence_key: str = "indic_canary_language_confidence"
    output_key: str = "language"
    confidence_key: str = "language_confidence"
    source_key: str = "language_source"
    name: str = "SelectBestLIDPrediction"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            self.speechbrain_language_key,
            self.speechbrain_confidence_key,
            self.indic_canary_language_key,
            self.indic_canary_confidence_key,
        ]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.output_key, self.confidence_key, self.source_key]

    def process(self, task: AudioTask) -> AudioTask:
        primary_language = str(task.data.get(self.speechbrain_language_key, "") or "")
        primary_confidence = float(task.data.get(self.speechbrain_confidence_key, 0.0) or 0.0)
        indic_language = str(task.data.get(self.indic_canary_language_key, "") or "")
        indic_confidence = float(task.data.get(self.indic_canary_confidence_key, 0.0) or 0.0)

        if indic_language and (not primary_language or indic_confidence >= primary_confidence):
            task.data[self.output_key] = indic_language
            task.data[self.confidence_key] = indic_confidence
            task.data[self.source_key] = "indic_canary"
        else:
            task.data[self.output_key] = primary_language
            task.data[self.confidence_key] = primary_confidence
            task.data[self.source_key] = "primary"
        return task
