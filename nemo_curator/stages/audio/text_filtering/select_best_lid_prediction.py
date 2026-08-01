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

"""Select the best language-ID prediction from SpeechBrain/AmberNet and Indic Canary.

Reads ``LangIDResult`` entries from ``task.data[lid_key]`` (tagged ``primary`` /
``secondary``). Routing:

- SpeechBrain predicted a **non-Indic** language → keep the SpeechBrain prediction.
- SpeechBrain predicted an **Indic** language → use Indic Canary as ``source_lang``.
  If both models agree on the language code, record an agreement note; otherwise
  record disagreement and set ``_skipme``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nemo_curator.stages.audio.inference.langid_base import LangIDResult
from nemo_curator.stages.audio.pipeline_utils import set_note
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_DEFAULT_INDIC_LANGUAGES: frozenset[str] = frozenset(
    {
        "hi",
        "ta",
        "bn",
        "ur",
        "gu",
        "mr",
        "ml",
        "kn",
        "te",
        "or",
        "as",
        "pa",
        "ne",
        "sa",
        "sd",
        "si",
        "kok",
        "mai",
        "doi",
        "ks",
        "mni",
        "sat",
        "brx",
        "bo",
    }
)

_MODEL_LABELS = {
    "SpeechBrainLangID": "speechbrain",
    "AmberNetLangID": "ambernet",
    "IndicCanaryLangID": "indic_canary",
}


def _model_label(model_name: str) -> str:
    return _MODEL_LABELS.get(model_name, model_name)


@dataclass
class SelectBestLIDPredictionStage(ProcessingStage[AudioTask, AudioTask]):
    """Route LID using primary Indic detection, with agreement notes vs Canary.

    Args:
        lid_key: Task data key holding the list of LID results.
        output_key: Task data key for the finalized language (default ``source_lang``).
        confidence_key: Task data key for the finalized confidence score.
        skip_me_key: Task data key for the shared skip flag.
        notes_key: Task data key for pipeline notes.
        indic_languages: Language codes treated as Indic (route to Canary).
    """

    lid_key: str = "lid"
    output_key: str = "source_lang"
    confidence_key: str = "source_lid_confidence"
    skip_me_key: str = "_skipme"
    notes_key: str = "additional_notes"
    indic_languages: frozenset[str] = field(default_factory=lambda: _DEFAULT_INDIC_LANGUAGES)
    name: str = "SelectBestLIDPrediction"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.lid_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_key, self.notes_key, self.skip_me_key]

    def _add_notes(self, task: AudioTask, lid_entries: list[dict[str, LangIDResult]]) -> None:
        for entry in lid_entries:
            for model_name, result in entry.items():
                if not isinstance(result, LangIDResult):
                    continue
                tag = result.tag
                set_note(task.data, f"{tag}_lid_model", _model_label(model_name), self.notes_key)
                set_note(task.data, f"{tag}_lid_prediction", result.language, self.notes_key)
                set_note(
                    task.data,
                    f"{tag}_lid_confidence",
                    f"{float(result.confidence):.3f}",
                    self.notes_key,
                )

    def process(self, task: AudioTask) -> AudioTask:  # noqa: C901
        lid_entries = task.data.pop(self.lid_key, [])
        if not lid_entries:
            task.data[self.skip_me_key] = "skipped due to missing langID predictions."
            set_note(task.data, self.name, "skipped (missing predictions)", self.notes_key)
            return task

        self._add_notes(task, lid_entries)

        sb_result: LangIDResult | None = None
        canary_result: LangIDResult | None = None
        for entry in lid_entries:
            for model_name, result in entry.items():
                if not isinstance(result, LangIDResult):
                    raise ValueError(f"Invalid LID result: {result}")
                if model_name in {"SpeechBrainLangID", "AmberNetLangID"}:
                    sb_result = result
                elif model_name == "IndicCanaryLangID":
                    canary_result = result
                else:
                    raise ValueError(f"Invalid model name: {model_name}")

        if sb_result is None or len(sb_result.language)==0:
            task.data[self.output_key] = ""
            task.data[self.notes_key][self.confidence_key] = 0.0
            task.data[self.skip_me_key] = "skipped due to missing or empty primary langID prediction."
            set_note(task.data, self.name, "skipped (missing or empty primary langID prediction)", self.notes_key)
            return task

        if sb_result.language in self.indic_languages:
            if canary_result is None:
                task.data[self.output_key] = sb_result.language
                task.data[self.notes_key][self.confidence_key] = float(sb_result.confidence)
                set_note(
                    task.data,
                    self.name,
                    f"used {sb_result.tag}, Indic language.",
                    self.notes_key,
                )
                return task
            if canary_result.language == sb_result.language:
                task.data[self.output_key] = canary_result.language
                task.data[self.notes_key][self.confidence_key] = float(canary_result.confidence)
                set_note(
                    task.data,
                    self.name,
                    f"used {canary_result.tag}, agreement between SpeechBrain and Indic Canary langID model.",
                    self.notes_key,
                )
                return task
            task.data[self.output_key] = canary_result.language
            task.data[self.notes_key][self.confidence_key] = float(canary_result.confidence)
            set_note(
                task.data,
                self.name,
                f"used {canary_result.tag}, disagreement between {sb_result.tag} and {canary_result.tag}",
                self.notes_key,
            )
            task.data[self.skip_me_key] = (
                f"skipped due to disagreement between {sb_result.tag} and {canary_result.tag} langID models."
            )
            return task

        task.data[self.output_key] = sb_result.language
        task.data[self.notes_key][self.confidence_key] = float(sb_result.confidence)
        set_note(task.data, self.name, f"used {sb_result.tag}, non-Indic language.", self.notes_key)
        return task
