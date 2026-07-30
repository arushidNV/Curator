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

"""Dual-agreement language-ID selection between a primary model and Indic Canary.

Assigns ``source_lang`` only when both models predict the same language code.
On disagreement, sets ``_skipme`` and records both predictions in ``additional_notes``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nemo_curator.stages.audio.pipeline_utils import set_note
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_DISAGREEMENT_SKIP_REASON = "skipped due to disagreement between primary and secondary langID model."


def _normalize_lang_code(raw: object) -> str:
    """Normalize a LID prediction to a lowercase ISO code.

    SpeechBrain may return ``"ta: Tamil"`` — keep only the code before the colon.
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    return text.split(":", 1)[0].strip().lower()


@dataclass
class SelectBestLIDPredictionStage(ProcessingStage[AudioTask, AudioTask]):
    """Keep ``source_lang`` only when primary and secondary LID predictions agree.

    Intermediate primary/secondary task keys are removed from the output. Model
    identity and per-model predictions are recorded in ``additional_notes``,
    mirroring how ASR records ``primary_model`` / ``recovery_model`` /
    ``SelectBestPrediction``.

    Args:
        primary_language_key: Task data key holding the primary (SpeechBrain/AmberNet) language prediction.
        primary_confidence_key: Task data key holding the primary model's confidence score.
        secondary_language_key: Task data key holding the secondary (Indic Canary) language prediction.
        secondary_confidence_key: Task data key holding the secondary model's confidence score.
        output_key: Task data key for the finalized language (default ``source_lang``).
        confidence_key: Task data key for the finalized confidence score (default ``source_lid_confidence``).
        skip_me_key: Task data key for the shared skip flag.
        notes_key: Task data key for pipeline notes.
        primary_lid_model_label: Value written to ``additional_notes["primary_lid_model"]``.
        secondary_lid_model_label: Value written to ``additional_notes["secondary_lid_model"]``.
    """

    primary_language_key: str = "primary_lang_pred"
    primary_confidence_key: str = "primary_lid_confidence"
    secondary_language_key: str = "secondary_lang_pred"
    secondary_confidence_key: str = "secondary_lid_confidence"
    output_key: str = "source_lang"
    confidence_key: str = "source_lid_confidence"
    skip_me_key: str = "_skipme"
    notes_key: str = "additional_notes"
    primary_lid_model_label: str = "speechbrain"
    secondary_lid_model_label: str = "indic_canary"
    name: str = "SelectBestLIDPrediction"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [
            self.primary_language_key,
            self.secondary_language_key,
        ]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_key, self.confidence_key, self.notes_key, self.skip_me_key]

    def process(self, task: AudioTask) -> AudioTask:
        primary_raw = task.data.pop(self.primary_language_key, "")
        primary_confidence = float(task.data.pop(self.primary_confidence_key, 0.0) or 0.0)
        secondary_raw = task.data.pop(self.secondary_language_key, "")
        secondary_confidence = float(task.data.pop(self.secondary_confidence_key, 0.0) or 0.0)

        primary_lang = _normalize_lang_code(primary_raw)
        secondary_lang = _normalize_lang_code(secondary_raw)

        set_note(task.data, "primary_lid_model", self.primary_lid_model_label, self.notes_key)
        set_note(task.data, "secondary_lid_model", self.secondary_lid_model_label, self.notes_key)
        set_note(task.data, "primary_lid_prediction", primary_lang, self.notes_key)
        set_note(task.data, "primary_lid_confidence", f"{primary_confidence:.3f}", self.notes_key)
        set_note(task.data, "secondary_lid_prediction", secondary_lang, self.notes_key)
        set_note(task.data, "secondary_lid_confidence", f"{secondary_confidence:.3f}", self.notes_key)

        if primary_lang and primary_lang == secondary_lang:
            task.data[self.output_key] = primary_lang
            task.data[self.confidence_key] = primary_confidence
            set_note(task.data, self.name, f"agreement (lang={primary_lang})", self.notes_key)
            return task

        task.data[self.skip_me_key] = _DISAGREEMENT_SKIP_REASON
        set_note(task.data, self.name, "skipped (disagreement)", self.notes_key)
        return task
