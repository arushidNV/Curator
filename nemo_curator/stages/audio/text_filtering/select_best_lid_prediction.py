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

"""Select the best language-identification prediction from SpeechBrain and Indic Canary.

Selection logic:
- If SpeechBrain predicted a non-Indic language → use SpeechBrain prediction.
- If SpeechBrain predicted an Indic language (hi, ta, bn, or) → use Indic Canary prediction,
  which has higher accuracy for Indic languages.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nemo_curator.stages.audio.pipeline_utils import set_note
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

# Language codes that SpeechBrain recognises as Indic and should be routed to Indic Canary.
_DEFAULT_INDIC_LANGUAGES: frozenset[str] = frozenset({
    "hi", # Hindi
    "ta", # Tamil
    "bn", # Bengali
    "ur", # Urdu
    "gu", # Gujarati
    "mr", # Marathi
    "ml", # Malayalam
    "kn", # Kannada
    "te", # Telugu
    "or", # Odia
    "as", # Assamese
    "pa", # Punjabi
    "ne", # Nepali
    "sa", # Sanskrit
    "sd", # Sindhi
    "si", # Sinhala
    "kok", # Konkani
    "mai", # Maithili
    "doi", # Dogri
    "ks", # Kashmiri
    "mni", # Manipuri (Meitei)
    "sat", # Santali
    "brx", # Bodo
    "bo", # Tibetan
})


@dataclass
class SelectBestLIDPredictionStage(ProcessingStage[AudioTask, AudioTask]):
    """Select the best LID prediction between SpeechBrain and Indic Canary.

    Routing rule:
    - SpeechBrain predicted a **non-Indic** language → keep SpeechBrain result.
    - SpeechBrain predicted an **Indic** language (``indic_languages``) → use Indic Canary,
      which is purpose-built for Indic speech and gives higher accuracy.

    Args:
        primary_language_key: Task data key holding the primary (SpeechBrain/AmberNet) language prediction.
        primary_confidence_key: Task data key holding the primary model's confidence score.
        secondary_language_key: Task data key holding the secondary (Indic Canary) language prediction.
        secondary_confidence_key: Task data key holding the secondary model's confidence score.
        output_key: Task data key for the finalized language (default ``source_lang``).
        confidence_key: Task data key for the finalized confidence score (default ``source_lid_confidence``).
        model_note_key: ``additional_notes`` field recording which model was chosen
            (``"speechbrain"`` or ``"indic_canary"``).
        notes_key: Task data key for pipeline notes.
        indic_languages: Set of language codes (lowercase) considered Indic and routed to Indic Canary.
    """

    primary_language_key: str = "primary_lang_pred"
    primary_confidence_key: str = "primary_lid_confidence"
    secondary_language_key: str = "secondary_lang_pred"
    secondary_confidence_key: str = "secondary_lid_confidence"
    output_key: str = "source_lang"
    confidence_key: str = "source_lid_confidence"
    model_note_key: str = "audio_language_id_model"
    notes_key: str = "additional_notes"
    indic_languages: frozenset[str] = field(default_factory=lambda: _DEFAULT_INDIC_LANGUAGES)
    name: str = "SelectBestLIDPrediction"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [
            self.primary_language_key,
            self.secondary_language_key,
        ]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_key, self.confidence_key, self.notes_key]

    def process(self, task: AudioTask) -> AudioTask:
        sb_raw = str(task.data.get(self.primary_language_key, "") or "").strip()
        # SpeechBrain may return "ta: Tamil" — extract just the code before the colon.
        sb_lang = sb_raw.split(":")[0].strip().lower()
        sb_confidence = float(task.data.get(self.primary_confidence_key, 0.0) or 0.0)

        canary_lang = str(task.data.get(self.secondary_language_key, "") or "").strip()
        canary_confidence = float(task.data.get(self.secondary_confidence_key, 0.0) or 0.0)

        if sb_lang in self.indic_languages:
            # SpeechBrain flagged an Indic language; defer to the specialist model.
            task.data[self.output_key] = canary_lang
            task.data[self.confidence_key] = canary_confidence
            set_note(task.data, self.model_note_key, "indic_canary", self.notes_key)
        else:
            # Non-Indic language detected by SpeechBrain; keep its prediction.
            task.data[self.output_key] = sb_lang
            task.data[self.confidence_key] = sb_confidence
            set_note(task.data, self.model_note_key, "speechbrain", self.notes_key)

        return task
