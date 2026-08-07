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

"""Tests for the Whisper-first LID selection logic in ``SelectBestLIDPredictionStage``.

Routing under test (see the stage docstring):
  * no predictions               -> skip
  * Whisper == "en"              -> accept Whisper (English)
  * Whisper missing/empty        -> skip
  * SpeechBrain + Canary + Whisper all agree -> use Canary
  * non-Indic Whisper + SpeechBrain agree    -> use Whisper
  * otherwise (disagreement)     -> skip
"""

from nemo_curator.stages.audio.inference.langid_base import LangIDResult
from nemo_curator.stages.audio.text_filtering.select_best_lid_prediction import SelectBestLIDPredictionStage
from nemo_curator.tasks import AudioTask


def test_missing_lid_sets_skipme() -> None:
    stage = SelectBestLIDPredictionStage()
    out = stage.process(AudioTask(data={}))

    assert out.data["source_lang"] == ""
    assert out.data["source_lid_confidence"] == 0.0
    assert out.data["_skipme"] == "skipped due to missing langID predictions."
    assert out.data["additional_notes"]["SelectBestLIDPrediction"] == "skipped (missing predictions)"


def test_whisper_english_accepted() -> None:
    """Whisper predicting English wins outright, regardless of the other models."""
    stage = SelectBestLIDPredictionStage()
    task = AudioTask(
        data={
            "lid": [
                {"SpeechBrainLangID": LangIDResult(language="hi", confidence=0.50, tag="primary")},
                {"IndicCanaryLangID": LangIDResult(language="hi", confidence=0.90, tag="secondary")},
                {"WhisperLangID": LangIDResult(language="en", confidence=0.95, tag="tertiary")},
            ]
        }
    )

    out = stage.process(task)

    assert out.data["source_lang"] == "en"
    assert out.data["source_lid_confidence"] == 0.95
    notes = out.data["additional_notes"]
    assert notes["tertiary_lid_model"] == "whisper"
    assert notes["tertiary_lid_prediction"] == "en"
    assert notes["SelectBestLIDPrediction"] == "used tertiary, English language."
    assert "_skipme" not in out.data


def test_missing_whisper_sets_skipme() -> None:
    """Whisper is required for the agreement checks; its absence is a skip."""
    stage = SelectBestLIDPredictionStage()
    task = AudioTask(
        data={
            "lid": [
                {"SpeechBrainLangID": LangIDResult(language="hi", confidence=0.80, tag="primary")},
                {"IndicCanaryLangID": LangIDResult(language="hi", confidence=0.90, tag="secondary")},
            ]
        }
    )

    out = stage.process(task)

    assert out.data["source_lang"] == ""
    assert out.data["source_lid_confidence"] == 0.0
    assert out.data["_skipme"] == "skipped due to missing or empty whisper langID prediction."
    notes = out.data["additional_notes"]
    assert notes["primary_lid_model"] == "speechbrain"
    assert notes["secondary_lid_model"] == "indic_canary"
    assert notes["SelectBestLIDPrediction"] == "skipped (missing or empty whisper langID prediction)"


def test_empty_whisper_language_sets_skipme() -> None:
    """An empty Whisper language string is treated the same as a missing Whisper result."""
    stage = SelectBestLIDPredictionStage()
    task = AudioTask(
        data={
            "lid": [
                {"SpeechBrainLangID": LangIDResult(language="hi", confidence=0.80, tag="primary")},
                {"WhisperLangID": LangIDResult(language="", confidence=0.0, tag="tertiary")},
            ]
        }
    )

    out = stage.process(task)

    assert out.data["source_lang"] == ""
    assert out.data["source_lid_confidence"] == 0.0
    assert out.data["_skipme"] == "skipped due to missing or empty whisper langID prediction."
    assert (
        out.data["additional_notes"]["SelectBestLIDPrediction"]
        == "skipped (missing or empty whisper langID prediction)"
    )


def test_all_three_agree_uses_canary() -> None:
    """SpeechBrain + Canary + Whisper agreeing on an Indic language -> Canary result."""
    stage = SelectBestLIDPredictionStage()
    task = AudioTask(
        data={
            "lid": [
                {"SpeechBrainLangID": LangIDResult(language="hi", confidence=0.80, tag="primary")},
                {"IndicCanaryLangID": LangIDResult(language="hi", confidence=0.97, tag="secondary")},
                {"WhisperLangID": LangIDResult(language="hi", confidence=0.90, tag="tertiary")},
            ]
        }
    )

    out = stage.process(task)

    assert out.data["source_lang"] == "hi"
    assert out.data["source_lid_confidence"] == 0.97
    notes = out.data["additional_notes"]
    assert notes["primary_lid_prediction"] == "hi"
    assert notes["secondary_lid_prediction"] == "hi"
    assert notes["tertiary_lid_prediction"] == "hi"
    assert notes["SelectBestLIDPrediction"] == "used secondary, agreement between all 3 langID models."
    assert "_skipme" not in out.data


def test_non_indic_agreement_uses_whisper() -> None:
    """A non-Indic Whisper prediction is accepted when SpeechBrain agrees (Canary absent)."""
    stage = SelectBestLIDPredictionStage()
    task = AudioTask(
        data={
            "lid": [
                {"SpeechBrainLangID": LangIDResult(language="fr", confidence=0.70, tag="primary")},
                {"WhisperLangID": LangIDResult(language="fr", confidence=0.88, tag="tertiary")},
            ]
        }
    )

    out = stage.process(task)

    assert out.data["source_lang"] == "fr"
    assert out.data["source_lid_confidence"] == 0.88
    notes = out.data["additional_notes"]
    assert notes["SelectBestLIDPrediction"] == (
        "used tertiary, agreement between primary and tertiary langID models."
    )
    assert "_skipme" not in out.data


def test_indic_without_full_agreement_sets_skipme() -> None:
    """Even when SpeechBrain and Canary agree on an Indic language, a disagreeing
    Whisper prevents the all-3 rule and the row is skipped."""
    stage = SelectBestLIDPredictionStage()
    task = AudioTask(
        data={
            "lid": [
                {"SpeechBrainLangID": LangIDResult(language="hi", confidence=0.80, tag="primary")},
                {"IndicCanaryLangID": LangIDResult(language="hi", confidence=0.97, tag="secondary")},
                {"WhisperLangID": LangIDResult(language="bn", confidence=0.60, tag="tertiary")},
            ]
        }
    )

    out = stage.process(task)

    assert out.data["source_lang"] == "skipped"
    assert out.data["source_lid_confidence"] == 0.0
    assert out.data["_skipme"] == "skipped due to disagreement between langID models."
    assert (
        out.data["additional_notes"]["SelectBestLIDPrediction"]
        == "skipped due to disagreement between langID models."
    )


def test_non_indic_disagreement_sets_skipme() -> None:
    """A non-Indic Whisper prediction that SpeechBrain disagrees with is skipped."""
    stage = SelectBestLIDPredictionStage()
    task = AudioTask(
        data={
            "lid": [
                {"SpeechBrainLangID": LangIDResult(language="de", confidence=0.70, tag="primary")},
                {"WhisperLangID": LangIDResult(language="fr", confidence=0.80, tag="tertiary")},
            ]
        }
    )

    out = stage.process(task)

    assert out.data["source_lang"] == "skipped"
    assert out.data["source_lid_confidence"] == 0.0
    assert out.data["_skipme"] == "skipped due to disagreement between langID models."


def test_ambernet_is_treated_as_speechbrain() -> None:
    """AmberNet fills the same 'primary' slot as SpeechBrain in the routing."""
    stage = SelectBestLIDPredictionStage()
    task = AudioTask(
        data={
            "lid": [
                {"AmberNetLangID": LangIDResult(language="ta", confidence=0.75, tag="primary")},
                {"IndicCanaryLangID": LangIDResult(language="ta", confidence=0.92, tag="secondary")},
                {"WhisperLangID": LangIDResult(language="ta", confidence=0.81, tag="tertiary")},
            ]
        }
    )

    out = stage.process(task)

    assert out.data["source_lang"] == "ta"
    assert out.data["source_lid_confidence"] == 0.92
    notes = out.data["additional_notes"]
    assert notes["primary_lid_model"] == "ambernet"
    assert notes["SelectBestLIDPrediction"] == "used secondary, agreement between all 3 langID models."
    assert "_skipme" not in out.data
