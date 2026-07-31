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

from nemo_curator.stages.audio.inference.langid_base import LangIDResult
from nemo_curator.stages.audio.text_filtering.select_best_lid_prediction import SelectBestLIDPredictionStage
from nemo_curator.tasks import AudioTask


def test_missing_lid_sets_skipme() -> None:
    stage = SelectBestLIDPredictionStage()
    out = stage.process(AudioTask(data={}))

    assert out.data["_skipme"] == "skipped due to missing langID predictions."
    assert out.data["additional_notes"]["SelectBestLIDPrediction"] == "skipped (missing predictions)"


def test_non_indic_uses_speechbrain() -> None:
    stage = SelectBestLIDPredictionStage()
    task = AudioTask(
        data={
            "lid": [
                {"SpeechBrainLangID": LangIDResult(language="en", confidence=0.91, tag="primary")},
                {"IndicCanaryLangID": LangIDResult(language="hi", confidence=1.0, tag="secondary")},
            ]
        }
    )

    out = stage.process(task)

    assert out.data["source_lang"] == "en"
    assert out.data["source_lid_confidence"] == 0.91
    notes = out.data["additional_notes"]
    assert notes["primary_lid_model"] == "speechbrain"
    assert notes["primary_lid_prediction"] == "en"
    assert notes["primary_lid_confidence"] == "0.910"
    assert notes["secondary_lid_model"] == "indic_canary"
    assert notes["SelectBestLIDPrediction"] == "used primary, non-Indic language."
    assert "_skipme" not in out.data


def test_indic_agreement_uses_speechbrain_lang() -> None:
    stage = SelectBestLIDPredictionStage()
    task = AudioTask(
        data={
            "lid": [
                {"SpeechBrainLangID": LangIDResult(language="bn", confidence=0.8, tag="primary")},
                {"IndicCanaryLangID": LangIDResult(language="bn", confidence=1.0, tag="secondary")},
            ]
        }
    )

    out = stage.process(task)

    assert out.data["source_lang"] == "bn"
    assert out.data["source_lid_confidence"] == 0.8
    notes = out.data["additional_notes"]
    assert notes["primary_lid_model"] == "speechbrain"
    assert notes["primary_lid_prediction"] == "bn"
    assert notes["primary_lid_confidence"] == "0.800"
    assert notes["secondary_lid_model"] == "indic_canary"
    assert notes["secondary_lid_prediction"] == "bn"
    assert notes["secondary_lid_confidence"] == "1.000"
    assert notes["SelectBestLIDPrediction"] == (
        "used secondary, agreement between SpeechBrain and Indic Canary langID model."
    )
    assert "_skipme" not in out.data


def test_indic_disagreement_uses_canary_and_sets_skipme() -> None:
    stage = SelectBestLIDPredictionStage()
    task = AudioTask(
        data={
            "lid": [
                {"AmberNetLangID": LangIDResult(language="hi", confidence=0.9, tag="primary")},
                {"IndicCanaryLangID": LangIDResult(language="ta", confidence=1.0, tag="secondary")},
            ]
        }
    )

    out = stage.process(task)

    assert out.data["source_lang"] == "ta"
    assert out.data["source_lid_confidence"] == 1.0
    assert out.data["_skipme"] == ("skipped due to disagreement between primary and secondary langID models.")
    notes = out.data["additional_notes"]
    assert notes["primary_lid_model"] == "ambernet"
    assert notes["primary_lid_prediction"] == "hi"
    assert notes["primary_lid_confidence"] == "0.900"
    assert notes["secondary_lid_model"] == "indic_canary"
    assert notes["secondary_lid_prediction"] == "ta"
    assert notes["secondary_lid_confidence"] == "1.000"
    assert notes["SelectBestLIDPrediction"] == ("used secondary, disagreement between primary and secondary")


def test_indic_without_canary_sets_skipme() -> None:
    stage = SelectBestLIDPredictionStage()
    task = AudioTask(
        data={
            "lid": [
                {"SpeechBrainLangID": LangIDResult(language="hi", confidence=0.9, tag="primary")},
            ]
        }
    )

    out = stage.process(task)

    assert "source_lang" not in out.data
    assert out.data["_skipme"] == "skipped due to missing secondary langID prediction."
    assert out.data["additional_notes"]["SelectBestLIDPrediction"] == "skipped (missing secondary)"
