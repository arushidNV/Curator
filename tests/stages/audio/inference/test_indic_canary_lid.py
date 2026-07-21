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

"""CPU structural tests for the Indic Canary LangID stage."""

from types import SimpleNamespace

import numpy as np
import pytest

from nemo_curator.stages.audio.inference.indic_canary_lid import IndicCanaryLangIDStage
from nemo_curator.tasks import AudioTask


def _make_task(n: int = 16000, sr: int = 16000) -> AudioTask:
    return AudioTask(
        data={
            "audio_filepath": "/test/audio.wav",
            "waveform": np.zeros(n, dtype=np.float32),
            "sample_rate": sr,
        }
    )


class TestStageContract:
    def test_inputs_outputs(self) -> None:
        stage = IndicCanaryLangIDStage(engine_dir="canary_engine")
        assert stage.inputs() == (["data"], ["waveform", "sample_rate"])
        assert stage.outputs() == (["data"], ["language", "language_confidence"])

    def test_process_batch_without_setup_raises(self) -> None:
        stage = IndicCanaryLangIDStage(engine_dir="canary_engine")
        with pytest.raises(RuntimeError, match="setup"):
            stage.process_batch([_make_task()])

    def test_empty_batch_returns_empty(self) -> None:
        stage = IndicCanaryLangIDStage(engine_dir="canary_engine")
        assert stage.process_batch([]) == []

    def test_process_delegates_to_process_batch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stage = IndicCanaryLangIDStage(engine_dir="canary_engine")
        task = _make_task()

        def fake_process_batch(tasks: list[AudioTask]) -> list[AudioTask]:
            tasks[0].data["language"] = "hi"
            tasks[0].data["language_confidence"] = 1.0
            return tasks

        monkeypatch.setattr(stage, "process_batch", fake_process_batch)

        out = stage.process(task)

        assert out.data["language"] == "hi"
        assert out.data["language_confidence"] == 1.0


class TestSetupOnNode:
    def test_missing_engine_dir_arg_raises(self) -> None:
        stage = IndicCanaryLangIDStage(engine_dir="")
        with pytest.raises(ValueError, match="engine_dir"):
            stage.setup_on_node()

    def test_missing_files_raises(self, tmp_path) -> None:  # noqa: ANN001
        stage = IndicCanaryLangIDStage(engine_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError, match="missing required file"):
            stage.setup_on_node()


class TestProcessBatch:
    def test_identifies_valid_segments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stage = IndicCanaryLangIDStage(engine_dir="canary_engine", min_duration_sec=0.1)
        stage.model = object()

        def fake_identify_batch(audio_signals, audio_lengths):  # noqa: ANN001, ANN202
            assert len(audio_signals) == 2
            assert audio_lengths == [16000, 20000]
            return ["hi", "ta"], [1.0, 1.0]

        monkeypatch.setattr(stage, "_identify_batch", fake_identify_batch)

        out = stage.process_batch([_make_task(16000), _make_task(20000)])

        assert out[0].data["language"] == "hi"
        assert out[0].data["language_confidence"] == 1.0
        assert out[1].data["language"] == "ta"

    def test_short_segments_are_marked_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stage = IndicCanaryLangIDStage(engine_dir="canary_engine", min_duration_sec=2.0)
        stage.model = object()

        def fake_identify_batch(_audio_signals, _audio_lengths):  # noqa: ANN001, ANN202
            msg = "short segments should not reach the model"
            raise AssertionError(msg)

        monkeypatch.setattr(stage, "_identify_batch", fake_identify_batch)

        out = stage.process_batch([_make_task(16000)])

        assert out[0].data["language"] == ""
        assert out[0].data["language_confidence"] == 0.0


class TestTokenHelpers:
    def _stage_with_tokenizer(self, candidate_langs: list[str] | None = None) -> IndicCanaryLangIDStage:
        stage = IndicCanaryLangIDStage(engine_dir="canary_engine", candidate_langs=candidate_langs)
        tokenizer = SimpleNamespace(
            prompt_format="canary1",
            eos_id=3,
            pad_id=2,
            id_to_token={
                4: "<|startoftranscript|>",
                5: "<|pnc|>",
                8: "<|itn|>",
                89: "<|hi|>",
                185: "<|ta|>",
                213: "<|0|>",
            },
            encode=lambda prompt: [4] if prompt == "<|startoftranscript|>" else [7, 4, 18],
        )
        stage.model = SimpleNamespace(tokenizer=tokenizer)
        stage._language_by_token_id = stage._collect_language_token_ids()
        return stage

    def test_collects_language_tokens_and_excludes_controls(self) -> None:
        stage = self._stage_with_tokenizer()
        assert stage._language_by_token_id == {89: "hi", 185: "ta"}

    def test_candidate_langs_restricts_outputs(self) -> None:
        stage = self._stage_with_tokenizer(candidate_langs=["hi"])
        assert stage._language_by_token_id == {89: "hi"}

    def test_lid_prompt_canary1(self) -> None:
        stage = self._stage_with_tokenizer()
        assert stage._lid_prompt_ids() == [4]

    def test_lid_prompt_canary2_matches_reference_prefix(self) -> None:
        stage = self._stage_with_tokenizer()
        stage.model.tokenizer.prompt_format = "canary2"
        assert stage._lid_prompt_ids() == [7, 4, 18]

    def test_parse_language_from_full_sequence(self) -> None:
        stage = self._stage_with_tokenizer()
        assert stage._parse_language([4, 89, 3], [4]) == ("hi", 1.0)

    def test_parse_language_from_generated_only_sequence(self) -> None:
        stage = self._stage_with_tokenizer()
        assert stage._parse_language([185, 3], [4]) == ("ta", 1.0)

    def test_parse_language_unknown_returns_empty(self) -> None:
        stage = self._stage_with_tokenizer(candidate_langs=["hi"])
        assert stage._parse_language([4, 185, 3], [4]) == ("", 0.0)
