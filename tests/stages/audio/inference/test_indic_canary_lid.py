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

from types import SimpleNamespace

from nemo_curator.stages.audio.inference.indic_canary_lid import (
    IndicCanaryLangIDStage,
    _language_code_from_special_token,
)


def test_language_code_from_special_token() -> None:
    assert _language_code_from_special_token("<|hi|>") == "hi"
    assert _language_code_from_special_token("<|bn-IN|>") == "bn-in"
    assert _language_code_from_special_token("<|pnc|>") is None
    assert _language_code_from_special_token("hi") is None


def test_collect_language_tokens_respects_candidates() -> None:
    stage = IndicCanaryLangIDStage(candidate_langs=["hi"])
    stage.model = SimpleNamespace(
        tokenizer=SimpleNamespace(id_to_token={1: "<|hi|>", 2: "<|bn|>", 3: "<|pnc|>"})
    )

    assert stage._collect_language_token_ids() == {1: "hi"}


def test_parse_language_ignores_prompt_and_padding() -> None:
    stage = IndicCanaryLangIDStage()
    stage.model = SimpleNamespace(tokenizer=SimpleNamespace(eos_id=9, pad_id=0))
    stage._language_by_token_id = {4: "ta"}

    assert stage._parse_language([1, 2, 0, 4, 9], [1, 2]) == ("ta", 1.0)
    assert stage._parse_language([1, 2, 0, 9], [1, 2]) == ("", 0.0)
