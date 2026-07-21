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

from typing import ClassVar
from unittest.mock import MagicMock

import numpy as np
import torch

from nemo_curator.stages.audio.inference.indic_conformer_hybrid import (
    _TARGET_SR,
    IndicConformerHybridASR,
    InferenceIndicConformerHybridStage,
)
from nemo_curator.tasks import AudioTask


class _Tokenizer:
    token_id_offset: ClassVar[dict[str, int]] = {"hi": 0, "ta": 0}

    def ids_to_text(self, ids: list[int]) -> str:
        table = {0: "a", 1: "b", 2: "c"}
        return "".join(table[i] for i in ids)


class _CtcDecoder:
    def __init__(self) -> None:
        self.language_ids: list[list[str]] = []

    def __call__(self, encoder_output: torch.Tensor, language_ids: list[str]) -> torch.Tensor:
        self.language_ids.append(language_ids)
        batch, _, time = encoder_output.shape
        log_probs = torch.full((batch, time, 4), -100.0)
        patterns = ([0, 1, 3], [2, 3, 3])
        for idx in range(batch):
            pattern = patterns[idx % len(patterns)]
            for time_idx in range(time):
                token = pattern[min(time_idx, len(pattern) - 1)]
                log_probs[idx, time_idx, token] = 0.0
        return log_probs


class _CtcModel:
    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()
        self.ctc_decoder = _CtcDecoder()
        self.calls: list[dict] = []

    def __call__(self, *, input_signal: torch.Tensor, input_signal_length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append({"shape": tuple(input_signal.shape), "lengths": input_signal_length.tolist()})
        encoded = torch.zeros((input_signal.shape[0], 1, input_signal.shape[1]))
        return encoded, input_signal_length


class _RnntDecoder:
    blank_idx = 3

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def initialize_state(self, y: torch.Tensor) -> list[torch.Tensor]:
        return [torch.zeros((1, y.shape[0], 1))]

    def predict(
        self,
        y: torch.Tensor | None = None,
        state: list[torch.Tensor] | None = None,
        *,
        add_sos: bool = False,
        batch_size: int | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        _ = (state, add_sos)
        batch = int(batch_size if y is None else y.shape[0])
        self.batch_sizes.append(batch)
        return torch.zeros((batch, 1, 1)), [torch.zeros((1, batch, 1))]

    @classmethod
    def batch_replace_states_mask(
        cls,
        src_states: list[torch.Tensor],
        dst_states: list[torch.Tensor],
        mask: torch.Tensor,
    ) -> None:
        torch.where(mask.view(1, -1, 1), src_states[0], dst_states[0], out=dst_states[0])


class _RnntJoint(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(1, 1)
        self.counts: dict[int, int] = {}
        self.language_ids: list[list[str]] = []

    def enc(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def pred(self, g: torch.Tensor) -> torch.Tensor:
        return g

    def joint_after_projection(
        self,
        f: torch.Tensor,
        g: torch.Tensor,
        *,
        language_ids: list[str],
    ) -> torch.Tensor:
        _ = g
        self.language_ids.append(list(language_ids))
        batch = f.shape[0]
        log_probs = torch.full((batch, 1, 1, 3), -100.0)
        for idx in range(batch):
            sample_key = int(f[idx, 0, 0].item())
            count = self.counts.get(sample_key, 0)
            self.counts[sample_key] = count + 1
            token = sample_key if count == 0 else 2
            log_probs[idx, 0, 0, token] = 0.0
        return log_probs


class _RnntModel:
    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()
        self.decoder = _RnntDecoder()
        self.joint = _RnntJoint()
        self.calls: list[dict] = []

    def __call__(self, *, input_signal: torch.Tensor, input_signal_length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append({"shape": tuple(input_signal.shape), "lengths": input_signal_length.tolist()})
        batch, time = input_signal.shape
        row_ids = torch.arange(batch, dtype=torch.float32).view(batch, 1, 1)
        encoded = row_ids.expand(batch, 1, time).contiguous()
        return encoded, input_signal_length


def _asr(model: object, *, decode_mode: str, batch_size: int) -> IndicConformerHybridASR:
    asr = IndicConformerHybridASR("dummy.nemo", decode_mode=decode_mode, inference_batch_size=batch_size)
    asr._model = model
    asr._device = torch.device("cpu")
    asr._per_lang_classes = 3 if decode_mode == "ctc" else 2
    asr._chunk_duration_sec = 30.0
    return asr


def test_ctc_generate_batches_encoder_calls_and_preserves_order() -> None:
    model = _CtcModel()
    asr = _asr(model, decode_mode="ctc", batch_size=2)
    waveforms = [
        np.zeros(10, dtype=np.float32),
        np.zeros(15, dtype=np.float32),
        np.zeros(0, dtype=np.float32),
        np.zeros(7, dtype=np.float32),
    ]

    texts, langs = asr.generate(waveforms, [_TARGET_SR] * 4, ["hi"] * 4)

    assert [call["shape"][0] for call in model.calls] == [2, 1]
    assert [call["lengths"] for call in model.calls] == [[7, 10], [15]]
    assert texts == ["c", "ab", "", "ab"]
    assert langs == ["hi", "hi", "hi", "hi"]


def test_rnnt_generate_decodes_active_rows_as_batches() -> None:
    model = _RnntModel()
    asr = _asr(model, decode_mode="rnnt", batch_size=2)
    waveforms = [np.zeros(8, dtype=np.float32), np.zeros(8, dtype=np.float32)]

    texts, _ = asr.generate(waveforms, [_TARGET_SR, _TARGET_SR], ["hi", "ta"])

    assert [call["shape"][0] for call in model.calls] == [2]
    assert set(model.decoder.batch_sizes) == {2}
    assert all(language_ids == ["hi", "ta"] for language_ids in model.joint.language_ids)
    assert texts == ["a", "b"]


def test_stage_passes_inference_batch_size_to_model_wrapper() -> None:
    default_stage = InferenceIndicConformerHybridStage(model_id="dummy.nemo", batch_size=8)
    stage = InferenceIndicConformerHybridStage(
        model_id="dummy.nemo",
        batch_size=8,
        inference_batch_size=4,
    )

    default_model = default_stage._create_model()
    model = stage._create_model()

    assert default_model.inference_batch_size == 8
    assert model.inference_batch_size == 4


def test_stage_passes_rnnt_precision_to_model_wrapper() -> None:
    stage = InferenceIndicConformerHybridStage(model_id="dummy.nemo", rnnt_precision="fp16")

    model = stage._create_model()

    assert model.rnnt_precision == "fp16"


def test_stage_process_batch_calls_generate_once_for_eligible_batch() -> None:
    stage = InferenceIndicConformerHybridStage(model_id="dummy.nemo", keep_waveform=True)
    stage._model = MagicMock()
    stage._model.generate.return_value = (["one", "two"], ["hi", "hi"])
    tasks = [
        AudioTask(data={"waveform": np.zeros(8, dtype=np.float32), "sampling_rate": _TARGET_SR, "source_lang": "hi"}),
        AudioTask(data={"waveform": np.zeros(9, dtype=np.float32), "sampling_rate": _TARGET_SR, "source_lang": "hi"}),
    ]

    out = stage.process_batch(tasks)

    stage._model.generate.assert_called_once()
    assert len(stage._model.generate.call_args.args[0]) == 2
    assert [task.data["asr_prediction"] for task in out] == ["one", "two"]
