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

"""CPU structural tests for the Indic Canary TRT-LLM stage (engine mocked)."""

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from nemo_curator.stages.audio.inference import indic_canary as indic_canary_mod
from nemo_curator.stages.audio.inference.indic_canary import (
    _MAX_SAMPLES,
    _MIN_DURATION_SAMPLES,
    _MIN_SAMPLES,
    _TARGET_SR,
    IndicCanaryTRTLLMASR,
    InferenceIndicCanaryStage,
)
from nemo_curator.tasks import AudioTask


def _make_task(lang: str, n: int = 8000) -> AudioTask:
    return AudioTask(
        data={
            "audio_filepath": f"/test/{lang}.wav",
            "source_lang": lang,
            "waveform": np.zeros(n, dtype=np.float32),
            "sampling_rate": 16000,
        }
    )


class TestStageContract:
    def test_inputs_outputs(self) -> None:
        stage = InferenceIndicCanaryStage(engine_dir="canary_engine")
        assert stage.inputs() == ([], ["waveform", "sampling_rate"])
        assert stage.outputs() == ([], ["asr_prediction", "asr_language"])

    def test_process_raises_not_implemented(self) -> None:
        stage = InferenceIndicCanaryStage(engine_dir="canary_engine")
        with pytest.raises(NotImplementedError):
            stage.process(_make_task("hi"))

    def test_process_batch_without_setup_raises(self) -> None:
        stage = InferenceIndicCanaryStage(engine_dir="canary_engine")
        with pytest.raises(RuntimeError, match="setup"):
            stage.process_batch([_make_task("hi")])

    def test_empty_batch_returns_empty(self) -> None:
        stage = InferenceIndicCanaryStage(engine_dir="canary_engine")
        assert stage.process_batch([]) == []


class TestSetupOnNode:
    def test_missing_engine_dir_arg_raises(self) -> None:
        stage = InferenceIndicCanaryStage(engine_dir="")
        with pytest.raises(ValueError, match="engine_dir"):
            stage.setup_on_node()

    def test_missing_files_raises(self, tmp_path: Path) -> None:
        stage = InferenceIndicCanaryStage(engine_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError, match="missing required file"):
            stage.setup_on_node()

    def test_all_files_present_passes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        for rel in [
            "encoder/encoder.plan",
            "decoder/config.json",
            "decoder/vocab.json",
            "preprocessor/config.json",
            "preprocessor/mel_basis.pt",
        ]:
            fp = tmp_path / rel
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text("x")
        # Stub the engine load so file validation is exercised without tensorrt_llm.
        monkeypatch.setattr(IndicCanaryTRTLLMASR, "setup", lambda _self: None)
        stage = InferenceIndicCanaryStage(engine_dir=str(tmp_path))
        stage.setup_on_node()  # should not raise
        assert stage._model is not None


def _mock_engine(supported: set[str]) -> MagicMock:
    """A stand-in IndicCanaryTRTLLMASR: gate by ``supported`` and echo langs."""
    model = MagicMock(spec=IndicCanaryTRTLLMASR)
    model.supports_language.side_effect = lambda lang: lang in supported

    def _generate(_waveforms, _sample_rates, lang_codes):  # noqa: ANN001, ANN202
        return [f"pred_{lang}" for lang in lang_codes], list(lang_codes)

    model.generate.side_effect = _generate
    return model


class TestProcessBatch:
    def test_transcribes_supported_and_skips_unsupported(self) -> None:
        stage = InferenceIndicCanaryStage(engine_dir="canary_engine")
        stage._model = _mock_engine(supported={"hi", "ta"})

        tasks = [_make_task("hi"), _make_task("zz"), _make_task("ta")]
        out = stage.process_batch(tasks)

        assert out[0].data["asr_prediction"] == "pred_hi"
        assert out[0].data["asr_language"] == "hi"
        assert out[2].data["asr_prediction"] == "pred_ta"
        # Unsupported language is annotated, not transcribed.
        assert out[1].data["asr_prediction"] == ""
        assert "lang_not_supported" in str(out[1].data.get("additional_notes", ""))

    def test_waveform_popped_by_default(self) -> None:
        stage = InferenceIndicCanaryStage(engine_dir="canary_engine")
        stage._model = _mock_engine(supported={"hi"})
        tasks = [_make_task("hi")]
        out = stage.process_batch(tasks)
        assert "waveform" not in out[0].data

    def test_waveform_kept_when_requested(self) -> None:
        stage = InferenceIndicCanaryStage(engine_dir="canary_engine", keep_waveform=True)
        stage._model = _mock_engine(supported={"hi"})
        out = stage.process_batch([_make_task("hi")])
        assert "waveform" in out[0].data

    def test_all_unsupported_returns_tasks_with_defaults(self) -> None:
        stage = InferenceIndicCanaryStage(engine_dir="canary_engine")
        stage._model = _mock_engine(supported=set())
        out = stage.process_batch([_make_task("zz")])
        assert out[0].data["asr_prediction"] == ""
        assert out[0].data["asr_language"] == ""


class TestEngineHelpers:
    def _engine_with_tokenizer(self, langs: list[str], spl: set[str]) -> IndicCanaryTRTLLMASR:
        eng = IndicCanaryTRTLLMASR(engine_dir="canary_engine")
        tok = SimpleNamespace(
            langs=langs,
            supports_prompt_language=lambda lang: lang in spl,
        )
        eng._model = SimpleNamespace(tokenizer=tok)
        return eng

    def test_normalize_lang_direct(self) -> None:
        eng = self._engine_with_tokenizer(["hi", "en"], {"hi", "en"})
        assert eng._normalize_lang("hi") == "hi"

    def test_normalize_lang_strips_country_code(self) -> None:
        eng = self._engine_with_tokenizer(["hi", "en"], {"hi", "en"})
        assert eng._normalize_lang("hi-IN") == "hi"

    def test_normalize_lang_unsupported(self) -> None:
        eng = self._engine_with_tokenizer(["hi"], {"hi"})
        assert eng._normalize_lang("zz") is None

    def test_supports_language(self) -> None:
        eng = self._engine_with_tokenizer(["hi"], {"hi"})
        assert eng.supports_language("hi") is True
        assert eng.supports_language("zz") is False

    def test_prompt_cfg_shape(self) -> None:
        eng = IndicCanaryTRTLLMASR(engine_dir="canary_engine", pnc=True)
        cfg = eng._prompt_cfg("hi")
        assert cfg["source_language"] == "hi"
        assert cfg["target_language"] == "hi"
        assert cfg["task"] == "transcribe"
        assert cfg["pnc"] is True

    def test_supports_language_without_setup_raises(self) -> None:
        eng = IndicCanaryTRTLLMASR(engine_dir="canary_engine")
        with pytest.raises(RuntimeError, match="setup"):
            eng.supports_language("hi")


class _RecordingCanaryModel:
    """Stand-in for the vendored CanaryTRTLLM engine.

    Records the (waveform, duration) args the engine wrapper passes to
    ``process_batch`` so tests can assert on the CPU-side audio preprocessing.
    """

    def __init__(self, max_batch_size: int = 8, langs: tuple[str, ...] = ("hi",)) -> None:
        self.max_batch_size = max_batch_size
        self.tokenizer = SimpleNamespace(
            langs=list(langs),
            supports_prompt_language=lambda lang: lang in set(langs),
        )
        self.calls: list[dict] = []

    def process_batch(
        self,
        padded: list,
        durations: list,
        prompts_cfg: list,
        num_beams: int = 1,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        _ = (num_beams, max_new_tokens)
        self.calls.append({"padded": padded, "durations": durations, "prompts_cfg": prompts_cfg})
        return ["text"] * len(padded)


@pytest.fixture
def _stub_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a lightweight ``pad_or_trim`` so ``generate`` runs without tensorrt_llm.

    The real runtime module imports tensorrt / tensorrt_llm at import time, which
    is unavailable on CPU CI; this stub mirrors ``pad_or_trim`` for 1-D tensors.
    """
    mod = types.ModuleType("nemo_curator.stages.audio.inference.indic_canary_trtllm_runtime")

    def pad_or_trim(array: torch.Tensor, length: int, *, axis: int = -1) -> torch.Tensor:
        n = array.shape[axis]
        if n > length:
            return array.narrow(axis, 0, length)
        if n < length:
            return torch.nn.functional.pad(array, (0, length - n))
        return array

    mod.pad_or_trim = pad_or_trim
    monkeypatch.setitem(sys.modules, mod.__name__, mod)


def _engine_for_generate(max_batch_size: int = 8) -> IndicCanaryTRTLLMASR:
    eng = IndicCanaryTRTLLMASR(engine_dir="canary_engine")
    eng._model = _RecordingCanaryModel(max_batch_size=max_batch_size)
    return eng


@pytest.mark.usefixtures("_stub_runtime")
class TestGenerateWaveformPrep:
    """Regression tests for waveform preprocessing inside ``generate``."""

    def test_channels_first_mono_not_collapsed(self) -> None:
        """A (1, N) channels-first clip must keep its N samples, not collapse to 1.

        Regression: ``mean(dim=-1)`` averaged over the samples axis, reducing the
        whole clip to a single value and yielding near-silent, truncated output.
        """
        eng = _engine_for_generate()
        n = _TARGET_SR  # 1 s @ 16 kHz
        wav = np.full((1, n), 0.1, dtype=np.float32)  # (channels, samples)

        texts, _ = eng.generate([wav], [_TARGET_SR], ["hi"])

        call = eng._model.calls[0]
        assert call["durations"][0] == n  # real duration preserved, not 1
        assert int(call["padded"][0].shape[0]) == _MIN_SAMPLES  # padded to the 3 s floor
        assert len(texts) == 1

    def test_stereo_downmixed_to_mono_length(self) -> None:
        eng = _engine_for_generate()
        n = _TARGET_SR
        wav = np.zeros((2, n), dtype=np.float32)  # (channels, samples), stereo

        eng.generate([wav], [_TARGET_SR], ["hi"])

        call = eng._model.calls[0]
        assert call["durations"][0] == n  # channel downmix keeps sample count

    def test_1d_waveform_passthrough(self) -> None:
        eng = _engine_for_generate()
        n = 2 * _TARGET_SR
        wav = np.zeros(n, dtype=np.float32)  # 1-D samples

        eng.generate([wav], [_TARGET_SR], ["hi"])

        assert eng._model.calls[0]["durations"][0] == n

    def test_tiny_clip_duration_clamped(self) -> None:
        """Near-empty clips get a min-duration floor so per-feature std is defined.

        Regression: a clip producing a single mel frame made torch.std() return
        NaN and raised, killing the whole Ray batch.
        """
        eng = _engine_for_generate()
        wav = np.zeros(100, dtype=np.float32)  # < _MIN_DURATION_SAMPLES

        eng.generate([wav], [_TARGET_SR], ["hi"])

        assert eng._model.calls[0]["durations"][0] == _MIN_DURATION_SAMPLES

    def test_long_clip_trimmed_to_window(self) -> None:
        eng = _engine_for_generate()
        wav = np.zeros(40 * _TARGET_SR, dtype=np.float32)  # 40 s > 30 s window

        eng.generate([wav], [_TARGET_SR], ["hi"])

        call = eng._model.calls[0]
        assert call["durations"][0] == _MAX_SAMPLES
        assert int(call["padded"][0].shape[0]) == _MAX_SAMPLES

    def test_resample_changes_sample_count(self) -> None:
        eng = _engine_for_generate()
        # 8 kHz, 1 s -> resampled to 16 kHz should roughly double the samples.
        wav = np.zeros(8000, dtype=np.float32)

        eng.generate([wav], [8000], ["hi"])

        assert eng._model.calls[0]["durations"][0] == pytest.approx(_TARGET_SR, rel=0.02)

    def test_batches_chunked_to_engine_max(self) -> None:
        """More clips than the engine's max batch size are split across calls."""
        eng = _engine_for_generate(max_batch_size=2)
        wavs = [np.zeros(_TARGET_SR, dtype=np.float32) for _ in range(5)]

        texts, _ = eng.generate(wavs, [_TARGET_SR] * 5, ["hi"] * 5)

        assert len(texts) == 5
        # 5 clips, max batch 2 -> chunks of 2, 2, 1.
        assert [len(c["padded"]) for c in eng._model.calls] == [2, 2, 1]

    def test_constants_are_sane(self) -> None:
        assert 0 < _MIN_DURATION_SAMPLES < _MIN_SAMPLES <= _MAX_SAMPLES
        assert indic_canary_mod._TARGET_SR == 16000
