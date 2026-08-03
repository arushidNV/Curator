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

"""AI4Bharat / NVIDIA (Indic) Canary ASR via a prebuilt TensorRT-LLM engine.

This module holds both the inference engine and its Curator pipeline stage:

- :class:`IndicCanaryTRTLLMASR` — wraps the vendored TensorRT-LLM Canary runtime
  (:mod:`nemo_curator.stages.audio.inference.indic_canary_trtllm_runtime`) and
  runs static-batch inference (waveforms in → text out) against a **prebuilt**
  engine directory.
- :class:`InferenceIndicCanaryStage` — the ``ProcessingStage`` that adapts it to
  the audio pipeline (per-sample language routing, Ray scaling, task I/O).

``tensorrt`` / ``tensorrt_llm`` are heavy, GPU-only dependencies that are NOT part
of Curator's default install. They are imported lazily inside :meth:`setup`, so
importing this module (and the containing package) never requires them. They also
cannot be co-resolved into ``audio_cuda12`` / ``uv.lock`` (tensorrt_llm pins
transformers<4.52 / torch~2.7, incompatible with nemo_toolkit / torch 2.9.1+cu128),
so install them as a separate step on top of the synced audio venv::

    uv pip install -r requirements-trt-llm.txt   # or: pip install -r ...

The ``engine_dir`` must point at an already-built engine (see the README shipped
with the TensorRT-LLM ``canary-indic`` example): ``convert_checkpoint.py`` →
``conformer_onnx_trt.py`` (encoder ``.plan``) → ``trtllm-build`` (decoder engine),
plus the exported ``preprocessor/`` artifacts. This stage does not build engines.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from nemo_curator.models.base import ModelInterface
from nemo_curator.stages.audio.pipeline_utils import set_note
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

_TARGET_SR = 16000
# Encoder engines are built for a 40-second window; clip anything longer. Both
# bounds are configurable (see IndicCanaryTRTLLMASR / InferenceIndicCanaryStage);
# these are just the defaults.
_DEFAULT_MAX_DURATION_SEC = 40.0
_DEFAULT_MIN_DURATION_SEC = 0.5
# `per_feature` normalization computes std over the valid frames; a single mel
# frame makes torch.std() return NaN and raises inside the preprocessor. Floor the
# reported valid duration so degenerate/near-empty clips (< ~10 ms) still yield
# several frames. The buffer is already zero-padded to min_samples, so this only
# spans padding and produces an empty transcription instead of crashing the batch.
_MIN_DURATION_SAMPLES = 400  # 25 ms @ 16 kHz -> ~3 mel frames


class IndicCanaryTRTLLMASR(ModelInterface):
    """(Indic) Canary ASR engine backed by a prebuilt TensorRT-LLM engine dir.

    Pure inference: ``setup()`` then ``generate(waveforms, sample_rates, lang_codes)``.
    Knows nothing about the Curator pipeline — :class:`InferenceIndicCanaryStage`
    (below) adapts it to ``AudioTask`` / Ray.
    """

    def __init__(  # noqa: PLR0913
        self,
        engine_dir: str,
        *,
        num_beams: int = 4,
        max_new_tokens: int = 374,
        pnc: bool = False,
        max_duration_sec: float = _DEFAULT_MAX_DURATION_SEC,
        min_duration_sec: float = _DEFAULT_MIN_DURATION_SEC,
        kv_cache_free_gpu_memory_fraction: float = 0.2,
        cross_kv_cache_fraction: float = 0.2,
    ):
        self.engine_dir = engine_dir
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.pnc = pnc
        self.kv_cache_free_gpu_memory_fraction = kv_cache_free_gpu_memory_fraction
        self.cross_kv_cache_fraction = cross_kv_cache_fraction
        # Window bounds in samples. max clips overly long clips to the encoder's
        # build-time window; min sets the floor the batch is zero-padded up to.
        # min is capped at max so a misconfigured min_duration_sec can never pad
        # beyond the encoder window.
        self.max_samples = int(max_duration_sec * _TARGET_SR)
        self.min_samples = min(int(min_duration_sec * _TARGET_SR), self.max_samples)
        self._model: Any = None

    @property
    def model_id_names(self) -> list[str]:
        return [self.engine_dir]

    def setup(self) -> None:
        # Lazy: tensorrt / tensorrt_llm are only needed here, on a GPU worker.
        from nemo_curator.stages.audio.inference.indic_canary_trtllm_runtime import CanaryTRTLLM

        logger.info(f"Loading Indic Canary TRT-LLM engine from {self.engine_dir}")
        self._model = CanaryTRTLLM(
            self.engine_dir,
            device="cuda:0",
            kv_cache_free_gpu_memory_fraction=self.kv_cache_free_gpu_memory_fraction,
            cross_kv_cache_fraction=self.cross_kv_cache_fraction,
        )
        logger.info(
            f"Indic Canary ready: prompt_format={self._model.tokenizer.prompt_format}, "
            f"max_batch_size={self._model.max_batch_size}, langs={len(self._model.tokenizer.langs)}"
        )

    def teardown(self) -> None:
        del self._model
        self._model = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001, S110
            pass

    # ------------------------------------------------------------------
    # Language support (dynamic — derived from the loaded tokenizer)
    # ------------------------------------------------------------------
    def _normalize_lang(self, lang: str) -> str | None:
        """Return the tokenizer-accepted form of ``lang`` (or None if unsupported)."""
        tok = self._model.tokenizer
        candidates = [lang]
        if "-" in lang:
            candidates.append(lang.split("-")[0])
        for cand in candidates:
            if tok.supports_prompt_language(cand) or cand in tok.langs:
                return cand
        return None

    def supports_language(self, lang: str) -> bool:
        if self._model is None:
            msg = "Model not initialized. Call setup() first."
            raise RuntimeError(msg)
        return self._normalize_lang(lang) is not None

    def _prompt_cfg(self, lang: str) -> dict:
        return {
            "task": "transcribe",
            "pnc": self.pnc,
            "source_language": lang,
            "target_language": lang,
            "itn": False,
            "romanized": False,
            "timestamp": False,
            "diarize": False,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def generate(
        self,
        waveforms: list[np.ndarray],
        sample_rates: list[int],
        lang_codes: list[str],
    ) -> tuple[list[str], list[str]]:
        if self._model is None:
            msg = "Model not initialized. Call setup() first."
            raise RuntimeError(msg)

        import torch
        import torchaudio.functional as AF  # noqa: N812

        from nemo_curator.stages.audio.inference.indic_canary_trtllm_runtime import pad_or_trim

        # Resample to 16 kHz mono float32 and clip to the engine window.
        prepared: list[Any] = []
        lengths: list[int] = []
        langs_norm: list[str] = []
        for w, sr, lang in zip(waveforms, sample_rates, lang_codes, strict=True):
            wav = torch.as_tensor(np.ascontiguousarray(w, dtype=np.float32))
            if wav.ndim > 1:
                # Curator's readers (see stages.audio.common.load_audio_file) yield
                # channels-first (channels, samples); downmix across channels (dim 0).
                # Averaging over dim=-1 would collapse the whole clip to one sample.
                wav = wav.mean(dim=0)
            wav = wav.reshape(-1)
            if int(sr) != _TARGET_SR:
                wav = AF.resample(wav, orig_freq=int(sr), new_freq=_TARGET_SR)
            if wav.shape[0] > self.max_samples:
                logger.warning(
                    f"Audio clip is {wav.shape[0] / _TARGET_SR:.2f}s, longer than the "
                    f"{self.max_samples / _TARGET_SR:.2f}s encoder window; truncating the remainder. "
                    "Split long input with VAD before Indic Canary inference."
                )
            wav = wav[: self.max_samples]
            prepared.append(wav)
            lengths.append(int(wav.shape[0]))
            langs_norm.append(self._normalize_lang(lang) or lang)

        texts: list[str] = [""] * len(prepared)
        # Chunk to the engine's max batch size (single job would otherwise assert).
        max_bs = max(1, int(self._model.max_batch_size))
        for start in range(0, len(prepared), max_bs):
            end = start + max_bs
            chunk = prepared[start:end]
            chunk_lengths = lengths[start:end]
            # Clips were already clipped to <= max_samples above, and min_samples is
            # capped at max_samples, so pad only up to the longest clip in the chunk
            # (floored at min_samples) — no need to re-clamp to max_samples here.
            pad_len = max(*chunk_lengths, self.min_samples)
            padded = [pad_or_trim(w, pad_len) for w in chunk]
            durations = [min(max(length, _MIN_DURATION_SAMPLES), pad_len) for length in chunk_lengths]
            prompts_cfg = [self._prompt_cfg(langs_norm[i]) for i in range(start, end) if i < len(langs_norm)]
            preds = self._model.process_batch(
                padded,
                durations,
                prompts_cfg,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
            )
            for offset, pred in enumerate(preds):
                texts[start + offset] = pred

        return texts, langs_norm


@dataclass
class InferenceIndicCanaryStage(ProcessingStage[AudioTask, AudioTask]):
    """Audio transcription with a prebuilt (Indic) Canary TensorRT-LLM engine.

    Pipeline adapter over :class:`IndicCanaryTRTLLMASR` (same module): reads
    in-memory waveforms from each ``AudioTask``, routes per-sample by
    ``source_lang`` (gated dynamically against the engine's tokenizer languages),
    and writes the predicted transcription.

    Args:
        engine_dir: Path to a prebuilt TensorRT-LLM Canary engine directory
            (``encoder/encoder.plan``, ``decoder/`` engine, ``preprocessor/``).
        num_beams: Decoder beam width (build engine with matching ``max_beam_width``).
        max_new_tokens: Max generated tokens (clamped to the engine's seq budget).
        pnc: Request punctuation & capitalization in the Canary control prompt.
        max_duration_sec: Upper bound of the audio window in seconds; longer clips
            are clipped. Should not exceed the encoder engine's build-time window.
        min_duration_sec: Lower bound in seconds; shorter clips are zero-padded up
            to this length before inference.
        source_lang_key: Task key holding the per-sample ISO language code.
        skip_me_key: Task key for the shared downstream "skip this entry" flag; set
            to ``lang_not_supported:<stage>`` for unsupported-language samples.
        keep_waveform: When True the waveform is left on the task for a later stage.
    """

    name: str = "IndicCanary_inference"
    engine_dir: str = ""
    num_beams: int = 4
    # Matches the default 40 s encoder window (build tooling: 30 s -> 246, 40 s -> 374);
    # clamped to the engine's own budget at inference time.
    max_new_tokens: int = 374
    pnc: bool = False
    max_duration_sec: float = _DEFAULT_MAX_DURATION_SEC
    min_duration_sec: float = _DEFAULT_MIN_DURATION_SEC
    kv_cache_free_gpu_memory_fraction: float = 0.2
    cross_kv_cache_fraction: float = 0.2
    source_lang_key: str = "source_lang"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sampling_rate"
    pred_text_key: str = "asr_prediction"
    language_key: str = "asr_language"
    notes_key: str = "additional_notes"
    skip_me_key: str = "_skipme"
    keep_waveform: bool = False
    num_workers_override: int | None = None
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    batch_size: int = 64
    _model: IndicCanaryTRTLLMASR | None = field(default=None, init=False, repr=False)

    def num_workers(self) -> int | None:
        return self.num_workers_override

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if self.num_workers_override is not None:
            spec["num_workers"] = self.num_workers_override
        return spec

    def _create_model(self) -> IndicCanaryTRTLLMASR:
        return IndicCanaryTRTLLMASR(
            engine_dir=self.engine_dir,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            pnc=self.pnc,
            max_duration_sec=self.max_duration_sec,
            min_duration_sec=self.min_duration_sec,
            kv_cache_free_gpu_memory_fraction=self.kv_cache_free_gpu_memory_fraction,
            cross_kv_cache_fraction=self.cross_kv_cache_fraction,
        )

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        # The engine is a local build artifact — validate it exists rather than download.
        if not self.engine_dir:
            msg = "InferenceIndicCanaryStage requires 'engine_dir' to point at a prebuilt TRT-LLM engine"
            raise ValueError(msg)
        required = [
            os.path.join(self.engine_dir, "encoder", "encoder.plan"),
            os.path.join(self.engine_dir, "decoder", "config.json"),
            os.path.join(self.engine_dir, "decoder", "vocab.json"),
            os.path.join(self.engine_dir, "preprocessor", "config.json"),
            os.path.join(self.engine_dir, "preprocessor", "mel_basis.pt"),
        ]
        missing = [p for p in required if not os.path.exists(p)]
        if missing:
            msg = f"engine_dir '{self.engine_dir}' is missing required file(s): {missing}"
            raise FileNotFoundError(msg)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._model is None:
            self._model = self._create_model()
            self._model.setup()
            logger.info(f"Indic Canary model ready: {self.engine_dir}")

    def teardown(self) -> None:
        if self._model is not None:
            self._model.teardown()
            self._model = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.waveform_key, self.sample_rate_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.pred_text_key, self.language_key]

    def process(self, task: AudioTask) -> AudioTask:
        msg = "InferenceIndicCanaryStage only supports process_batch"
        raise NotImplementedError(msg)

    def _eligible_indices(self, tasks: list[AudioTask]) -> list[int]:
        """Indices of tasks whose source language the engine supports; flag the rest."""
        eligible: list[int] = []
        for i, task in enumerate(tasks):
            lang = str(task.data.get(self.source_lang_key, "") or "").strip().lower()
            if self._model.supports_language(lang):
                eligible.append(i)
            else:
                # Leave primary/fallback predictions empty and flag via _skipme.
                task.data[self.pred_text_key] = ""
                if not task.data.get(self.skip_me_key, ""):
                    task.data[self.skip_me_key] = f"lang_not_supported:{self.name}"
                set_note(task.data, self.name, f"skipped (unsupported language: {lang})", self.notes_key)
        return eligible

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:  # noqa: C901
        if len(tasks) == 0:
            return []
        if self._model is None:
            msg = "Model not initialized — setup() was not called"
            raise RuntimeError(msg)

        for task in tasks:
            task.data.setdefault(self.pred_text_key, "")
            task.data.setdefault(self.language_key, "")

        eligible_indices = self._eligible_indices(tasks)
        lang_skipped = len(tasks) - len(eligible_indices)
        if not eligible_indices:
            if not self.keep_waveform:
                for task in tasks:
                    task.data.pop(self.waveform_key, None)
            logger.info(f"{self.name}: skipped entire batch of {len(tasks)} (no supported languages)")
            return tasks

        eligible_tasks = [tasks[i] for i in eligible_indices]
        waveforms = [t.data[self.waveform_key] for t in eligible_tasks]
        sample_rates = [t.data[self.sample_rate_key] for t in eligible_tasks]
        lang_codes = [str(t.data.get(self.source_lang_key, "") or "").strip().lower() for t in eligible_tasks]

        # Record encoder-window truncation in the manifest (additional_notes), not
        # just the worker log: generate() clips clips longer than the build-time
        # window, which drops the tail of the transcription.
        for t, w, sr in zip(eligible_tasks, waveforms, sample_rates, strict=True):
            n_samples = w.shape[-1] if getattr(w, "ndim", 1) > 1 else len(w)
            duration = n_samples / sr if sr else 0.0
            if duration >= self.max_duration_sec:
                set_note(
                    t.data,
                    self.name,
                    f"audio {duration:.2f}s exceeds {self.max_duration_sec:.0f}s encoder window; "
                    "transcription truncated (split with VAD before inference)",
                    self.notes_key,
                )

        pred_texts, langs_out = self._model.generate(waveforms, sample_rates, lang_codes)

        for task_idx, pred, lang in zip(eligible_indices, pred_texts, langs_out, strict=True):
            tasks[task_idx].data[self.pred_text_key] = pred
            tasks[task_idx].data[self.language_key] = lang

        if not self.keep_waveform:
            for task in tasks:
                task.data.pop(self.waveform_key, None)

        logger.info(
            f"{self.name}: generated {len(eligible_indices)} predictions, "
            f"skipped {lang_skipped} (unsupported language)"
        )
        return tasks
