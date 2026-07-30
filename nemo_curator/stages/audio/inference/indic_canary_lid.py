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

"""Indic Canary language identification using a prebuilt TensorRT-LLM engine."""

from __future__ import annotations

import gc
import os
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

from nemo_curator.stages.audio.inference.indic_canary import (
    _DEFAULT_MAX_DURATION_SEC,
    _MIN_DURATION_SAMPLES,
    _TARGET_SR,
)
from nemo_curator.stages.audio.inference.langid_base import BaseLangIDStage
from nemo_curator.stages.resources import Resources

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata
    from nemo_curator.tasks import AudioTask

_NON_LANGUAGE_SPECIAL_TOKENS = frozenset({"pnc", "itn"})
_LANGUAGE_CODE_MAX_PARTS = 2
_LANGUAGE_CODE_LENGTHS = frozenset({2, 3})
_LANGUAGE_COUNTRY_CODE_LENGTH = 2


def _validate_engine_dir(engine_dir: str, stage_name: str) -> None:
    if not engine_dir:
        msg = f"{stage_name} requires 'engine_dir' to point at a prebuilt TRT-LLM engine"
        raise ValueError(msg)
    required = [
        os.path.join(engine_dir, "encoder", "encoder.plan"),
        os.path.join(engine_dir, "decoder", "config.json"),
        os.path.join(engine_dir, "decoder", "vocab.json"),
        os.path.join(engine_dir, "preprocessor", "config.json"),
        os.path.join(engine_dir, "preprocessor", "mel_basis.pt"),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        msg = f"engine_dir '{engine_dir}' is missing required file(s): {missing}"
        raise FileNotFoundError(msg)


def _language_code_from_special_token(token: str) -> str | None:
    if not token.startswith("<|") or not token.endswith("|>"):
        return None
    code = token[2:-2]
    if code in _NON_LANGUAGE_SPECIAL_TOKENS:
        return None
    parts = code.split("-")
    if len(parts) > _LANGUAGE_CODE_MAX_PARTS or not all(part.isalpha() for part in parts):
        return None
    if len(parts[0]) not in _LANGUAGE_CODE_LENGTHS:
        return None
    if len(parts) == _LANGUAGE_CODE_MAX_PARTS and len(parts[1]) != _LANGUAGE_COUNTRY_CODE_LENGTH:
        return None
    return code.lower()


def _normalize_candidate_langs(candidate_langs: list[str] | None) -> frozenset[str] | None:
    if candidate_langs is None:
        return None
    return frozenset(str(lang).strip().lower() for lang in candidate_langs if str(lang).strip())


@dataclass
class IndicCanaryLangIDStage(BaseLangIDStage):
    """Language identification with the Indic Canary TRT-LLM decoder.

    The stage mirrors SpeechBrain/AmberNet LangID I/O: it reads waveform audio
    from each task and writes ``language`` and ``language_confidence``. The
    current engine does not expose generation logits, so confidence is ``1.0``
    when a language token is emitted and ``0.0`` otherwise.
    """

    name: str = "IndicCanaryLangID"
    engine_dir: str = ""
    num_beams: int = 1
    max_new_tokens: int = 1
    prompt_text: str | None = None
    max_duration_sec: float = _DEFAULT_MAX_DURATION_SEC
    candidate_langs: list[str] | None = None
    batch_size: int = 32
    kv_cache_free_gpu_memory_fraction: float = 0.2
    cross_kv_cache_fraction: float = 0.2
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))

    model: Any = field(default=None, init=False, repr=False)
    _language_by_token_id: dict[int, str] = field(default_factory=dict, init=False, repr=False)

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        _validate_engine_dir(self.engine_dir, self.__class__.__name__)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self.model is not None:
            return
        _validate_engine_dir(self.engine_dir, self.__class__.__name__)

        from nemo_curator.stages.audio.inference.indic_canary_trtllm_runtime import CanaryTRTLLM

        logger.info(f"IndicCanaryLangID: loading TRT-LLM engine from {self.engine_dir}")
        self.model = CanaryTRTLLM(
            self.engine_dir,
            device="cuda:0",
            kv_cache_free_gpu_memory_fraction=self.kv_cache_free_gpu_memory_fraction,
            cross_kv_cache_fraction=self.cross_kv_cache_fraction,
        )
        self._language_by_token_id = self._collect_language_token_ids()
        if not self._language_by_token_id:
            msg = "Indic Canary tokenizer has no language special tokens matching candidate_langs"
            raise RuntimeError(msg)
        logger.info(
            f"IndicCanaryLangID: model ready "
            f"(prompt_format={self.model.tokenizer.prompt_format}, candidates={len(self._language_by_token_id)})"
        )

    def teardown(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        self._language_by_token_id = {}
        gc.collect()
        with suppress(Exception):
            torch.cuda.empty_cache()

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        if self.model is None:
            msg = "Model not initialised - setup() was not called"
            raise RuntimeError(msg)

        valid_indices: list[int] = []
        audio_signals: list[torch.Tensor] = []
        audio_lengths: list[int] = []

        for i, task in enumerate(tasks):
            audio = self._prepare_audio(task)
            if audio is None:
                continue
            audio = audio.reshape(-1)
            valid_indices.append(i)
            audio_signals.append(audio[: self.max_samples])
            audio_lengths.append(min(len(audio), self.max_samples))

        if not audio_signals:
            return tasks

        languages, confidences = self._identify_batch(audio_signals, audio_lengths)

        for task_idx, lang, confidence in zip(valid_indices, languages, confidences, strict=True):
            task = tasks[task_idx]
            task.data[self.output_key] = lang
            task.data[self.confidence_key] = confidence

        return tasks

    @property
    def max_samples(self) -> int:
        return int(self.max_duration_sec * _TARGET_SR)

    def _collect_language_token_ids(self) -> dict[int, str]:
        candidates = _normalize_candidate_langs(self.candidate_langs)
        token_map = getattr(self.model.tokenizer, "id_to_token", {})
        languages: dict[int, str] = {}
        for token_id, token in token_map.items():
            code = _language_code_from_special_token(str(token))
            if code is None:
                continue
            if candidates is not None and code not in candidates:
                continue
            languages[int(token_id)] = code
        return languages

    def _lid_prompt_ids(self) -> list[int]:
        if self.prompt_text:
            return self.model.tokenizer.encode(self.prompt_text)
        if self.model.tokenizer.prompt_format == "canary2":
            return self.model.tokenizer.encode("<|startofcontext|> <|startoftranscript|> <|emo:undefined|>")
        return self.model.tokenizer.encode("<|startoftranscript|>")

    @staticmethod
    def _generated_part(sequence: list[int], prompt_ids: list[int]) -> list[int]:
        if sequence[: len(prompt_ids)] == prompt_ids:
            return sequence[len(prompt_ids) :]
        return sequence

    def _parse_language(self, sequence: list[int], prompt_ids: list[int]) -> tuple[str, float]:
        eos_id = getattr(self.model.tokenizer, "eos_id", None)
        pad_id = getattr(self.model.tokenizer, "pad_id", None)
        for token_id in self._generated_part([int(t) for t in sequence], prompt_ids):
            if token_id == eos_id:
                break
            if token_id == pad_id:
                continue
            lang = self._language_by_token_id.get(token_id)
            if lang is not None:
                return lang, 1.0
        return "", 0.0

    def _identify_batch(
        self,
        audio_signals: list[torch.Tensor],
        audio_lengths: list[int],
    ) -> tuple[list[str], list[float]]:
        from nemo_curator.stages.audio.inference.indic_canary_trtllm_runtime import pad_or_trim

        prompt_ids = self._lid_prompt_ids()
        if len(prompt_ids) > int(self.model.decoder.max_input_len):
            msg = (
                f"Indic Canary LangID prompt length {len(prompt_ids)} exceeds decoder max_input_len "
                f"{self.model.decoder.max_input_len}"
            )
            raise ValueError(msg)

        languages: list[str] = [""] * len(audio_signals)
        confidences: list[float] = [0.0] * len(audio_signals)
        max_bs = min(max(1, int(self.batch_size)), max(1, int(self.model.max_batch_size)))

        for start in range(0, len(audio_signals), max_bs):
            end = start + max_bs
            chunk = audio_signals[start:end]
            chunk_lengths = audio_lengths[start:end]
            pad_len = min(max(chunk_lengths), self.max_samples)
            padded = [pad_or_trim(audio, pad_len) for audio in chunk]
            durations = [min(max(length, _MIN_DURATION_SAMPLES), pad_len) for length in chunk_lengths]

            decoder_input_ids = (
                torch.tensor(prompt_ids, dtype=torch.int64).repeat(len(padded), 1).to(self.model.device)
            )
            stream = torch.cuda.current_stream("cuda")
            mel, mel_input_lengths = self.model.preprocessor.get_feats(padded, durations)
            encoder_output, encoder_output_lengths = self.model.encoder.infer(mel, mel_input_lengths, stream)
            output_ids = self.model.decoder.generate(
                decoder_input_ids,
                encoder_output,
                encoder_output_lengths,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
            )

            for offset, row in enumerate(output_ids):
                lang, confidence = self._parse_language(row[0], prompt_ids)
                languages[start + offset] = lang
                confidences[start + offset] = confidence

        return languages, confidences
