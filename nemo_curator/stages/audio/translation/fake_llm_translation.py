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

"""CPU-only fake translation stage for dry-running the pipeline.

``FakeLLMTranslationStage`` is a drop-in replacement for ``LLMTranslationStage``
that produces a **deterministic pseudo-random phrase in the target language**
instead of calling vLLM. It mirrors the real stage's I/O contract and skip/empty
handling exactly (same ``translations`` dict, same ``additional_notes``, same
``translation_skipme`` / ``empty_source`` behaviour), so the rest of the pipeline
— expander, bitext filters, QE, finalize, directional writer, ``.done`` resume —
behaves identically while skipping the heavy model load and GPU.

Use it to test the pipeline end-to-end (filters/QE/writer/resume) in seconds:
the example exposes it via ``--nmt_dry_run``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from nemo_curator.stages.audio.pipeline_utils import set_note
from nemo_curator.stages.audio.translation.language_map import _normalize_code, name_to_code
from nemo_curator.stages.audio.translation.translation_utils import (
    EMPTY_SOURCE_REASON,
    SOURCE_LANG_NAME_KEY,
    TRANSLATE_TO_KEY,
    TRANSLATIONS_KEY,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

# A few canned phrases per target-language ISO code; one is picked per row by a
# hash of (source text, target lang), so output is deterministic but varied.
_FAKE_BANK: dict[str, list[str]] = {
    "en": ["This is a test translation.", "The quick brown fox jumps over the lazy dog.", "Hello world."],
    "de": ["Dies ist eine Testübersetzung.", "Der schnelle braune Fuchs springt über den faulen Hund.", "Hallo Welt."],
    "fr": ["Ceci est une traduction de test.", "Le renard brun rapide saute par-dessus le chien paresseux.", "Bonjour le monde."],
    "ru": ["Это тестовый перевод.", "Быстрая бурая лиса прыгает через ленивую собаку.", "Привет, мир."],
    "it": ["Questa è una traduzione di prova.", "La rapida volpe marrone salta sopra il cane pigro.", "Ciao mondo."],
    "es": ["Esta es una traducción de prueba.", "El rápido zorro marrón salta sobre el perro perezoso.", "Hola mundo."],
    "pt": ["Esta é uma tradução de teste.", "A rápida raposa marrom pula sobre o cão preguiçoso.", "Olá mundo."],
    "sv": ["Detta är en testöversättning.", "Den snabba bruna räven hoppar över den lata hunden.", "Hej världen."],
    "pl": ["To jest tłumaczenie testowe.", "Szybki brązowy lis przeskakuje nad leniwym psem.", "Witaj świecie."],
    "zh": ["这是一个测试翻译。", "敏捷的棕色狐狸跳过了懒狗。", "你好，世界。"],
    "ko": ["이것은 테스트 번역입니다.", "빠른 갈색 여우가 게으른 개를 뛰어넘습니다.", "안녕하세요 세계."],
}


@dataclass
class FakeLLMTranslationStage(ProcessingStage[AudioTask, AudioTask]):
    """Fake (CPU, no-vLLM) translator: emits a target-language phrase per direction.

    Same fields/contract as ``LLMTranslationStage`` for the parts the pipeline
    relies on; the model/sampling knobs are dropped. Stateless (no ``setup``), so
    it runs as a cheap CPU task pool.
    """

    name: str = "LLMTranslation"  # keep the real name so additional_notes/output match
    text_key: str = "tn_raw"
    source_lang_key: str = SOURCE_LANG_NAME_KEY
    target_lang_key: str = TRANSLATE_TO_KEY
    translations_key: str = TRANSLATIONS_KEY
    skip_me_key: str = "translation_skipme"
    notes_key: str = "additional_notes"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 512
    _n_processed: int = field(default=0, init=False, repr=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key, self.target_lang_key, self.source_lang_key, self.skip_me_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.translations_key]

    def _fake_translation(self, text: str, target_display_name: str) -> str:
        lang = _normalize_code(name_to_code(target_display_name))
        phrases = _FAKE_BANK.get(lang) or _FAKE_BANK["en"]
        idx = int(hashlib.md5(f"{text}|{lang}".encode()).hexdigest(), 16) % len(phrases)
        return phrases[idx]

    def _emit_empty_translations(self, task: AudioTask, targets: list[str], note: str | None = None) -> None:
        translations = task.data.get(self.translations_key) or {}
        for target_lang in targets:
            translations.setdefault(target_lang, "")
        task.data[self.translations_key] = translations
        if note is not None:
            set_note(task.data, self.name, note, self.notes_key)

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        for task in tasks:
            data = task.data
            # Pop the resolver scratch up-front, matching LLMTranslationStage.
            raw_targets = data.pop(self.target_lang_key, None) or []
            data.pop(self.source_lang_key, "")

            if isinstance(raw_targets, str):
                raw_targets = [raw_targets]
            targets = list(dict.fromkeys(raw_targets))
            if not targets:
                continue

            # Flagged (input/source-prefilter) skip: empty translations, no note.
            if data.get(self.skip_me_key, ""):
                self._emit_empty_translations(task, targets)
                continue

            text = data.get(self.text_key, "")
            if not text or not text.strip():
                # Empty source: mark the working flag + emit empty (same as the real stage).
                data[self.skip_me_key] = EMPTY_SOURCE_REASON
                self._emit_empty_translations(task, targets, "applied (empty_text_skipped)")
                continue

            translations = data.get(self.translations_key) or {}
            for target_lang in targets:
                translations[target_lang] = self._fake_translation(text, target_lang)
            data[self.translations_key] = translations
            set_note(data, self.name, "applied (translated)", self.notes_key)
            self._n_processed += 1
        return tasks
