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

"""AudioTask-native bitext filtering stages for the translation pipeline.

These stages mark (never drop) rows, so the row count per ``(shard, direction)``
stays equal to ``direction_counts`` and ``DirectionalShardedWriterStage`` can
still rename each shard's ``.jsonl`` to ``.jsonl.done``. Each applied stage records
an ``applied (...)`` note (with its score) via ``set_note`` into ``additional_notes``
— the same convention ``LLMTranslationStage`` uses — and a rejected row also gets the
working ``translation_skipme`` gate set (the gate carries the skip decision; the note
carries the score). The score is kept only in the note, so no temporary score column
is left on the row.

Each marker reuses the existing string-level scoring objects
(``DocumentFilter`` / ``BitextFilter`` from ``nemo_curator.stages.text.filters``)
and the QE ``QEModel`` wrappers, applied per row, so there is no
``DocumentBatch`` round-trip and every ``AudioTask`` keeps its ``_metadata``
(``_shard_key`` / ``direction_counts``) intact for the writer.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.audio.pipeline_utils import set_note
from nemo_curator.stages.audio.translation.language_map import _normalize_code
from nemo_curator.stages.audio.translation.translation_utils import EMPTY_SOURCE_REASON
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata
    from nemo_curator.stages.text.filters.bitext import BitextFilter
    from nemo_curator.stages.text.filters.doc_filter import DocumentFilter

# Working skip flag every filter/LLM gates on. The reader seeds it from the input
# skip column (the ONLY place that reads the original ``_skipme``); nothing here
# touches the original flag.
WORK_SKIP_KEY = "translation_skipme"
NOTES_KEY = "additional_notes"

# Fixed regex cleanup applied to the translated field (ported verbatim from the
# bitext-cleaning recipe). The substitutions run in order and the result is
# stripped.
REGEX_PARAMS_LIST: list[dict[str, str]] = [
    {"pattern": "’", "repl": "'"},
    {"pattern": "‘", "repl": "'"},
    {"pattern": "—", "repl": "-"},
    {"pattern": "–", "repl": "-"},
    {"pattern": "-", "repl": "-"},
    {"pattern": "_", "repl": " "},
    {"pattern": "——", "repl": "-"},
    {"pattern": "Ё", "repl": "Е"},
    {"pattern": "ё", "repl": "е"},
    {"pattern": "♫", "repl": " "},
    {"pattern": "♪", "repl": " "},
    {"pattern": "♬", "repl": " "},
    {"pattern": "♩", "repl": " "},
    {"pattern": "♭", "repl": " "},
    {"pattern": r"\|", "repl": " "},
    {"pattern": ";", "repl": ","},
    {"pattern": r"\[[^\]]*\]", "repl": ""},
    {"pattern": r" ?\([^\)]+\)", "repl": ""},
    {"pattern": r" ?{[^}]+}", "repl": ""},
    # DISABLED — destructive and NOT script-aware: this negated character class keeps
    # ONLY Latin/Cyrillic/Greek letters + digits/punctuation/currency and DELETES
    # everything else, so for a target in CJK/Arabic/Hebrew/Hangul/Devanagari/Thai it
    # strips the whole translation -> "". Commented out so regex cleanup runs only the
    # benign normalizations (quotes/dashes/whitespace/bracket-stripping). Re-enable
    # ONLY if every target language uses Latin/Cyrillic/Greek script.
    # {
    #     "pattern": "[^ !$%',-.0123456789;?ABCDEFGHIJKLMNOPQRSßTUVWXYŸZabcdefghijklmnopqrsẞtuvwxyÿz¡£¿ÀÁÂÃÄÅÆÇÈÉÊÌÍÎÑÒÓÔÕÖØÙÚÜÝàáâãäåæçèéêëìíîïñòóôõöøùúûüýĀāĂăĄąĆćĊċČčĎďĐđĒēĖėĘęĚěĠġĢģĦħĪīĮįĶķĹĺĻļĽľŁłŃńŅņŇňŐőŒœŔŕŘřŚśŠšŤťŪūŮůŰűŲųŹźŻżŽžȘșȚțΆΈΉΌΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩάέήίαβγδεζηθικλμνξοπρστυφχψωϊόύώЁЄІЇАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяёєіїҐґ€₴₽/:]",
    #     "repl": "",
    # },
    {"pattern": r"\s+\.", "repl": "."},
    {"pattern": r"\?+", "repl": "?"},
    {"pattern": r"\.+", "repl": "."},
    {"pattern": ",+", "repl": ","},
    {"pattern": "!+", "repl": "!"},
    {"pattern": r"\s+", "repl": " "},
    {"pattern": r" ([.,!?])", "repl": r"\1"},
    {"pattern": " '", "repl": "'"},
    {"pattern": "' ", "repl": "'"},
    {"pattern": "'+", "repl": "'"},
    {"pattern": " - ", "repl": "-"},
]


def _fmt_score(score: Any) -> str:
    """Render a filter score for notes / skip reasons, rounding floats to 2 digits.

    Handles ints (char counts), floats (ratios / confidences), lists/tuples, and the
    stringified ``[confidence, lang]`` that ``FastTextLangId.score_document`` returns
    (rounding the confidence, keeping the language tag).
    """
    if isinstance(score, bool):
        return str(score)
    if isinstance(score, float):
        return f"{score:.2f}"
    if isinstance(score, (list, tuple)):
        return "[" + ", ".join(_fmt_score(x) for x in score) + "]"
    if isinstance(score, str):
        try:
            parsed = ast.literal_eval(score)
        except (ValueError, SyntaxError):
            return score
        if isinstance(parsed, (list, tuple, float)):
            return _fmt_score(parsed)
        return score
    return str(score)


def _is_skipped(task: AudioTask, skip_key: str = WORK_SKIP_KEY) -> bool:
    """True when this row is already marked for skipping (non-empty reason string)."""
    return bool(task.data.get(skip_key, ""))


def _set_skip(task: AudioTask, reason: str, skip_key: str = WORK_SKIP_KEY) -> None:
    """Mark the row skipped by writing the **reason** string into the working gate.

    The gate is a string (like the input ``_skipme``): empty = keep, non-empty =
    skip-because-of-``reason`` — a verbose ``"<StageName> (<score_key>=<value>)"``
    describing which filter rejected the row and its score. Written only on the
    **first** skip — an already-set reason is preserved (first reason wins), matching
    the per-row short-circuit in the markers. The same score is also recorded in
    ``additional_notes`` by ``_add_note``.
    """
    if not task.data.get(skip_key):
        task.data[skip_key] = reason


def _add_note(task: AudioTask, stage_name: str, detail: str, notes_key: str = NOTES_KEY) -> None:
    """Record a per-stage ``applied (...)`` note in ``additional_notes``.

    Written on every application (pass *or* fail) so ``additional_notes`` lists the
    important stages that actually ran on the row, with their score. The skip
    decision itself is carried only by ``translation_skipme`` (no skip text here).
    """
    set_note(task.data, stage_name, detail, notes_key)


def _sanitize_qe_field(text: object) -> str:
    """Make one field safe for the cometoid/pymarian QE input (NON-mutating helper).

    Marian reads one tab-separated line per pair with ``#fields == #vocabs`` (2 for comet-qe QE)
    and length-filters empty sentences. So a ``\\t`` in the text adds a column (-> hard SIGABRT
    ``Number of fields does not match number of vocabs``), and an **empty** field makes Marian
    return fewer scores than inputs (-> ``assert len(scores) == len(batch)`` in pymarian). Collapse
    whitespace runs (tabs/newlines -> space) and replace an empty field with ``"."``.

    This is applied ONLY to the throwaway strings handed to the QE model — it never rewrites the
    row's stored ``translation``/source fields. Verified locally against ``marian-nmt/cometoid22-wmt23``.
    """
    return " ".join(str(text or "").split()) or "."


class CharCountFilter:
    """Length filter counting Unicode characters (CJK-safe; no word splitter).

    A drop-in replacement for ``WordCountFilter`` that counts characters instead of
    words, so it works for space-less scripts (zh/ja/...) without jieba/MeCab. Counts
    every character of the outer-stripped text (internal spaces included). Plain
    scoring object (no ``DocumentFilter`` base) — the marker stages duck-type
    ``score_document`` / ``keep_document``.
    """

    def __init__(self, min_chars: int = 1, max_chars: int = 100000) -> None:
        self._min_chars = min_chars
        self._max_chars = max_chars
        self._name = "char_count"

    def score_document(self, text: str) -> float:
        return len(text.strip())

    def keep_document(self, score: float) -> bool:
        return self._min_chars <= score <= self._max_chars


class CharLengthRatioFilter:
    """Source/target character-count ratio (CJK-safe; language-agnostic).

    A drop-in replacement for ``LengthRatioFilter`` that uses character counts
    instead of word counts, so no per-language word splitter is needed. Mirrors
    ``LengthRatioFilter.score_bitext``: ``math.inf`` when either side is empty, else
    ``max(src/tgt, tgt/src)``. Plain scoring object — markers duck-type
    ``score_bitext`` / ``keep_bitext``.
    """

    def __init__(self, max_ratio: float = 9.0) -> None:
        self._max_ratio = float(max_ratio)
        self._name = "char_length_ratio"

    def score_bitext(self, src: str, tgt: str) -> float:
        src_len = len(src.strip())
        tgt_len = len(tgt.strip())
        if src_len == 0 or tgt_len == 0:
            return math.inf
        return max(src_len / tgt_len, tgt_len / src_len)

    def keep_bitext(self, score: float) -> bool:
        return score < self._max_ratio


class RegexSubstitutionModifier:
    """Apply a fixed sequence of regex substitutions to one text value."""

    def __init__(self, regex_params_list: list[dict[str, str]]) -> None:
        self._substitutions = [(re.compile(p["pattern"]), p["repl"]) for p in regex_params_list]

    def modify_document(self, text: object) -> str:
        value = "" if text is None else str(text)
        for pattern, repl in self._substitutions:
            value = pattern.sub(repl, value)
        return value.strip()


@dataclass
class AudioTaskFieldMarker(ProcessingStage[AudioTask, AudioTask]):
    """Mark rows by scoring one text field with a single-field ``DocumentFilter``.

    Two modes:
      * fixed: pass ``filter_obj`` — every row scored with the same filter.
      * per-row language dispatch: pass ``filters_by_lang`` (``{lang_code:
        DocumentFilter}``) + ``lang_key``. Each row is scored with the filter for
        its own ``_normalize_code(row[lang_key])``. This keeps genuinely
        language-specific filters (histogram, fastText) correct across a
        bidirectional run (en->X and X->en) where the source/target language
        varies per row. Rows whose language has no entry are left unscored.

    Reuses the filter's ``score_document`` / ``keep_document``. Rows already
    marked are not re-scored. The score is recorded only in ``additional_notes``
    (``applied (score_key=value)``) — no temporary score column is left on the
    row, so each stage cleans up after itself and the output keeps just the input
    fields plus the translation fields. Rejected rows additionally get the working
    skip gate set.
    """

    filter_obj: DocumentFilter | None = None
    filters_by_lang: dict[str, DocumentFilter] | None = None
    lang_key: str | None = None
    text_key: str = ""
    score_key: str = ""
    name: str = "FieldMarker"
    skip_key: str = WORK_SKIP_KEY
    notes_key: str = NOTES_KEY

    def _all_filters(self) -> list[DocumentFilter]:
        if self.filters_by_lang is not None:
            return list(self.filters_by_lang.values())
        return [self.filter_obj] if self.filter_obj is not None else []

    def _pick_filter(self, task: AudioTask) -> DocumentFilter | None:
        if self.filters_by_lang is not None:
            lang = _normalize_code(str(task.data.get(self.lang_key, "")))
            return self.filters_by_lang.get(lang)
        return self.filter_obj

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.notes_key]

    def requires_setup(self) -> bool:
        return any(hasattr(f, "load_model") or hasattr(f, "load_tokenizer") for f in self._all_filters())

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_ACTOR_STAGE: self.requires_setup()}

    def setup_on_node(self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None) -> None:
        for filter_obj in self._all_filters():
            if hasattr(filter_obj, "model_check_or_download"):
                filter_obj.model_check_or_download()

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        for filter_obj in self._all_filters():
            if hasattr(filter_obj, "load_model"):
                filter_obj.load_model()
            if hasattr(filter_obj, "load_tokenizer"):
                filter_obj.load_tokenizer()

    def apply_row(self, task: AudioTask) -> bool:
        """Score and (maybe) mark one row. Returns True if the row is now skipped.

        Used both by ``process_batch`` and by ``AudioTaskMarkerChain`` to run
        several markers in one stage. Returns True for an already-marked row so a
        chain short-circuits the remaining markers.
        """
        if _is_skipped(task, self.skip_key):
            return True
        filter_obj = self._pick_filter(task)
        if filter_obj is None:
            return False
        text = str(task.data.get(self.text_key, "") or "")
        score = filter_obj.score_document(text)
        detail = f"{self.score_key}={_fmt_score(score)}"
        _add_note(task, self.name, f"applied ({detail})", self.notes_key)
        if not filter_obj.keep_document(score):
            _set_skip(task, f"{self.name} ({detail})", self.skip_key)
            return True
        task.data.setdefault(self.skip_key, "")
        return False

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        for task in tasks:
            self.apply_row(task)
        return tasks


@dataclass
class AudioTaskBitextMarker(ProcessingStage[AudioTask, AudioTask]):
    """Mark rows by scoring a source/target pair with a ``BitextFilter``.

    Two modes:
      * fixed: pass ``filter_obj``.
      * per-row language dispatch: pass ``filters_by_pair`` (``{(src_lang,
        tgt_lang): BitextFilter}``) + ``src_lang_key``/``tgt_lang_key``. Each row
        is scored with the filter for its own
        ``(_normalize_code(row[src_lang_key]), _normalize_code(row[tgt_lang_key]))``,
        so length-ratio word splitting stays correct per direction. Rows whose
        language pair has no entry are left unscored.

    Reuses ``score_bitext`` / ``keep_bitext`` per row; same mark-only / skip-
    already-marked semantics as ``AudioTaskFieldMarker``.
    """

    filter_obj: BitextFilter | None = None
    filters_by_pair: dict[tuple[str, str], BitextFilter] | None = None
    src_lang_key: str | None = None
    tgt_lang_key: str | None = None
    src_key: str = ""
    tgt_key: str = ""
    score_key: str = ""
    name: str = "BitextMarker"
    skip_key: str = WORK_SKIP_KEY
    notes_key: str = NOTES_KEY

    def _pick_filter(self, task: AudioTask) -> BitextFilter | None:
        if self.filters_by_pair is not None:
            src_lang = _normalize_code(str(task.data.get(self.src_lang_key, "")))
            tgt_lang = _normalize_code(str(task.data.get(self.tgt_lang_key, "")))
            return self.filters_by_pair.get((src_lang, tgt_lang))
        return self.filter_obj

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.src_key, self.tgt_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.notes_key]

    def requires_setup(self) -> bool:
        return False  # BitextFilter (length ratio) loads no model

    def setup_on_node(self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None) -> None:
        return None

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        return None

    def apply_row(self, task: AudioTask) -> bool:
        """Score and (maybe) mark one row; returns True if the row is now skipped."""
        if _is_skipped(task, self.skip_key):
            return True
        filter_obj = self._pick_filter(task)
        if filter_obj is None:
            return False
        src = str(task.data.get(self.src_key, "") or "")
        tgt = str(task.data.get(self.tgt_key, "") or "")
        score = filter_obj.score_bitext(src, tgt)
        detail = f"{self.score_key}={_fmt_score(score)}"
        _add_note(task, self.name, f"applied ({detail})", self.notes_key)
        if not filter_obj.keep_bitext(score):
            _set_skip(task, f"{self.name} ({detail})", self.skip_key)
            return True
        task.data.setdefault(self.skip_key, "")
        return False

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        for task in tasks:
            self.apply_row(task)
        return tasks


@dataclass
class AudioTaskMarkerChain(ProcessingStage[AudioTask, AudioTask]):
    """Run several field/bitext markers in a single pass over each row.

    Fuses what would otherwise be separate per-row CPU marker stages into one
    stage, so a row is handed between pipeline operators once instead of N times.
    Each member marker (``AudioTaskFieldMarker`` / ``AudioTaskBitextMarker``)
    still records its own ``additional_notes`` score note; the chain applies them
    in order and short-circuits as soon as one sets the working skip gate
    (preserving the standalone "first reject wins" behavior). Members
    are held as scoring objects only — their resources/specs are ignored; the
    chain's own ``.with_(...)``/``ray_stage_spec`` govern scheduling.
    """

    markers: list[Any] = field(default_factory=list)
    name: str = "MarkerChain"
    notes_key: str = NOTES_KEY

    def inputs(self) -> tuple[list[str], list[str]]:
        keys: list[str] = []
        for marker in self.markers:
            keys.extend(marker.inputs()[1])
        return [], sorted(set(keys))

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.notes_key]

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_ACTOR_STAGE: any(m.requires_setup() for m in self.markers)}

    def setup_on_node(self, node_info: NodeInfo | None = None, worker_metadata: WorkerMetadata | None = None) -> None:
        for marker in self.markers:
            marker.setup_on_node(node_info, worker_metadata)

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        for marker in self.markers:
            marker.setup(worker_metadata)

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        for task in tasks:
            # Each marker's apply_row short-circuits an already-skipped row (returns
            # True), so the first marker breaks the loop — no separate skip check.
            for marker in self.markers:
                if marker.apply_row(task):
                    break
        return tasks


@dataclass
class AudioTaskQEMarker(ProcessingStage[AudioTask, AudioTask]):
    """Score each translation with a quality-estimation model (score-only).

    ``setup()`` loads the QE model once per worker (``COMET`` or ``PyMarian``
    Cometoid). ``process_batch`` gathers every not-yet-skipped row, builds QE
    inputs (per-row direction handling for ``always_en_x`` / ``bidi``), runs a
    single ``model.predict(...)``, and records the score in ``additional_notes``
    (flagging scores below ``cutoff`` as ``(low)``). This is the only filter that
    needs a real batch.

    QE **does not drop rows** — it never sets ``translation_skipme``; ``cutoff`` is
    used only to annotate low scores in the note. When ``surface_quality`` is set,
    the score is also written to ``quality_key`` (``translation_quality_score``) so
    it becomes the row's final quality score (good or low) — no temporary per-model
    score column is left behind. With several QE models only the first surfaces its
    score (the rest only add notes).
    """

    model_name: str
    cutoff: float
    mode: str = "always_en_x"
    gpu: bool = True
    src_key: str = "tn_raw"
    tgt_key: str = "translation"
    src_lang_key: str = "source_lang"
    tgt_lang_key: str = "target_lang"
    score_key: str = "qe_score"
    quality_key: str = "translation_quality_score"
    surface_quality: bool = True
    name: str = "QEFilter"
    skip_key: str = WORK_SKIP_KEY
    notes_key: str = NOTES_KEY
    num_workers_override: int | None = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    model: Any = None
    # Set in setup(): cometoid/pymarian needs its TSV input sanitized (empty/tab guards);
    # comet-qe (torch) does not, so we gate the sanitizer on the loaded model type.
    _is_pymarian: bool = field(default=False, init=False, repr=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.src_key, self.tgt_key, self.src_lang_key, self.tgt_lang_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.quality_key, self.notes_key] if self.surface_quality else [self.notes_key]

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_ACTOR_STAGE: True}

    def num_workers(self) -> int | None:
        # Pin the actor-pool size when set (else the backend autoscales to fill CPUs,
        # which spawns many actors — each a slow model load at startup). Capping it
        # cuts warmup; throughput is unaffected since each actor uses qe_cpus threads.
        return self.num_workers_override

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if self.num_workers_override is not None:
            spec["num_workers"] = self.num_workers_override
        return spec

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        from nemo_curator.stages.text.filters.qe import PyMarianQEModel, QualityEstimationFilter

        model_cls = QualityEstimationFilter.SUPPORTED_MODELS[self.model_name]
        self.model = model_cls.load_model(self.model_name, gpu=self.gpu, **self.model_kwargs)
        # cometoid/pymarian aborts on tabs and asserts on empty fields; sanitize its input.
        self._is_pymarian = isinstance(self.model, PyMarianQEModel)

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        if self.model is None:
            msg = "QE model not initialised — setup() was not called"
            raise RuntimeError(msg)

        pending = [task for task in tasks if not _is_skipped(task, self.skip_key)]
        if not pending:
            return tasks

        scores = self._score(pending)
        for task, score in zip(pending, scores, strict=True):
            value = float(score)
            # Score-only: record the score (and flag if below cutoff) but NEVER set
            # translation_skipme — QE annotates quality, it does not drop rows.
            detail = f"{self.score_key}={_fmt_score(value)}"
            if value < self.cutoff:
                detail += f"<{self.cutoff} (low)"
            _add_note(task, self.name, f"applied ({detail})", self.notes_key)
            if self.surface_quality:
                task.data[self.quality_key] = value
        return tasks

    def _score(self, pending: list[AudioTask]) -> list[float]:
        # Build the model inputs as LOCAL copies — never mutate task.data here. For
        # cometoid/pymarian, sanitize each field (collapse tabs/newlines, "." for empties) so the
        # evaluator doesn't SIGABRT on tabs or desync its score count on empty fields. comet-qe
        # (torch) handles those itself, so it gets the raw fields.
        prep = _sanitize_qe_field if self._is_pymarian else (lambda t: str(t or ""))
        srcs = [prep(task.data.get(self.src_key, "")) for task in pending]
        tgts = [prep(task.data.get(self.tgt_key, "")) for task in pending]

        if self.mode == "simple":
            inputs = [self.model.wrap_qe_input(src, tgt) for src, tgt in zip(srcs, tgts, strict=True)]
            return [float(score) for score in self.model.predict(inputs)]

        if self.mode == "always_en_x":
            src_langs = [str(task.data.get(self.src_lang_key, "")).lower() for task in pending]
            tgt_langs = [str(task.data.get(self.tgt_lang_key, "")).lower() for task in pending]
            inputs = [
                self.model.wrap_qe_input(src, tgt, reverse=(sl != "en" and tl == "en"))
                for src, tgt, sl, tl in zip(srcs, tgts, src_langs, tgt_langs, strict=True)
            ]
            return [float(score) for score in self.model.predict(inputs)]

        # bidi: average forward and reverse scores.
        forward = [self.model.wrap_qe_input(src, tgt) for src, tgt in zip(srcs, tgts, strict=True)]
        reverse = [self.model.wrap_qe_input(src, tgt, reverse=True) for src, tgt in zip(srcs, tgts, strict=True)]
        all_scores = [float(score) for score in self.model.predict(forward + reverse)]
        mid = len(forward)
        return [(fwd + rev) / 2 for fwd, rev in zip(all_scores[:mid], all_scores[mid:], strict=True)]


@dataclass
class AudioTaskRegexModifier(ProcessingStage[AudioTask, AudioTask]):
    """Apply the fixed ``REGEX_PARAMS_LIST`` cleanup to one text field per row.

    Overwrites ``field_key`` in place. The untouched original is preserved upstream
    by ``TranslationExpanderStage`` under ``translation_raw`` (always written), so
    the verbatim text survives even when the destructive char-class rule empties or
    mangles ``translation``. Already-skipped rows are left untouched. Each evaluated
    row gets an ``additional_notes`` note (``applied (modified|unchanged)``).
    """

    field_key: str = "translation"
    regex_params: list[dict[str, str]] = field(default_factory=lambda: REGEX_PARAMS_LIST)
    name: str = "RegexCleanup"
    skip_key: str = WORK_SKIP_KEY
    notes_key: str = NOTES_KEY
    num_workers_override: int | None = None

    def __post_init__(self) -> None:
        self._modifier = RegexSubstitutionModifier(self.regex_params)

    def num_workers(self) -> int | None:
        return self.num_workers_override

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if self.num_workers_override is not None:
            spec["num_workers"] = self.num_workers_override
        return spec

    def ray_stage_spec(self) -> dict[str, Any]:
        # The per-row regex is stateless and CPU-bound. By default it's a task stage that Ray Data
        # tends to FUSE into the single-actor writer -> it runs single-threaded. Pinning a worker
        # count makes it its OWN actor pool (un-fused), so it parallelizes across CPUs while the
        # writer stays single-actor. None -> keep the prior (task) behavior.
        return {RayStageSpecKeys.IS_ACTOR_STAGE: True} if self.num_workers_override is not None else {}

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.field_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.field_key, self.notes_key]

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        for task in tasks:
            if _is_skipped(task, self.skip_key):
                continue
            original = str(task.data.get(self.field_key, "") or "")
            cleaned = self._modifier.modify_document(original)
            task.data[self.field_key] = cleaned
            _add_note(task, self.name, f"applied ({'modified' if cleaned != original else 'unchanged'})", self.notes_key)
        return tasks


@dataclass
class FinalizeTranslationStage(ProcessingStage[AudioTask, AudioTask]):
    """Finalize each row's translation fields for output.

    The internal working flag ``translation_skipme`` is **removed** here (popped) so it never reaches
    the output. Driven by its reason string, three cases — each also recording a note under this
    stage's name in ``additional_notes``:

    - empty source (reason ``"empty_source"``, set by the LLM): kept distinct from a quality
      rejection — empty ``translation`` + ``translation_raw``, ``translation_quality_score = 1``,
      note ``"empty_source"``;
    - filtered (any other reason — a filter name or an input ``_skipme`` reason):
      ``translation_quality_score = min_quality_score`` (-1.0, so rejected rows sort worst); the
      ``translation`` is KEPT (the verbatim LLM output stays auditable), and the reason that was in
      ``translation_skipme`` is moved into the note (``"skipped: <reason>"``);
    - kept (reason ``""``): ``translation_quality_score`` = the QE score already surfaced by the QE
      stage (``None`` when QE is off / didn't run), note ``"kept (quality=<score>)"``.

    Finally, for every row, if ``translation`` is empty it falls back to ``translation_raw``.
    """

    translation_key: str = "translation"
    translation_raw_key: str = "translation_raw"
    quality_key: str = "translation_quality_score"
    skip_key: str = WORK_SKIP_KEY
    notes_key: str = NOTES_KEY
    min_quality_score: float = -1.0  # score for filter-rejected rows (sorts below cometoid's 0..1)
    name: str = "FinalizeTranslation"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.translation_key, self.translation_raw_key, self.quality_key, self.notes_key]

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        for task in tasks:
            data = task.data
            reason = data.pop(self.skip_key, "") or ""  # drop translation_skipme from the output
            if reason == EMPTY_SOURCE_REASON:
                # Empty source — expected empty translation, NOT a quality rejection. Keep its
                # distinct sentinel quality 1 so it's separable from filter-rejected rows.
                data[self.translation_key] = ""
                data[self.translation_raw_key] = ""
                data[self.quality_key] = 1
                set_note(data, self.name, "empty_source", self.notes_key)
            elif reason:
                # Filtered: keep the verbatim translation, mark worst quality, note the reason
                # (moved out of translation_skipme).
                data[self.quality_key] = self.min_quality_score
                set_note(data, self.name, f"skipped: {reason}", self.notes_key)
            else:
                # Kept: keep the QE score the QE stage surfaced, else None.
                quality = data.setdefault(self.quality_key, None)
                set_note(data, self.name, f"kept (quality={quality})", self.notes_key)

            # Never emit an empty translation when a verbatim one exists.
            if not data.get(self.translation_key):
                data[self.translation_key] = data.get(self.translation_raw_key, "")
        return tasks


__all__ = [
    "AudioTaskBitextMarker",
    "AudioTaskFieldMarker",
    "AudioTaskMarkerChain",
    "AudioTaskQEMarker",
    "AudioTaskRegexModifier",
    "CharCountFilter",
    "CharLengthRatioFilter",
    "FinalizeTranslationStage",
    "REGEX_PARAMS_LIST",
    "RegexSubstitutionModifier",
]
