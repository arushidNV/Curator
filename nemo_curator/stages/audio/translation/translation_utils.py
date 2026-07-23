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

"""Shared contract helpers for the translation pipeline.

Two things live here so the stages cannot drift apart:

1. The per-direction shard file contract. Every output file is named::

    {output_dir}/{shard_key}_{src}-{tgt}.jsonl[.done]

   where ``shard_key`` may itself contain subdirectories mirrored from the input
   manifest tree (e.g. ``en/m1``). The reader (resume + pre-flight) and the writer
   (recovery + completion) all build and parse this contract.

2. The internal scratch row keys. The reader writes them, the LLM stage consumes
   and pops them, and the expander strips them before output, so they never reach
   the final manifest. They are a fixed internal contract, intentionally not
   user-configurable.
"""

from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING, Any

from nemo_curator.stages.audio.translation.language_map import _normalize_code
from nemo_curator.stages.resources import Resources

if TYPE_CHECKING:
    from nemo_curator.stages.base import ProcessingStage

JSONL_EXT = ".jsonl"

# Canonical row field holding the source-language ISO code. The reader copies the
# manifest's source-lang column (whatever ``--source_lang_code_key`` names) into this
# fixed key, so every downstream stage (filters, writer) reads one stable field
# regardless of the input column name.
SOURCE_LANG_CODE_KEY = "source_lang"

# Working skip flag for the filtering pipeline. The reader seeds it from the input
# ``_skipme`` column; all filter markers and the LLM gate on THIS field, leaving the
# original ``_skipme`` untouched. It is a reason STRING ("" = keep).
TRANSLATION_SKIP_KEY = "translation_skipme"

# Reserved ``translation_skipme`` reason set by LLMTranslationStage when the source
# text is empty/whitespace: an expected empty translation, NOT a quality rejection.
# Filters short-circuit it (it's a non-empty skip reason) and FinalizeTranslationStage
# maps it to the empty-source sentinel instead of the filtered outcome.
EMPTY_SOURCE_REASON = "empty_source"

# Reserved ``translation_skipme`` reason set by the reader when an input row's
# ``high_quality`` field is false: skip the row (no translation) and let
# FinalizeTranslationStage map it to the filtered outcome (min quality score + note).
LOW_QUALITY_REASON = "low_quality"

# Internal scratch keys (never appear in the output manifest).
SOURCE_LANG_NAME_KEY = "source_lang_name"
TRANSLATE_TO_KEY = "translate_to"
# Internal handoff: LLMTranslationStage writes this dict, TranslationExpander
# reads it and strips it before output, so it never reaches the final manifest.
TRANSLATIONS_KEY = "translations"


def count_lines(path: str) -> int:
    """Count newline-terminated lines in a file (binary read, fast).

    Shared by the writer (partial-count recovery + completion check) and the
    reader (``.done`` validation) so the per-direction line accounting cannot
    drift between the two stages.
    """
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def direction_key(src_code: str, tgt_code: str) -> str:
    """Build the normalised ``"{src}-{tgt}"`` direction key (e.g. ``"en-de"``)."""
    return f"{_normalize_code(src_code)}-{_normalize_code(tgt_code)}"


def handle_key(shard_key: str, direction: str) -> str:
    """Build the relative handle key ``"{shard_key}_{direction}"`` (no extension)."""
    return f"{shard_key}_{direction}"


def output_paths(output_dir: str, shard_key: str, direction: str) -> tuple[str, str]:
    """Return the ``(jsonl_path, done_path)`` pair for one (shard, direction).

    Layout: ``{output_dir}/{shard_subdirs}/{direction}/{stem}_{direction}.jsonl[.done]`` — the input
    tree is mirrored, with a per-direction folder placed right above the file (e.g.
    ``.../europarl/en/en-de/m1_en-de.jsonl``). When the shard has no subdirs it's just
    ``{output_dir}/{direction}/{stem}_{direction}.jsonl``.
    """
    subdirs, stem = os.path.split(shard_key)
    jsonl = os.path.join(output_dir, subdirs, direction, f"{handle_key(stem, direction)}{JSONL_EXT}")
    return jsonl, jsonl + ".done"


def parse_handle_key(relpath_no_ext: str) -> tuple[str, str] | None:
    """Split a ``"{shard_key}_{src}-{tgt}"`` handle key into its parts.

    The last ``_`` separates the (possibly subdirectoried) shard key from the
    direction; the direction must contain ``-``. Returns ``(shard_key, direction)``
    or ``None`` when it does not match the contract. Operates on the basename handle
    key (no direction folder) — for on-disk output paths use ``parse_output_relpath``.
    """
    parts = relpath_no_ext.rsplit("_", 1)
    if len(parts) == 2 and "-" in parts[1]:
        return parts[0], parts[1]
    return None


def parse_output_relpath(relpath_no_ext: str) -> tuple[str, str] | None:
    """Parse an on-disk output relpath (sans extension) into ``(shard_key, direction)``.

    Inverse of :func:`output_paths`' layout ``"{shard_subdirs}/{direction}/{stem}_{direction}"``:
    the filename gives ``(stem, direction)``; the folder directly above it must equal that
    ``direction``; everything above that folder is the shard's subdirs. Returns ``None`` for
    anything that doesn't match (e.g. an old flat-layout file with no direction folder).
    """
    head, sep, filename = relpath_no_ext.rpartition(os.sep)
    if not sep:
        return None  # need at least "{direction}/{stem}_{direction}"
    parsed = parse_handle_key(filename)
    if parsed is None:
        return None
    stem, direction = parsed
    subdirs, sep2, dir_folder = head.rpartition(os.sep)
    if not sep2:  # head is just the direction folder, no subdirs
        subdirs, dir_folder = "", head
    if dir_folder != direction:
        return None
    shard_key = os.path.join(subdirs, stem) if subdirs else stem
    return shard_key, direction


# ----------------------------------------------------------------------------
# Bitext filtering: CLI args + stage builder
#
# ``build_bitext_filter_stages`` composes the AudioTask-native marker stages from
# ``bitext_filters`` into the post-translation filter block (placed between the
# expander and the writer). Heavy filter imports are function-local so importing
# this contract module (done early by the reader/writer stages) stays cheap.
# ----------------------------------------------------------------------------

# The translation field written by ``TranslationExpanderStage`` (its
# ``translation_key`` default) and the lang keys the writer/QE read per row.
_TRANSLATION_FIELD = "translation"
_SOURCE_LANG_FIELD = SOURCE_LANG_CODE_KEY
_TARGET_LANG_FIELD = "target_lang"

QE_SCORE_FIELDS = {
    "comet-qe": "comet_qe_score",
    "cometoid-wmt23": "pymarian_qe_score",
    "cometoid-wmt23-mqm": "pymarian_mqm_qe_score",
}


def qe_cutoff_for_model(model_name: str, args: argparse.Namespace) -> float:
    """Return the keep-cutoff for one QE model from the parsed CLI args."""
    if model_name == "comet-qe":
        return args.qe_comet_cutoff
    if model_name.startswith("cometoid"):
        return args.qe_pymarian_cutoff
    msg = f"Unsupported QE model: {model_name}"
    raise ValueError(msg)


def add_bitext_filter_args(parser: argparse.ArgumentParser) -> None:
    """Add the bitext-filtering CLI args to the translation pipeline parser.

    Args are named ``<filter>_<param>`` and grouped per filter in ``--help``.
    Every filter is **opt-in** via a single ``store_true`` flag (no ``--no-*``
    variants). All filters run **post-translation** on the (source, translation)
    pair — there is no source-side pre-translation filtering. With no filter flags
    the pipeline only translates. All filters are mark-only: rows are annotated
    (``_skipme`` + ``additional_notes``) but never dropped, so the directional
    writer's ``.done`` counting is preserved.
    """
    shared = parser.add_argument_group("bitext filtering: resources")
    shared.add_argument("--cpu_stage_cpus", type=float, default=1.0, help="CPUs per CPU filter stage.")

    cc = parser.add_argument_group("bitext filtering: character count (cc)")
    cc.add_argument(
        "--cc_tgt", action="store_true",
        help="Enable the character-count filter on the translation, post-translation.",
    )
    cc.add_argument(
        "--cc_min_chars", type=int, default=1,
        help="Minimum Unicode characters to keep in the translation. CJK-safe (no word splitter).",
    )

    lr = parser.add_argument_group("bitext filtering: length ratio")
    lr.add_argument(
        "--length_ratio", action="store_true",
        help="Enable the character-count length-ratio filter on the (source, translation) pair.",
    )
    lr.add_argument(
        "--length_ratio_max", type=float, default=9.0,
        help="Max src/tgt character-count ratio (CJK-safe; no word splitter).",
    )

    hist = parser.add_argument_group("bitext filtering: histogram")
    hist.add_argument(
        "--histogram_tgt", action="store_true",
        help="Enable the NLLB histogram language check on the translation, post-translation.",
    )
    hist.add_argument("--histogram_threshold", type=float, default=0.8)
    hist.add_argument("--histogram_cache_dir", type=str, default=None)

    langid = parser.add_argument_group("bitext filtering: language id (fastText)")
    langid.add_argument(
        "--langid_tgt", action="store_true",
        help="Enable the fastText language-id check on the translation, post-translation. Requires --langid_model_path.",
    )
    langid.add_argument(
        "--langid_model_path", type=str, default=None,
        help="Path to a fastText langid model (required when --langid_tgt is enabled).",
    )
    langid.add_argument("--langid_min_score", type=float, default=0.5)

    qe = parser.add_argument_group("bitext filtering: quality estimation (QE)")
    qe.add_argument("--qe", action="store_true", help="Enable COMET / Cometoid QE filtering.")
    qe.add_argument("--qe_models", nargs="+", default=["cometoid-wmt23"], choices=sorted(QE_SCORE_FIELDS))
    qe.add_argument("--qe_mode", choices=["simple", "always_en_x", "bidi"], default="always_en_x")
    qe.add_argument("--qe_cpu", action="store_true", help="Run QE on CPU (avoids GPU contention with vLLM).")
    qe.add_argument("--qe_cpus", type=float, default=1.0)
    qe.add_argument(
        "--qe_num_workers", type=int, default=None,
        help="Pin the QE actor-pool size (default: autoscale to fill CPUs). A small value "
             "cuts startup time — fewer actors means fewer model loads.",
    )
    qe.add_argument("--qe_batch_size", type=int, default=64)
    qe.add_argument("--qe_comet_cutoff", type=float, default=-0.5)
    qe.add_argument("--qe_pymarian_cutoff", type=float, default=0.6)
    qe.add_argument("--qe_pymarian_shard_size", type=int, default=5000)
    qe.add_argument(
        "--qe_pymarian_workspace", type=int, default=8000,
        help="Marian GPU workspace in MB (-w). GPU/cometoid only.",
    )
    qe.add_argument(
        "--qe_pymarian_mini_batch", type=int, default=32,
        help="Marian GPU mini-batch (--mini-batch): pairs per GPU forward. GPU/cometoid only. "
             "Benchmark sweet spot ~64-128 (NOT the same as --qe_batch_size, which is the Ray batch).",
    )
    qe.add_argument(
        "--qe_pymarian_log_level", type=str, default="info",
        choices=["trace", "debug", "info", "warn", "err", "critical", "off"],
        help="Marian Evaluator --log-level (cometoid only). Use 'trace'/'debug' to see how far "
             "init/inference gets before a crash. (Marian has no --debug flag.)",
    )
    qe.add_argument(
        "--qe_pymarian_args", type=str, default=None,
        help="Full Marian arg string override; REPLACES the composed -w/--mini-batch/-d/--log-level args.",
    )

    regex = parser.add_argument_group("bitext filtering: regex cleanup")
    regex.add_argument(
        "--regex_cleanup", action="store_true",
        help="Enable regex cleanup of the translation output. (Strips non-Latin/Cyrillic/Greek scripts.)",
    )
    regex.add_argument(
        "--regex_num_workers", type=int, default=None,
        help="Pin the regex-cleanup actor-pool size to run it as its OWN parallel stage "
             "(un-fused from the single-actor writer). Default (None) keeps it a task stage. "
             "Raise it when regex cleanup is the tail bottleneck.",
    )


def _candidate_langs(args: argparse.Namespace) -> list[str]:
    """Languages a row's source/target may be, for building per-language filters.

    The reader is English-centric and drops any row whose source language is
    neither ``en`` nor a configured target, so every row that reaches the filters
    has a source/target language in ``{"en"} ∪ target_langs``.
    """
    return sorted(lang for lang in ({"en"} | {_normalize_code(c) for c in args.target_langs}) if lang)


def _require_fasttext_model(args: argparse.Namespace, flag_name: str, enabled: bool) -> None:
    """A langid stage needs a model path; fail loudly if it was enabled without one."""
    if enabled and not args.langid_model_path:
        msg = f"{flag_name} requires --langid_model_path to be set"
        raise ValueError(msg)


def build_bitext_filter_stages(args: argparse.Namespace) -> list[ProcessingStage]:
    """Post-translation bitext markers placed between the expander and the writer.

    Operate on the source text (``args.text_key``) and the translated text
    (``translation``). All mark-only; QE and regex cleanup are opt-in.
    """
    from nemo_curator.stages.audio.translation.bitext_filters import (
        AudioTaskBitextMarker,
        AudioTaskFieldMarker,
        AudioTaskMarkerChain,
        AudioTaskQEMarker,
        AudioTaskRegexModifier,
        CharCountFilter,
        CharLengthRatioFilter,
        FinalizeTranslationStage,
    )

    _require_fasttext_model(args, "--langid_tgt", args.langid_tgt)
    cpu = Resources(cpus=args.cpu_stage_cpus)
    cand = _candidate_langs(args)

    # Per-row mark-only markers, fused into one chain stage (one hand-off per row).
    markers: list[Any] = []

    if args.cc_tgt:
        # Character count (CJK-safe): language-agnostic, single filter (no per-lang dispatch).
        markers.append(
            AudioTaskFieldMarker(
                filter_obj=CharCountFilter(min_chars=args.cc_min_chars),
                text_key=_TRANSLATION_FIELD,
                score_key="tgt_char_count_score",
                name="TgtCharCount",
            )
        )

    # Length ratio is a bitext (both-sides) filter. Character-count ratio is
    # language-agnostic, so a single filter handles every direction.
    if args.length_ratio:
        markers.append(
            AudioTaskBitextMarker(
                filter_obj=CharLengthRatioFilter(max_ratio=args.length_ratio_max),
                src_key=args.text_key,
                tgt_key=_TRANSLATION_FIELD,
                score_key="length_ratio_score",
                name="LengthRatio",
            )
        )

    if args.histogram_tgt:
        from nemo_curator.stages.text.filters.histogram import HistogramFilter

        markers.append(
            AudioTaskFieldMarker(
                filters_by_lang={
                    lang: HistogramFilter(
                        lang=lang, threshold=args.histogram_threshold, cache_dir=args.histogram_cache_dir
                    )
                    for lang in cand
                },
                lang_key=_TARGET_LANG_FIELD,
                text_key=_TRANSLATION_FIELD,
                score_key="tgt_histogram_score",
                name="TgtHistogram",
            )
        )

    if args.langid_tgt:
        from nemo_curator.stages.text.filters.fasttext import FastTextLangId

        markers.append(
            AudioTaskFieldMarker(
                filters_by_lang={
                    lang: FastTextLangId(
                        model_path=args.langid_model_path,
                        min_langid_score=args.langid_min_score,
                        expected_lang=lang,
                    )
                    for lang in cand
                },
                lang_key=_TARGET_LANG_FIELD,
                text_key=_TRANSLATION_FIELD,
                score_key="tgt_langid_score",
                name="TgtLangId",
            )
        )

    stages: list[ProcessingStage] = []
    if markers:
        stages.append(AudioTaskMarkerChain(markers=markers, name="TargetFilters").with_(resources=cpu))

    # QE is a separate batched (GPU) stage; it scores rows not already marked
    # translation_skipme (score-only — it never skips). Regex cleanup is a separate
    # CPU stage that also skips already-marked rows.
    if args.qe:
        for idx, model_name in enumerate(args.qe_models):
            model_kwargs: dict[str, Any] = {}
            if model_name.startswith("cometoid"):
                model_kwargs["shard_size"] = args.qe_pymarian_shard_size
                if args.qe_pymarian_args:
                    # Full override: caller supplies the entire Marian arg string.
                    model_kwargs["marian_args"] = args.qe_pymarian_args
                elif args.qe_cpu:
                    # CPU marian defaults to --cpu-threads 1, which wastes the cores each
                    # actor reserves (qe_cpus). Use them all so the actor isn't single-
                    # threaded (otherwise QE is the pipeline's tail bottleneck).
                    model_kwargs["marian_args"] = (
                        f"--cpu-threads {int(args.qe_cpus)} -w 2000 "
                        f"--log-level {args.qe_pymarian_log_level}"
                    )
                else:
                    # GPU: compose -w/--mini-batch from CLI so they're tunable (default 32
                    # mini-batch leaves GPU throughput on the table; ~64-128 is the sweet
                    # spot). -d 0 because Ray pins one GPU per actor (seen as device 0).
                    # --log-level trace/debug surfaces how far init gets before a segfault.
                    model_kwargs["marian_args"] = (
                        f"-w {args.qe_pymarian_workspace} "
                        f"--mini-batch {args.qe_pymarian_mini_batch} -d 0 "
                        f"--log-level {args.qe_pymarian_log_level}"
                    )
            gpu = not args.qe_cpu
            qe_resources = Resources(cpus=args.qe_cpus, gpus=1.0) if gpu else Resources(cpus=args.qe_cpus)
            stages.append(
                AudioTaskQEMarker(
                    model_name=model_name,
                    cutoff=qe_cutoff_for_model(model_name, args),
                    mode=args.qe_mode,
                    gpu=gpu,
                    src_key=args.text_key,
                    tgt_key=_TRANSLATION_FIELD,
                    src_lang_key=_SOURCE_LANG_FIELD,
                    tgt_lang_key=_TARGET_LANG_FIELD,
                    score_key=QE_SCORE_FIELDS[model_name],
                    # Only the first QE model surfaces translation_quality_score; the
                    # rest only note their score / gate skip (no temp column either).
                    surface_quality=(idx == 0),
                    num_workers_override=args.qe_num_workers,
                    # CamelCase, colon-free stage name, e.g. cometoid-wmt23 -> QECometoidWmt23.
                    name="QE" + "".join(part.capitalize() for part in model_name.replace("-", " ").split()),
                    model_kwargs=model_kwargs,
                ).with_(resources=qe_resources, batch_size=args.qe_batch_size)
            )

    if args.regex_cleanup:
        stages.append(
            AudioTaskRegexModifier(
                field_key=_TRANSLATION_FIELD,
                num_workers_override=args.regex_num_workers,
            ).with_(resources=cpu)
        )

    # Finalize (always): normalize the translation fields for the empty-source /
    # filtered / kept cases (driven by the translation_skipme reason). Each upstream
    # stage already cleaned up its own temporary fields.
    stages.append(FinalizeTranslationStage().with_(resources=cpu))

    return stages
