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

"""LLM-based translation pipeline (text-only, vLLM).

Architecture
------------
::

    TranslationManifestReader     (CPU, _EmptyTask → AudioTask)
        Composite stage = FilePartitioningStage + per-file reader.
        One input manifest == one shard.  For each row it resolves the
        source_lang ISO code → display name, writes the per-row
        translate_to list (En→X / X→En), and tags the AudioTask with
        _metadata = {_shard_key, _shard_total, direction_counts}.
        On resume, shards whose every expected direction is already
        .done are skipped; partial .jsonl files are deleted so the
        writer's append mode starts clean.

    LLMTranslationStage           (GPU, AudioTask → AudioTask)
        Batched vLLM inference; writes data["translations"]
        as {display_name: translated_text}.  Rows flagged
        translation_skipme (from the input skip column) get an empty
        translation without an LLM call.

    TranslationExpanderStage      (CPU, AudioTask → list[AudioTask])
        Fan-out: one task per direction with flat schema
        {text, source_lang, target_lang (ISO), translation}.

    build_bitext_filter_stages    (CPU/GPU, AudioTask → AudioTask)  [opt-in]
        Post-translation mark-only filters on (source, translation):
        tgt character count, length ratio, histogram/fastText, QE,
        regex cleanup, then a finalize step.
        Rows are annotated (translation_skipme + additional_notes) but
        never dropped, so the writer's per-direction .done counting holds.
        (No source-side pre-translation filtering.)

    DirectionalShardedWriterStage (CPU, AudioTask → AudioTask)
        Appends batched rows (grouped per (shard_key, direction)) to
        {output_dir}/{shard_key}_{src}-{tgt}.jsonl; renames to .done
        inline once the per-direction counter equals direction_counts
        for that direction.  setup() recovers counters from disk so
        actor restarts pick up where they left off.

Final outputs land directly at::

    {output_dir}/{shard_key}_{src}-{tgt}.jsonl.done

where ``shard_key`` mirrors the input manifest path under the input root
(subdirectories preserved), or is just the manifest stem for flat input —
e.g. ``m1_en-de.jsonl.done``, ``m1_en-fr.jsonl.done``,
``m2_en-de.jsonl.done``, …

There is no ``shards/`` subdir and no separate reconciliation step — the
writer's per-direction file *is* the final output.

Resume behaviour
----------------
Re-running with the same ``--output_dir`` skips any shard whose expected
direction ``.done`` files are all present.  Only failed or partial shards
are re-processed.

Example
-------
::

    python run_translation_pipeline.py \\
        --manifest /data/manifests \\
        --output_dir /data/translations \\
        --target_langs de fr ru ja \\
        --nmt_model_id Qwen/Qwen3-8B
"""

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

import argparse
import time

from loguru import logger

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import SlurmRayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.translation import (
    DirectionalShardedWriterStage,
    FakeLLMTranslationStage,
    LLMTranslationStage,
    TranslationExpanderStage,
    TranslationManifestReader,
    add_bitext_filter_args,
    build_bitext_filter_stages,
    merge_translation_cache,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="LLM translation pipeline (text-only, vLLM).")

    # ------------------------------------------------------------------ I/O
    ap.add_argument(
        "--manifest",
        type=str,
        required=True,
        help=(
            "Path to JSONL manifest(s). Accepts a single file, a directory (scanned "
            "recursively for *.jsonl/*.json), or a glob pattern. FilePartitioningStage "
            "handles discovery. One file == one shard."
        ),
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=(
            "Output directory. Final per-(manifest, direction) files land directly here as "
            "{stem}_{src}-{tgt}.jsonl.done — no shards/ subdir, no reconciliation step."
        ),
    )

    # ------------------------------------------------------------------ Languages
    ap.add_argument(
        "--target_langs",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Target language ISO codes (e.g. 'de fr ru ja'). "
            "TranslationManifestReader generates En→X and X→En pairs."
        ),
    )
    ap.add_argument(
        "--source_lang_code_key",
        type=str,
        default="source_lang",
        help="Input manifest key holding the source language ISO code.",
    )
    ap.add_argument(
        "--verify_done_line_counts",
        action="store_true",
        help=(
            "On resume, validate each existing .jsonl.done by comparing its line count to the "
            "expected row count; a mismatch (short/stale file) is deleted and reprocessed. Adds "
            "read I/O per resume; default off (existence-only check)."
        ),
    )

    # ------------------------------------------------------------ Translation (NMT) model
    # All translation-stage knobs are prefixed --nmt_* to mark them as LLMTranslationStage config.
    ap.add_argument(
        "--nmt_model_id",
        type=str,
        default=None,
        help="Translation LLM model ID (required unless --nmt_dry_run).",
    )
    ap.add_argument(
        "--nmt_dry_run",
        action="store_true",
        help=(
            "Use a fake CPU translator (deterministic random target-language phrases, no "
            "vLLM/GPU) to dry-run the pipeline (filters/QE/writer/.done resume) without "
            "loading the model."
        ),
    )

    # Prompt overrides (mutually exclusive pairs)
    tpg = ap.add_mutually_exclusive_group()
    tpg.add_argument("--nmt_translation_prompt", type=str, default=None)
    tpg.add_argument("--nmt_translation_prompt_file", type=str, default=None)

    spg = ap.add_mutually_exclusive_group()
    spg.add_argument("--nmt_system_prompt", type=str, default=None)
    spg.add_argument("--nmt_system_prompt_file", type=str, default=None)

    ap.add_argument("--text_key", type=str, default="tn_raw", help="Manifest key for source text.")
    ap.add_argument(
        "--skip_me_key",
        type=str,
        default="_skipme",
        help=(
            "Manifest key flagging rows to skip. When truthy (non-empty string or boolean True), "
            "the row is kept but receives an empty translation instead of being sent to the LLM."
        ),
    )
    ap.add_argument(
        "--high_quality_key",
        type=str,
        default="high_quality",
        help=(
            "Manifest key marking row quality. When explicitly false (false/0/no), the row is "
            "skipped (no translation), noted as low quality, and gets the minimum quality score."
        ),
    )

    # vLLM params
    ap.add_argument("--nmt_tensor_parallel_size", type=int, default=None)
    ap.add_argument(
        "--nmt_num_workers",
        type=int,
        default=None,
        help="Explicit number of GPU worker replicas for the translation stage under Xenna.",
    )
    ap.add_argument("--nmt_batch_size", type=int, default=512)
    ap.add_argument("--nmt_max_output_tokens", type=int, default=256)
    ap.add_argument("--nmt_max_model_len", type=int, default=1024)
    ap.add_argument("--nmt_max_num_seqs", type=int, default=512)
    ap.add_argument("--nmt_max_num_batched_tokens", type=int, default=16384)
    ap.add_argument("--nmt_gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--nmt_kv_cache_dtype", type=str, default="fp8")
    ap.add_argument("--nmt_temperature", type=float, default=0.7)
    ap.add_argument("--nmt_top_p", type=float, default=0.8)
    ap.add_argument("--nmt_top_k", type=int, default=20)
    ap.add_argument("--nmt_min_p", type=float, default=0.0)
    ap.add_argument("--nmt_presence_penalty", type=float, default=1.5)
    ap.add_argument("--nmt_repetition_penalty", type=float, default=1.0)
    ap.add_argument("--nmt_seed", type=int, default=1234)

    # ----------------------------------------------------------- Translation cache
    # Lock-free per-worker cache for short, recurring source texts. Each worker keeps an
    # in-memory cache (warm-started from a canonical file), dumps its touched entries at
    # teardown, and the driver merges all dumps into the canonical file after the run.
    ap.add_argument(
        "--nmt_cache",
        action="store_true",
        help="Enable the short-text translation cache (warm start across runs).",
    )
    ap.add_argument(
        "--nmt_cache_dir",
        type=str,
        default=None,
        help="Directory holding the canonical cache + per-worker dumps. Required with --nmt_cache.",
    )
    ap.add_argument(
        "--nmt_cache_max_chars",
        type=int,
        default=40,
        help="Only source texts with <= this many (stripped) characters are cached.",
    )
    ap.add_argument(
        "--nmt_cache_max_entries",
        type=int,
        default=0,
        help="Optional safety cap on canonical cache size (0 = unbounded).",
    )

    # ------------------------------------------------------------------ Executor
    ap.add_argument(
        "--execution_mode",
        type=str,
        default="streaming",
        choices=["streaming", "batch"],
        help="Xenna execution mode. Ignored when --executor=ray_data (Ray Data manages its own scheduling).",
    )
    ap.add_argument(
        "--executor",
        type=str,
        default="ray_data",
        choices=["ray_data", "xenna"],
        help="Pipeline executor backend. 'ray_data' uses RayDataExecutor; 'xenna' uses XennaExecutor.",
    )
    ap.add_argument(
        "--slurm",
        action="store_true",
        help=(
            "Bootstrap a multi-node Ray cluster via SlurmRayClient. Launch the script on every "
            "node (srun --ntasks-per-node=1): the head (SLURM_NODEID=0) runs the pipeline while "
            "workers join the cluster and block until teardown. Omit for single-node runs."
        ),
    )

    add_bitext_filter_args(ap)
    return ap


def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.nmt_cache and not args.nmt_cache_dir:
        raise ValueError("--nmt_cache requires --nmt_cache_dir.")
    # One run id shared by every worker this run, so their dumps land in a common dir
    # ({nmt_cache_dir}/dumps/{run_id}/) that the driver merges afterwards.
    cache_run_id = os.environ.get("SLURM_JOB_ID") or str(int(time.time()))

    # Fake (CPU) translator for dry-runs, else the real vLLM stage. Both share the
    # same I/O contract, so the rest of the pipeline is identical either way.
    if args.nmt_dry_run:
        logger.info("DRY RUN: using FakeLLMTranslationStage (no vLLM/GPU).")
        translate_stage = FakeLLMTranslationStage(
            text_key=args.text_key,
            batch_size=args.nmt_batch_size,
        )
    else:
        if not args.nmt_model_id:
            raise ValueError("--nmt_model_id is required (or use --nmt_dry_run).")
        translate_stage = LLMTranslationStage(
            model_id=args.nmt_model_id,
            translation_prompt=args.nmt_translation_prompt,
            translation_prompt_file=args.nmt_translation_prompt_file,
            system_prompt=args.nmt_system_prompt,
            system_prompt_file=args.nmt_system_prompt_file,
            text_key=args.text_key,
            # skip_me_key defaults to the working flag (translation_skipme), which the
            # reader seeds from the input skip column — so input-flagged rows skip vLLM.
            tensor_parallel_size=args.nmt_tensor_parallel_size,
            num_workers_override=args.nmt_num_workers,
            max_output_tokens=args.nmt_max_output_tokens,
            max_model_len=args.nmt_max_model_len,
            max_num_seqs=args.nmt_max_num_seqs,
            max_num_batched_tokens=args.nmt_max_num_batched_tokens,
            gpu_memory_utilization=args.nmt_gpu_memory_utilization,
            kv_cache_dtype=args.nmt_kv_cache_dtype,
            temperature=args.nmt_temperature,
            top_p=args.nmt_top_p,
            top_k=args.nmt_top_k,
            min_p=args.nmt_min_p,
            presence_penalty=args.nmt_presence_penalty,
            repetition_penalty=args.nmt_repetition_penalty,
            seed=args.nmt_seed,
            batch_size=args.nmt_batch_size,
            cache_enabled=args.nmt_cache,
            cache_dir=args.nmt_cache_dir,
            cache_run_id=cache_run_id,
            cache_max_chars=args.nmt_cache_max_chars,
        )

    stages = [
        TranslationManifestReader(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            target_lang_codes=args.target_langs,
            source_lang_key=args.source_lang_code_key,
            input_skip_key=args.skip_me_key,
            high_quality_key=args.high_quality_key,
            verify_done_line_counts=args.verify_done_line_counts,
        ),
        translate_stage,
        TranslationExpanderStage(
            source_lang_key=args.source_lang_code_key,
        ),
        # Bitext filters on the (source, translation) pair; all mark-only so the
        # writer's per-direction .done counting is preserved.
        *build_bitext_filter_stages(args),
        DirectionalShardedWriterStage(
            output_dir=args.output_dir,
            source_lang_key="source_lang",
            target_lang_key="target_lang",
        ),
    ]

    pipeline = Pipeline(name="translation_pipeline", stages=stages)
    logger.info("Pipeline:\n{}", pipeline.describe())

    # Multi-node Ray bootstrap. With --slurm the script is launched on every node
    # (srun --ntasks-per-node=1); SlurmRayClient elects the head from SLURM_NODEID,
    # while worker nodes join the cluster and block inside start() until teardown
    # (only the head returns here). XennaExecutor then connects via RAY_ADDRESS.
    # Without --slurm, XennaExecutor manages its own single-node Ray as before.
    ray_client = SlurmRayClient() if args.slurm else None
    if ray_client is not None:
        ray_client.start()

    t0 = time.time()
    try:
        if args.executor == "ray_data":
            # RayDataExecutor connects to the cluster SlurmRayClient bootstrapped
            # (via RAY_ADDRESS) and ignores execution_mode — Ray Data manages its
            # own streaming scheduling and backpressure.
            #
            # preserve_order=True makes outputs flow in input (manifest) order: each
            # manifest's rows reach the directional writer together, so its .done fires
            # before the next manifest's rows arrive. That gives incremental, durable
            # per-manifest progress under the Slurm time limit (a partial run leaves
            # earlier manifests .done instead of all-partial), without reloading the
            # vLLM engine per manifest. Bounded by backpressure; costs some reorder
            # freedom. (No effect under --executor xenna.)
            from ray.data import DataContext
            DataContext.get_current().execution_options.preserve_order = True
            executor = RayDataExecutor()
        else:
            executor = XennaExecutor(config={"execution_mode": args.execution_mode})
        logger.info("Using executor: {}", type(executor).__name__)
        pipeline.run(executor=executor)
        logger.info("Pipeline finished in {:.1f} min.", (time.time() - t0) / 60)
    finally:
        if ray_client is not None:
            ray_client.stop()

    # Driver-side cache merge: fold every worker's per-run dump (+ the previous canonical) into one
    # canonical file. Runs once on the driver/head, after all workers' teardown() dumps exist.
    # Never fails the run.
    if args.nmt_cache and args.nmt_cache_dir:
        try:
            n = merge_translation_cache(args.nmt_cache_dir, args.nmt_cache_max_entries)
            logger.info("NMT cache: merged to {} entries in {}", n, args.nmt_cache_dir)
        except Exception as exc:  # noqa: BLE001 - cache is best-effort
            logger.warning("NMT cache merge skipped: {}", exc)

    logger.info(
        "Done. Output files (*.jsonl.done) are in: {}",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
