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

"""Qwen3-Omni-only pipeline for benchmark / reference-improvement datasets.

``--pipeline_mode full`` (default) runs:

    NeMoSpeechAudioReader
        → InitializeFieldsStage (text → granary_v1_prediction)
    InferenceQwenOmniStage
        → [optional] DisfluencyWerGuardStage
    WhisperHallucinationStage
    SelectBestPredictionStage
        → with optional reference fallback on hallucination
    [optional] TextLLMStage (entity replacement via Qwen3.5 text-only)
    RegexSubstitutionStage
    AbbreviationConcatStage
    ShardedManifestWriterStage

``--pipeline_mode omni_entity`` runs only:

    NeMoSpeechAudioReader → InitializeFieldsStage → InferenceQwenOmniStage
        → TextLLMStage (entity restoration) → ShardedManifestWriterStage

No SED, no recovery ASR.  Use ``--reference_text_key granary_v1_prediction``
with a prompt that contains ``{transcript}`` for reference-improvement runs.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from loguru import logger

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.alm.sharded_manifest_writer import ShardedManifestWriterStage
from nemo_curator.stages.audio.inference.qwen_omni import InferenceQwenOmniStage
from nemo_curator.stages.audio.io.nemo_speech_reader import NeMoSpeechAudioReader
from nemo_curator.stages.audio.text_filtering import (
    AbbreviationConcatStage,
    DisfluencyWerGuardStage,
    InitializeFieldsStage,
    RegexSubstitutionStage,
    WhisperHallucinationStage,
)
from nemo_curator.stages.audio.text_filtering.select_best_prediction import SelectBestPredictionStage
from nemo_curator.stages.audio.text_filtering.text_llm_stage import TextLLMStage


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Qwen3-Omni-only reference-improvement pipeline")
    ap.add_argument(
        "--pipeline_mode",
        type=str,
        choices=("full", "omni_entity"),
        default="full",
        help="full: hallucination filter + regex + optional entity stage. "
             "omni_entity: Omni inference + entity restoration only.",
    )
    ap.add_argument("--data_config", type=str, required=True, help="Granary YAML data config.")
    ap.add_argument("--corpus", type=str, nargs="*", default=None, help="Process only these corpora.")
    ap.add_argument("--output_dir", type=str, required=True, help="Output directory for per-shard manifests.")
    ap.add_argument("--language", type=str, default=None, help="ISO 639-1 language code filter.")
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument(
        "--ml_prompt",
        type=str,
        default="Transcribe the audio.",
        help="Multilingual prompt text. Supports {language} and {transcript} placeholders.",
    )
    ap.add_argument("--ml_prompt_file", type=str, default=None, help="Read multilingual prompt from file.")
    ap.add_argument(
        "--en_prompt_file",
        type=str,
        default=None,
        help="English-specific prompt file (e.g. reference-improvement prompt with {transcript}).",
    )
    ap.add_argument("--followup_prompt", type=str, default=None, help="Turn 2 follow-up prompt text.")
    ap.add_argument("--followup_prompt_file", type=str, default=None, help="Read Turn 2 follow-up prompt from file.")
    ap.add_argument("--system_prompt", type=str, default=None, help="System prompt text or path to file.")
    ap.add_argument(
        "--reference_text_key",
        type=str,
        default="granary_v1_prediction",
        help="Manifest key for the dataset reference transcript. "
             "Bound to {transcript} in prompts and used as hallucination fallback when enabled.",
    )
    ap.add_argument(
        "--use_reference_on_hallucination",
        action="store_true",
        help="When Omni output is flagged as hallucinated, keep the reference_text_key text instead.",
    )
    ap.add_argument("--tensor_parallel_size", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_output_tokens", type=int, default=256)
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--max_num_seqs", type=int, default=16)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    ap.add_argument("--prep_workers", type=int, default=16)
    ap.add_argument("--source_lang_key", type=str, default="source_lang")
    ap.add_argument("--primary_num_workers", type=int, default=None)

    tf = ap.add_argument_group("text filtering")
    tf.add_argument(
        "--hall_phrases",
        type=str,
        default=None,
        help="Path to hallucination phrases text file (required for --pipeline_mode full).",
    )
    tf.add_argument(
        "--regex_yaml",
        type=str,
        default=None,
        help="Path to regex substitution rules YAML (required for --pipeline_mode full).",
    )
    tf.add_argument("--unique_words_threshold", type=float, default=0.4)
    tf.add_argument("--long_word_threshold", type=int, default=25)
    tf.add_argument("--long_word_rel_threshold", type=float, default=3.0)
    tf.add_argument("--max_char_rate", type=float, default=40.0)

    er = ap.add_argument_group("entity replacement (Qwen3.5 text-only)")
    er.add_argument(
        "--enable_entity_replacement",
        action="store_true",
        help="Run a text-only LLM stage after Omni to swap named entities in the "
             "Omni output with entity forms from the ground-truth transcript.",
    )
    er.add_argument(
        "--entity_text_model_id",
        type=str,
        default="Qwen/Qwen3.5-35B-A3B-FP8",
        help="HuggingFace model ID for the entity-replacement text LLM.",
    )
    er.add_argument(
        "--entity_prompt_file",
        type=str,
        default=None,
        help="Prompt file for entity replacement. Defaults to bundled entity_replacement_prompt.md.",
    )
    er.add_argument(
        "--entity_input_text_key",
        type=str,
        default=None,
        help="Manifest key for the normalized Omni transcript. "
             "Defaults to primary_model_prediction for omni_entity mode, best_prediction otherwise.",
    )
    er.add_argument(
        "--entity_output_text_key",
        type=str,
        default="entity_corrected_text",
        help="Manifest key to write the entity-corrected transcript to.",
    )
    er.add_argument("--entity_tensor_parallel_size", type=int, default=1)
    er.add_argument("--entity_batch_size", type=int, default=64)
    er.add_argument("--entity_max_output_tokens", type=int, default=512)
    er.add_argument("--entity_max_model_len", type=int, default=4096)
    er.add_argument("--entity_max_num_seqs", type=int, default=32)
    er.add_argument("--entity_gpu_memory_utilization", type=float, default=0.90)
    er.add_argument("--entity_num_workers", type=int, default=None)
    return ap


def _load_prompt_file(path: str | None) -> str | None:
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def _validate_args(args: argparse.Namespace, followup_prompt: str | None) -> None:
    if args.pipeline_mode == "full":
        missing = [
            name
            for name, value in (("hall_phrases", args.hall_phrases), ("regex_yaml", args.regex_yaml))
            if not value
        ]
        if missing:
            msg = f"--pipeline_mode full requires: {', '.join(f'--{n}' for n in missing)}"
            raise SystemExit(msg)
    if args.pipeline_mode == "omni_entity" and followup_prompt:
        msg = "--pipeline_mode omni_entity does not support followup prompts"
        raise SystemExit(msg)


def _entity_input_text_key(args: argparse.Namespace, primary_text_key: str) -> str:
    if args.entity_input_text_key:
        return args.entity_input_text_key
    if args.pipeline_mode == "omni_entity":
        return primary_text_key
    return "best_prediction"


def _append_entity_stage(
    stages: list,
    args: argparse.Namespace,
    entity_input_key: str,
) -> None:
    _default_entity_prompt = (
        Path(__file__).resolve().parent / "prompts" / "entity_replacement_prompt.md"
    )
    entity_prompt_file = args.entity_prompt_file or str(_default_entity_prompt)
    stages.append(TextLLMStage(
        name="EntityReplacement",
        model_id=args.entity_text_model_id,
        prompt_file=entity_prompt_file,
        text_key=entity_input_key,
        reference_text_key=args.reference_text_key,
        output_text_key=args.entity_output_text_key,
        enable_validation=False,
        tensor_parallel_size=args.entity_tensor_parallel_size,
        batch_size=args.entity_batch_size,
        max_output_tokens=args.entity_max_output_tokens,
        max_model_len=args.entity_max_model_len,
        max_num_seqs=args.entity_max_num_seqs,
        gpu_memory_utilization=args.entity_gpu_memory_utilization,
        num_workers_override=args.entity_num_workers,
    ))


def _build_stages(
    args: argparse.Namespace,
    *,
    prompt: str,
    en_prompt: str | None,
    followup_prompt: str | None,
    system_prompt: str | None,
    language_filter: list[str] | None,
) -> list:
    primary_text_key = (
        "primary_model_prediction_s2"
        if followup_prompt
        else "primary_model_prediction"
    )

    stages = [
        NeMoSpeechAudioReader(
            yaml_path=args.data_config,
            corpus_filter=args.corpus,
            language_filter=language_filter,
            output_dir=args.output_dir,
        ),
        InitializeFieldsStage(
            pipeline_notes={
                "primary_model": "qwen_omni",
                "recovery_model": "none",
                "pipeline_mode": args.pipeline_mode,
            },
        ),
        InferenceQwenOmniStage(
            model_id=args.model_id,
            prompt_text=prompt,
            en_prompt_text=en_prompt,
            followup_prompt=followup_prompt,
            system_prompt=system_prompt,
            reference_text_key=args.reference_text_key,
            tensor_parallel_size=args.tensor_parallel_size,
            batch_size=args.batch_size,
            max_output_tokens=args.max_output_tokens,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            prep_workers=args.prep_workers,
            source_lang_key=args.source_lang_key,
            pred_text_key="primary_model_prediction",
            disfluency_text_key="primary_model_prediction_s2",
            num_workers_override=args.primary_num_workers,
        ),
    ]

    if args.pipeline_mode == "omni_entity":
        entity_input_key = _entity_input_text_key(args, primary_text_key)
        _append_entity_stage(stages, args, entity_input_key)
        stages.append(ShardedManifestWriterStage(output_dir=args.output_dir))
        return stages

    if followup_prompt:
        stages.append(DisfluencyWerGuardStage(
            ref_text_key="primary_model_prediction",
            hyp_text_key="primary_model_prediction_s2",
            max_wer_pct=50.0,
        ))

    stages.append(WhisperHallucinationStage(
        name="WhisperHallucination_primary",
        common_hall_file=args.hall_phrases,
        text_key=primary_text_key,
        language_key=args.source_lang_key,
        unique_words_threshold=args.unique_words_threshold,
        long_word_threshold=args.long_word_threshold,
        long_word_rel_threshold=args.long_word_rel_threshold,
        max_char_rate=args.max_char_rate,
    ))

    stages.append(SelectBestPredictionStage(
        primary_text_key=primary_text_key,
        reference_text_key=args.reference_text_key,
        use_reference_on_hallucination=args.use_reference_on_hallucination,
        primary_source_label="primary",
    ))

    regex_input_key = _entity_input_text_key(args, primary_text_key)
    if args.enable_entity_replacement:
        _append_entity_stage(stages, args, regex_input_key)
        regex_input_key = args.entity_output_text_key

    stages.extend([
        RegexSubstitutionStage(
            regex_params_yaml=args.regex_yaml,
            text_key=regex_input_key,
            output_text_key="cleaned_text",
        ),
        AbbreviationConcatStage(
            text_key="cleaned_text",
            output_text_key="abbreviated_text",
            source_lang_key=args.source_lang_key,
        ),
    ])

    stages.append(ShardedManifestWriterStage(output_dir=args.output_dir))
    return stages


def main() -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

    args = _build_arg_parser().parse_args()

    prompt = args.ml_prompt
    if args.ml_prompt_file:
        prompt = _load_prompt_file(args.ml_prompt_file) or prompt

    en_prompt = _load_prompt_file(args.en_prompt_file)
    followup_prompt = args.followup_prompt
    if args.followup_prompt_file:
        followup_prompt = _load_prompt_file(args.followup_prompt_file)

    system_prompt = None
    if args.system_prompt:
        if os.path.isfile(args.system_prompt):
            system_prompt = _load_prompt_file(args.system_prompt)
        else:
            system_prompt = args.system_prompt

    _validate_args(args, followup_prompt)

    language_filter = [args.language.lower().strip()] if args.language else None
    stages = _build_stages(
        args,
        prompt=prompt,
        en_prompt=en_prompt,
        followup_prompt=followup_prompt,
        system_prompt=system_prompt,
        language_filter=language_filter,
    )

    pipeline_name = (
        "qwen_omni_entity_pipeline"
        if args.pipeline_mode == "omni_entity"
        else "qwen_omni_reference_pipeline"
    )
    pipeline = Pipeline(name=pipeline_name, stages=stages)
    logger.info(f"Pipeline: {pipeline.describe()}")
    entity_enabled = (
        args.pipeline_mode == "omni_entity" or args.enable_entity_replacement
    )
    logger.info(
        "pipeline_mode={}, language_filter={}, reference_text_key={}, "
        "use_reference_on_hallucination={}, followup={}, entity_replacement={}",
        args.pipeline_mode,
        language_filter,
        args.reference_text_key,
        args.use_reference_on_hallucination,
        bool(followup_prompt),
        entity_enabled,
    )

    t0 = time.time()
    pipeline.run(executor=RayDataExecutor())
    logger.info(f"Pipeline finished in {(time.time() - t0) / 60:.1f} min. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
