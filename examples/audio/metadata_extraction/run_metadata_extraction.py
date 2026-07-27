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

"""Metadata extraction pipeline for unsegmented audio.

Reads long unsegmented audio from NeMo input_cfg YAML, optionally runs
speaker diarization (Sortformer) on the full audio, segments with Silero VAD,
runs SED and language ID on each segment, then writes output as opus files
with a NeMo-compatible JSONL manifest (16kHz mono).

``input_cfg`` YAML notes
------------------------
- Tarred data: use ``tarred_audio_filepaths`` (not ``audio_filepaths``) with
  ``type: nemo_tarred``.
- ``shard_key_prefix``: optional; sets output/checkpoint layout when ``corpus``
  is a catalog label or when the same dataset folder appears under multiple
  locales. Preferred layout: ``<catalog>/<locale>/<dataset-id>/...``. See
  ``nemo_curator.stages.audio.io.shard_key`` module docstring for examples.

Pipeline:
    NeMoSpeechAudioReader (reads full audio from input_cfg)
        -> MonoDownsampleStage (mono + resample, stores original SR/channels)
        -> InferenceSortformerStage (speaker diarization on full audio) [optional]
        -> VADSegmentationStage (segments into speech chunks, fan-out)
        -> SqueezeWaveformStage (flatten VAD output shape)
        -> SEDInferenceStage (sound event detection on each segment) [optional]
        -> SEDPostprocessingStage (converts framewise probs to event labels) [optional]
        -> LangID: AmberNet (NeMo, 20 langs) or SpeechBrain VoxLingua107 (107 langs);
             with --indic, a second Indic Canary pass + dual-agreement selection
        -> NeMoSpeechWriterStage (encodes to opus at 16kHz)
"""

from __future__ import annotations

import argparse
import time

from loguru import logger

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.inference.ambernet_langid import AmberNetLangIDStage
from nemo_curator.stages.audio.inference.sed import SEDInferenceStage
from nemo_curator.stages.audio.inference.sortformer import InferenceSortformerStage
from nemo_curator.stages.audio.io.nemo_speech_reader import NeMoSpeechAudioReader
from nemo_curator.stages.audio.io.nemo_speech_writer import NeMoSpeechWriterStage
from nemo_curator.stages.audio.postprocessing.sed_postprocessing import SEDPostprocessingStage
from nemo_curator.stages.audio.preprocessing import MonoDownsampleStage, SqueezeWaveformStage
from nemo_curator.stages.audio.segmentation import VADSegmentationStage
from nemo_curator.stages.audio.text_filtering.select_best_lid_prediction import SelectBestLIDPredictionStage
from nemo_curator.stages.resources import Resources

_SORTFORMER_BATCH_WINDOW_MULTIPLIER = 4


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Metadata extraction pipeline for unsegmented audio")

    ap.add_argument("--data_config", type=str, default=None, help="Path to input_cfg YAML.")
    ap.add_argument("--output_dir", type=str, required=True, help="Output directory for opus files + manifest.")
    ap.add_argument("--corpus", type=str, default=None, help="Filter to specific corpus in the YAML.")
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Filter to specific language(s) in the YAML (comma-separated, e.g. 'en,de').",
    )
    ap.add_argument(
        "--indic",
        action="store_true",
        default=False,
        help="Enable two-pass Indic LID: primary LID + Indic Canary secondary with dual-agreement selection.",
    )
    ap.add_argument(
        "--indic_canary_engine_dir",
        type=str,
        default=None,
        help="Path to prebuilt Indic Canary TRT-LLM engine directory. Required when --indic is set.",
    )
    ap.add_argument(
        "--indic_canary_kv_cache_free_gpu_memory_fraction",
        type=float,
        default=0.2,
        help="Fraction of free GPU memory the Indic Canary TRT-LLM decoder may claim for KV cache "
        "(default 0.2; raise toward 0.9 only when Canary owns the GPU).",
    )
    ap.add_argument(
        "--indic_canary_cross_kv_cache_fraction",
        type=float,
        default=0.2,
        help="Fraction of the KV-cache budget reserved for cross-attention "
        "(default 0.2; keep low when Canary shares a GPU).",
    )
    ap.add_argument(
        "--resampled_output_dir",
        type=str,
        default=None,
        help="Directory to write resampled 16kHz mono WAV files. The output filename matches the input stem with a .wav extension.",
    )

    vad = ap.add_argument_group("VAD (Silero)")
    vad.add_argument(
        "--vad_threshold",
        type=float,
        default=0.5,
        help="VAD confidence threshold (0.5 is Silero's recommended default).",
    )
    vad.add_argument(
        "--min_duration_sec",
        type=float,
        default=0.5,
        help="Minimum segment duration (seconds). Segments shorter than this are discarded.",
    )
    vad.add_argument(
        "--max_duration_sec",
        type=float,
        default=40.0,
        help="Maximum segment duration (seconds). Longer speech regions are split.",
    )
    vad.add_argument(
        "--speech_pad_ms",
        type=int,
        default=100,
        help="Silero VAD internal padding (ms) — extends detected speech boundaries to avoid cutting onsets/offsets.",
    )
    vad.add_argument(
        "--min_interval_ms",
        type=int,
        default=500,
        help="Minimum silence gap (ms) between speech segments — higher values merge more, reducing short segments.",
    )
    vad.add_argument(
        "--vad_backend",
        choices=["torch", "onnx", "tensorrt"],
        default="torch",
        help="Silero inference backend.",
    )
    vad.add_argument(
        "--vad_tensorrt_engine",
        type=str,
        default=None,
        help="TensorRT engine path; required when --vad_backend=tensorrt.",
    )
    vad.add_argument(
        "--vad_batch_size",
        type=int,
        default=1,
        help="Recordings per VAD call; use a TensorRT engine profile that supports this value.",
    )
    vad.add_argument(
        "--vad_gpu_memory_gb",
        type=float,
        default=2.0,
        help="GPU memory in GB for each TensorRT VAD worker.",
    )

    sed = ap.add_argument_group("SED (Sound Event Detection)")
    sed.add_argument("--sed_checkpoint", type=str, default=None, help="Path to PANNs CNN14 checkpoint. Enables SED.")
    sed.add_argument(
        "--sed_backend",
        choices=["torch", "tensorrt"],
        default="torch",
        help="CNN14 inference backend.",
    )
    sed.add_argument(
        "--sed_tensorrt_engine",
        type=str,
        default=None,
        help="TensorRT engine path; required when --sed_backend=tensorrt.",
    )
    sed.add_argument("--sed_threshold", type=float, default=0.5, help="SED event confidence threshold.")
    sed.add_argument(
        "--sed_batch_size",
        type=int,
        default=32,
        help="SED GPU batch size; the TensorRT engine profile must support this value.",
    )
    sed.add_argument("--sed_gpu_memory_gb", type=float, default=4.0, help="GPU memory for SED stage.")
    sed.add_argument(
        "--sed_emit_superclasses",
        type=lambda x: x.lower() not in ("false", "0", "no"),
        default=True,
        help="Emit superclass labels only — speech/music/noise (default: True). Set to False for all 527 AudioSet classes.",
    )

    lid = ap.add_argument_group("Language ID")
    lid.add_argument(
        "--langid_backend",
        type=str,
        default="speechbrain",
        choices=["ambernet", "speechbrain"],
        help="LangID backend: 'speechbrain' (VoxLingua107, 107 languages, default) or 'ambernet' (NeMo, 20 languages).",
    )
    lid.add_argument("--langid_model", type=str, default=None, help="Model name/path (default depends on backend).")
    lid.add_argument("--langid_gpu_memory_gb", type=float, default=4.0, help="GPU memory for LangID stage.")
    lid.add_argument(
        "--langid_max_workers",
        type=int,
        default=2,
        help="Hard cap on concurrent LangID actors per GPU (0/negative = executor autoscales). "
        "Default 2: prevents the autoscaler from packing ~10 actors on one GPU (a common "
        "SpeechBrain OOM cause when co-resident with other GPU stages).",
    )
    lid.add_argument("--langid_batch_size", type=int, default=16, help="LangID inference batch size.")
    lid.add_argument("--skip_langid", action="store_true", default=False, help="Skip language ID stage.")

    diar = ap.add_argument_group("Speaker Diarization (Sortformer)")
    diar.add_argument(
        "--sortformer_model",
        type=str,
        default=None,
        help="HuggingFace model id or local .nemo path. Enables Sortformer diarization on full audio.",
    )
    diar.add_argument("--sortformer_gpu_memory_gb", type=float, default=8.0, help="GPU memory for Sortformer stage.")
    diar.add_argument(
        "--sortformer_gpus",
        type=float,
        default=None,
        help="GPUs per Sortformer actor (e.g. 1.0 for one full GPU). Overrides sortformer_gpu_memory_gb.",
    )
    diar.add_argument("--sortformer_batch_size", type=int, default=1, help="Sortformer inference batch size.")
    diar.add_argument(
        "--sortformer_backend",
        choices=["nemo", "tensorrt"],
        default="nemo",
        help="Sortformer inference backend.",
    )
    diar.add_argument("--sortformer_tensorrt_engine", type=str, default=None, help="Sortformer TensorRT plan.")
    diar.add_argument("--sortformer_tensorrt_config", type=str, default=None, help="Sortformer TensorRT runtime JSON.")
    diar.add_argument(
        "--sortformer_tensorrt_runtime_module",
        type=str,
        default=None,
        help="Matching Riva sortformer_modules.py.",
    )
    diar.add_argument(
        "--sortformer_precision",
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Sortformer inference precision.",
    )
    diar.add_argument(
        "--sortformer_compile_encoder",
        action="store_true",
        help="Compile the Sortformer encoder with torch.compile.",
    )
    diar.add_argument("--rttm_out_dir", type=str, default=None, help="Directory to write RTTM files.")

    io = ap.add_argument_group("I/O")
    io.add_argument(
        "--max_io_threads",
        type=int,
        default=8,
        help="Max concurrent threads per reader batch for loading audio from S3/object storage (default: 8).",
    )
    io.add_argument(
        "--read_concurrency",
        type=int,
        default=2,
        help="Max parallel Ray reader tasks (default: 2). Increase to overlap more S3/AIS reads.",
    )
    io.add_argument(
        "--writer_concurrency",
        type=int,
        default=1,
        help="Parallel Ray writer actors for opus + manifest output (default: 1).",
    )

    out = ap.add_argument_group("Output")
    out.add_argument("--target_sample_rate", type=int, default=16000, help="Output sample rate.")

    ex = ap.add_argument_group("Executor")
    ex.add_argument(
        "--executor",
        choices=["ray_data", "xenna"],
        default="ray_data",
        help="Backend executor. 'xenna' (Cosmos-Xenna) supports batch mode where stages run "
        "sequentially so GPU stages never co-reside (avoids single-GPU OOM/contention).",
    )
    ex.add_argument(
        "--execution_mode",
        choices=["streaming", "batch"],
        default="streaming",
        help="Xenna execution mode: 'batch' materializes each stage before the next (one GPU "
        "stage resident at a time); 'streaming' pipelines them. Only used with --executor xenna.",
    )

    return ap


def _build_stages(args: argparse.Namespace, language_filter: list[str] | None) -> list:
    corpus_filter = [args.corpus] if args.corpus else None
    vad_resources = Resources(cpus=1.0)
    if args.vad_backend == "tensorrt":
        vad_resources = Resources(cpus=1.0, gpu_memory_gb=args.vad_gpu_memory_gb)

    stages = [
        NeMoSpeechAudioReader(
            yaml_path=args.data_config,
            corpus_filter=corpus_filter,
            language_filter=language_filter,
            output_dir=args.output_dir,
            max_io_threads=args.max_io_threads,
            read_concurrency=args.read_concurrency,
            resampled_output_dir=args.resampled_output_dir,
            keep_waveform=not args.resampled_output_dir,
        ),
    ]

    if not args.resampled_output_dir:
        stages.append(MonoDownsampleStage(target_sample_rate=args.target_sample_rate))

    if args.sortformer_model or args.sortformer_tensorrt_engine:
        model_path = (
            args.sortformer_model if args.sortformer_model and args.sortformer_model.endswith(".nemo") else None
        )
        model_name = args.sortformer_model or "nvidia/diar_streaming_sortformer_4spk-v2"
        if args.sortformer_gpus is not None:
            sortformer_resources = Resources(gpus=args.sortformer_gpus)
        else:
            sortformer_resources = Resources(gpu_memory_gb=args.sortformer_gpu_memory_gb)
        stages.append(
            InferenceSortformerStage(
                model_name=model_name,
                model_path=model_path,
                inference_batch_size=args.sortformer_batch_size,
                batch_size=args.sortformer_batch_size * _SORTFORMER_BATCH_WINDOW_MULTIPLIER,
                backend=args.sortformer_backend,
                tensorrt_engine_path=args.sortformer_tensorrt_engine,
                tensorrt_config_path=args.sortformer_tensorrt_config,
                tensorrt_runtime_module_path=args.sortformer_tensorrt_runtime_module,
                precision=args.sortformer_precision,
                compile_encoder=args.sortformer_compile_encoder,
                rttm_out_dir=args.rttm_out_dir,
                resources=sortformer_resources,
                filepath_key="resampled_audio_filepath" if args.resampled_output_dir else "audio_filepath",
            )
        )

    stages.append(
        VADSegmentationStage(
            threshold=args.vad_threshold,
            min_interval_ms=args.min_interval_ms,
            min_duration_sec=args.min_duration_sec,
            max_duration_sec=args.max_duration_sec,
            speech_pad_ms=args.speech_pad_ms,
            backend=args.vad_backend,
            tensorrt_engine_path=args.vad_tensorrt_engine,
            batch_size=args.vad_batch_size,
            nested=False,
            filepath_key="resampled_audio_filepath" if args.resampled_output_dir else "audio_filepath",
            resources=vad_resources,
        )
    )
    stages.append(SqueezeWaveformStage())

    if args.sed_checkpoint:
        stages.append(
            SEDInferenceStage(
                checkpoint_path=args.sed_checkpoint,
                backend=args.sed_backend,
                tensorrt_engine_path=args.sed_tensorrt_engine,
                batch_size=args.sed_batch_size,
                resources=Resources(gpu_memory_gb=args.sed_gpu_memory_gb),
            )
        )
        stages.append(
            SEDPostprocessingStage(
                threshold=args.sed_threshold,
                emit_superclasses=args.sed_emit_superclasses,
            )
        )

    if not args.skip_langid:
        langid_max_workers = args.langid_max_workers if args.langid_max_workers > 0 else None
        if args.langid_backend == "speechbrain":
            from nemo_curator.stages.audio.inference.speechbrain_langid import SpeechBrainLangIDStage

            langid_source = args.langid_model or "speechbrain/lang-id-voxlingua107-ecapa"
            stages.append(
                SpeechBrainLangIDStage(
                    source=langid_source,
                    tag="primary",
                    batch_size=args.langid_batch_size,
                    max_workers=langid_max_workers,
                    resources=Resources(gpu_memory_gb=args.langid_gpu_memory_gb),
                )
            )
        else:
            langid_model = args.langid_model or "langid_ambernet"
            stages.append(
                AmberNetLangIDStage(
                    model_name=langid_model,
                    tag="primary",
                    batch_size=args.langid_batch_size,
                    max_workers=langid_max_workers,
                    resources=Resources(gpu_memory_gb=args.langid_gpu_memory_gb),
                )
            )

        if args.indic:
            from nemo_curator.stages.audio.inference.indic_canary_lid import IndicCanaryLangIDStage

            if not args.indic_canary_engine_dir:
                msg = "--indic_canary_engine_dir is required when --indic is set"
                raise ValueError(msg)
            stages.append(
                IndicCanaryLangIDStage(
                    engine_dir=args.indic_canary_engine_dir,
                    tag="secondary",
                    kv_cache_free_gpu_memory_fraction=args.indic_canary_kv_cache_free_gpu_memory_fraction,
                    cross_kv_cache_fraction=args.indic_canary_cross_kv_cache_fraction,
                    resources=Resources(gpu_memory_gb=args.langid_gpu_memory_gb),
                )
            )
        stages.append(SelectBestLIDPredictionStage())

    stages.append(
        NeMoSpeechWriterStage(
            output_dir=args.output_dir,
            target_sample_rate=args.target_sample_rate,
            writer_concurrency=args.writer_concurrency,
        )
    )
    return stages


def main() -> None:
    args = _build_arg_parser().parse_args()

    if not args.data_config:
        msg = "--data_config is required"
        raise SystemExit(msg)

    language_filter = [lang.strip() for lang in args.language.split(",")] if args.language else None
    stages = _build_stages(args, language_filter)
    pipeline_name = "metadata_extraction"

    pipeline = Pipeline(name=pipeline_name, stages=stages)

    if args.executor == "xenna":
        from nemo_curator.backends.xenna import XennaExecutor

        executor = XennaExecutor(config={"execution_mode": args.execution_mode})
        logger.info(f"Executor: XennaExecutor (execution_mode={args.execution_mode})")
    else:
        from nemo_curator.backends.ray_data import RayDataExecutor

        executor = RayDataExecutor()
        logger.info("Executor: RayDataExecutor (streaming)")

    logger.info(f"Metadata extraction pipeline: {len(stages)} stages ({pipeline_name})")
    logger.info(f"  Output: {args.output_dir}")
    logger.info(f"  Input: {args.data_config}")
    if language_filter:
        logger.info(f"  Language filter: {language_filter}")
    if args.sortformer_model or args.sortformer_tensorrt_engine:
        sf_desc = (
            f"gpus={args.sortformer_gpus}/actor"
            if args.sortformer_gpus is not None
            else f"gpu_memory_gb={args.sortformer_gpu_memory_gb}"
        )
        model = args.sortformer_tensorrt_engine if args.sortformer_backend == "tensorrt" else args.sortformer_model
        logger.info(f"  Sortformer: {model} ({sf_desc}, on full audio before VAD)")
    logger.info(
        f"  VAD: backend={args.vad_backend}, threshold={args.vad_threshold}, "
        f"min_interval_ms={args.min_interval_ms}, "
        f"speech_pad_ms={args.speech_pad_ms}, duration=[{args.min_duration_sec}, {args.max_duration_sec}]s"
    )
    if args.vad_backend == "tensorrt":
        logger.info(f"  VAD GPU memory: {args.vad_gpu_memory_gb} GB/worker")
    if args.sed_checkpoint:
        logger.info(
            f"  SED: backend={args.sed_backend}, checkpoint={args.sed_checkpoint}, "
            f"batch_size={args.sed_batch_size}"
        )
        if args.sed_backend == "tensorrt":
            logger.info(f"  SED TensorRT engine: {args.sed_tensorrt_engine}")
    if not args.skip_langid:
        langid_desc = args.langid_model or (
            "speechbrain/lang-id-voxlingua107-ecapa" if args.langid_backend == "speechbrain" else "langid_ambernet"
        )
        if args.indic:
            logger.info(
                f"  LangID: two-pass Indic mode — primary={args.langid_backend} ({langid_desc})"
                f" + Indic Canary ({args.indic_canary_engine_dir}) -> SelectBestLIDPrediction"
            )
        else:
            logger.info(f"  LangID: {args.langid_backend} ({langid_desc})")
    logger.info(f"  Target sample rate: {args.target_sample_rate}Hz, writer_concurrency={args.writer_concurrency}")

    t0 = time.time()
    pipeline.run(executor=executor)
    elapsed = time.time() - t0
    logger.info(f"Pipeline finished in {elapsed / 60:.1f} min. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
