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

Pipeline:
    NeMoSpeechAudioReader (reads full audio from input_cfg)
        -> MonoDownsampleStage (mono + resample, stores original SR/channels)
        -> InferenceSortformerStage (speaker diarization on full audio) [optional]
        -> VADSegmentationStage (segments into speech chunks, fan-out)
        -> SqueezeWaveformStage (flatten VAD output shape)
        -> SEDInferenceStage (sound event detection on each segment) [optional]
        -> SEDPostprocessingStage (converts framewise probs to event labels) [optional]
        -> LangID: AmberNet (NeMo, 20 langs) or SpeechBrain VoxLingua107 (107 langs)
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
from nemo_curator.stages.resources import Resources


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Metadata extraction pipeline for unsegmented audio")

    ap.add_argument("--data_config", type=str, required=True, help="Path to input_cfg YAML.")
    ap.add_argument("--output_dir", type=str, required=True, help="Output directory for opus files + manifest.")
    ap.add_argument("--corpus", type=str, default=None, help="Filter to specific corpus in the YAML.")
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Filter to specific language(s) in the YAML (comma-separated, e.g. 'en,de').",
    )

    vad = ap.add_argument_group("VAD (Silero)")
    vad.add_argument(
        "--vad_threshold", type=float, default=0.5,
        help="VAD confidence threshold (0.5 is Silero's recommended default).",
    )
    vad.add_argument(
        "--min_duration_sec", type=float, default=0.5,
        help="Minimum segment duration (seconds). Segments shorter than this are discarded.",
    )
    vad.add_argument(
        "--max_duration_sec", type=float, default=40.0,
        help="Maximum segment duration (seconds). Longer speech regions are split.",
    )
    vad.add_argument(
        "--speech_pad_ms", type=int, default=300,
        help="Silero VAD internal padding (ms) — extends detected speech boundaries to avoid cutting onsets/offsets.",
    )

    sed = ap.add_argument_group("SED (Sound Event Detection)")
    sed.add_argument("--sed_checkpoint", type=str, default=None, help="Path to PANNs CNN14 checkpoint. Enables SED.")
    sed.add_argument("--sed_threshold", type=float, default=0.5, help="SED event confidence threshold.")
    sed.add_argument("--sed_batch_size", type=int, default=32, help="SED GPU batch size.")
    sed.add_argument("--sed_gpu_memory_gb", type=float, default=4.0, help="GPU memory for SED stage.")

    lid = ap.add_argument_group("Language ID")
    lid.add_argument(
        "--langid_backend", type=str, default="ambernet", choices=["ambernet", "speechbrain"],
        help="LangID backend: 'ambernet' (NeMo, 20 languages) or 'speechbrain' (VoxLingua107, 107 languages).",
    )
    lid.add_argument("--langid_model", type=str, default=None, help="Model name/path (default depends on backend).")
    lid.add_argument("--langid_gpu_memory_gb", type=float, default=4.0, help="GPU memory for LangID stage.")
    lid.add_argument("--skip_langid", action="store_true", default=False, help="Skip language ID stage.")

    diar = ap.add_argument_group("Speaker Diarization (Sortformer)")
    diar.add_argument(
        "--sortformer_model",
        type=str,
        default=None,
        help="HuggingFace model id or local .nemo path. Enables Sortformer diarization on full audio.",
    )
    diar.add_argument("--sortformer_gpu_memory_gb", type=float, default=8.0, help="GPU memory for Sortformer stage.")
    diar.add_argument("--sortformer_batch_size", type=int, default=1, help="Sortformer inference batch size.")
    diar.add_argument("--rttm_out_dir", type=str, default=None, help="Directory to write RTTM files.")

    out = ap.add_argument_group("Output")
    out.add_argument("--target_sample_rate", type=int, default=16000, help="Output sample rate.")

    return ap


def main() -> None:
    args = _build_arg_parser().parse_args()

    language_filter = [lang.strip() for lang in args.language.split(",")] if args.language else None
    corpus_filter = [args.corpus] if args.corpus else None

    stages = [
        NeMoSpeechAudioReader(
            yaml_path=args.data_config,
            corpus_filter=corpus_filter,
            language_filter=language_filter,
            output_dir=args.output_dir,
        ),
        MonoDownsampleStage(target_sample_rate=args.target_sample_rate),
    ]

    if args.sortformer_model:
        model_path = args.sortformer_model if args.sortformer_model.endswith(".nemo") else None
        model_name = args.sortformer_model if model_path is None else "nvidia/diar_streaming_sortformer_4spk-v2"
        stages.append(
            InferenceSortformerStage(
                model_name=model_name,
                model_path=model_path,
                inference_batch_size=args.sortformer_batch_size,
                rttm_out_dir=args.rttm_out_dir,
                resources=Resources(gpu_memory_gb=args.sortformer_gpu_memory_gb),
            )
        )

    stages.append(
        VADSegmentationStage(
            threshold=args.vad_threshold,
            min_duration_sec=args.min_duration_sec,
            max_duration_sec=args.max_duration_sec,
            speech_pad_ms=args.speech_pad_ms,
            nested=False,
        )
    )
    stages.append(SqueezeWaveformStage())

    if args.sed_checkpoint:
        stages.append(
            SEDInferenceStage(
                checkpoint_path=args.sed_checkpoint,
                batch_size=args.sed_batch_size,
                resources=Resources(gpu_memory_gb=args.sed_gpu_memory_gb),
            )
        )
        stages.append(
            SEDPostprocessingStage(
                threshold=args.sed_threshold,
            )
        )

    if not args.skip_langid:
        if args.langid_backend == "speechbrain":
            from nemo_curator.stages.audio.inference.speechbrain_langid import SpeechBrainLangIDStage

            langid_source = args.langid_model or "speechbrain/lang-id-voxlingua107-ecapa"
            stages.append(
                SpeechBrainLangIDStage(
                    source=langid_source,
                    resources=Resources(gpu_memory_gb=args.langid_gpu_memory_gb),
                )
            )
        else:
            langid_model = args.langid_model or "langid_ambernet"
            stages.append(
                AmberNetLangIDStage(
                    model_name=langid_model,
                    resources=Resources(gpu_memory_gb=args.langid_gpu_memory_gb),
                )
            )

    stages.append(
        NeMoSpeechWriterStage(
            output_dir=args.output_dir,
            target_sample_rate=args.target_sample_rate,
        )
    )

    pipeline = Pipeline(name="metadata_extraction", stages=stages)

    from nemo_curator.backends.ray_data import RayDataExecutor

    executor = RayDataExecutor()

    logger.info(f"Metadata extraction pipeline: {len(stages)} stages")
    logger.info(f"  Input: {args.data_config}")
    logger.info(f"  Output: {args.output_dir}")
    if language_filter:
        logger.info(f"  Language filter: {language_filter}")
    if args.sortformer_model:
        logger.info(f"  Sortformer: {args.sortformer_model} (on full audio before VAD)")
    logger.info(f"  VAD: threshold={args.vad_threshold}, duration=[{args.min_duration_sec}, {args.max_duration_sec}]s")
    if args.sed_checkpoint:
        logger.info(f"  SED: enabled (checkpoint={args.sed_checkpoint})")
    if not args.skip_langid:
        langid_desc = args.langid_model or ("speechbrain/lang-id-voxlingua107-ecapa" if args.langid_backend == "speechbrain" else "langid_ambernet")
        logger.info(f"  LangID: {args.langid_backend} ({langid_desc})")
    logger.info(f"  Output: individual opus files at {args.target_sample_rate}Hz")

    t0 = time.time()
    pipeline.run(executor=executor)
    elapsed = time.time() - t0
    logger.info(f"Pipeline finished in {elapsed / 60:.1f} min. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
