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
runs SED and language ID on each segment, then writes output as a NeMo
tarred dataset (16kHz mono opus).

Pipeline:
    NeMoSpeechAudioReader (reads full audio from input_cfg)
        -> InferenceSortformerStage (speaker diarization on full audio) [optional]
        -> VADSegmentationStage (segments into speech chunks, fan-out)
        -> SEDInferenceStage (sound event detection on each segment)
        -> SEDPostprocessingStage (converts framewise probs to event labels)
        -> AmberNetLangIDStage (language identification per segment)
        -> NeMoSpeechWriterStage (encodes to opus at 16kHz + original SR)
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from loguru import logger

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.inference.ambernet_langid import AmberNetLangIDStage
from nemo_curator.stages.audio.inference.sed import SEDInferenceStage
from nemo_curator.stages.audio.inference.sortformer import InferenceSortformerStage
from nemo_curator.stages.audio.io.nemo_speech_reader import NeMoSpeechAudioReader
from nemo_curator.stages.audio.io.nemo_speech_writer import NeMoSpeechWriterStage
from nemo_curator.stages.audio.postprocessing.sed_postprocessing import SEDPostprocessingStage
from nemo_curator.stages.audio.segmentation import VADSegmentationStage
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class MonoDownsampleStage(ProcessingStage[AudioTask, AudioTask]):
    """Convert to mono and downsample to target sample rate. Runs once before VAD."""

    name: str = "MonoDownsample"
    target_sample_rate: int = 16000
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.waveform_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.waveform_key, self.sample_rate_key]

    def process(self, task: AudioTask) -> AudioTask:
        import numpy as np

        wav = task.data.get(self.waveform_key)
        sr = task.data.get(self.sample_rate_key, self.target_sample_rate)

        if wav is None:
            return task

        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=0)

        if sr != self.target_sample_rate:
            import librosa

            task.data["original_sampling_rate"] = sr
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.target_sample_rate)

        task.data[self.waveform_key] = wav
        task.data[self.sample_rate_key] = self.target_sample_rate
        task.data["sampling_rate"] = self.target_sample_rate
        return task


@dataclass
class SqueezeWaveformStage(ProcessingStage[AudioTask, AudioTask]):
    """Squeeze (1, N) waveforms to (N,) for downstream compatibility.

    VAD outputs waveform with shape (1, N) from .unsqueeze(0), but
    downstream stages (SED, LangID) expect 1D (N,) arrays.
    """

    name: str = "SqueezeWaveform"
    waveform_key: str = "waveform"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.waveform_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.waveform_key]

    def process(self, task: AudioTask) -> AudioTask:
        import numpy as np

        wav = task.data.get(self.waveform_key)
        if wav is not None:
            wav = np.asarray(wav)
            if wav.ndim > 1:
                wav = wav.squeeze()
            task.data[self.waveform_key] = wav
        return task


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Metadata extraction pipeline for unsegmented audio")

    ap.add_argument("--data_config", type=str, required=True, help="Path to input_cfg YAML.")
    ap.add_argument("--output_dir", type=str, required=True, help="Output directory for tarred dataset.")
    ap.add_argument("--corpus", type=str, default=None, help="Filter to specific corpus in the YAML.")

    vad = ap.add_argument_group("VAD (Silero)")
    vad.add_argument("--vad_threshold", type=float, default=0.5, help="VAD confidence threshold.")
    vad.add_argument("--min_duration_sec", type=float, default=2.0, help="Minimum segment duration (seconds).")
    vad.add_argument("--max_duration_sec", type=float, default=60.0, help="Maximum segment duration (seconds).")
    vad.add_argument("--speech_pad_ms", type=int, default=300, help="Padding before/after speech (ms).")

    sed = ap.add_argument_group("SED (Sound Event Detection)")
    sed.add_argument("--sed_checkpoint", type=str, default=None, help="Path to PANNs CNN14 checkpoint. Enables SED.")
    sed.add_argument("--sed_threshold", type=float, default=0.5, help="SED event confidence threshold.")
    sed.add_argument("--sed_batch_size", type=int, default=32, help="SED GPU batch size.")
    sed.add_argument("--sed_gpu_memory_gb", type=float, default=4.0, help="GPU memory for SED stage.")

    lid = ap.add_argument_group("Language ID (AmberNet)")
    lid.add_argument("--langid_model", type=str, default="langid_ambernet", help="NeMo LangID model name.")
    lid.add_argument("--langid_gpu_memory_gb", type=float, default=4.0, help="GPU memory for LangID stage.")
    lid.add_argument("--skip_langid", action="store_true", default=False, help="Skip language ID stage.")

    diar = ap.add_argument_group("Speaker Diarization (Sortformer)")
    diar.add_argument(
        "--sortformer_model", type=str, default=None,
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

    stages = [
        NeMoSpeechAudioReader(
            yaml_path=args.data_config,
            corpus_filter=args.corpus,
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
        stages.append(
            AmberNetLangIDStage(
                model_name=args.langid_model,
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
    if args.sortformer_model:
        logger.info(f"  Sortformer: {args.sortformer_model} (on full audio before VAD)")
    logger.info(f"  VAD: threshold={args.vad_threshold}, duration=[{args.min_duration_sec}, {args.max_duration_sec}]s")
    if args.sed_checkpoint:
        logger.info(f"  SED: enabled (checkpoint={args.sed_checkpoint})")
    if not args.skip_langid:
        logger.info(f"  LangID: {args.langid_model}")
    logger.info(f"  Output: individual opus files at {args.target_sample_rate}Hz + original SR")

    t0 = time.time()
    pipeline.run(executor=executor)
    elapsed = time.time() - t0
    logger.info(f"Pipeline finished in {elapsed / 60:.1f} min. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
