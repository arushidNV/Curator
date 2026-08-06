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

"""Reader -> Writer opus backfill pipeline.

Minimal two-stage variant of ``run_metadata_extraction.py`` that re-emits the
per-segment ``.opus`` clips (and an opus-backed JSONL manifest) for datasets that
were previously processed with ``--no_save_audio`` (manifest only). It does NOT
re-run any GPU stage (Sortformer / VAD / SED / LangID): the segments and all
metadata are taken verbatim from the previous run's per-segment manifest, which
is fed back in as a segment-level ``type: nemo`` input (each row carries
``offset`` / ``duration`` + the source ``original_audio_filepath``).

Pipeline:
    NeMoSpeechAudioReader   (reads the 16 kHz WAV, slices [offset, offset+dur],
                             keeps the waveform in memory)
        -> SegmentOffsetTagStage  (restores start_ms/end_ms + original_file from
                                    the manifest row so the writer reproduces the
                                    original ``<stem>_<offset_ms>ms.opus`` naming
                                    and the offset/original_end manifest fields)
        -> NeMoSpeechWriterStage  (encodes opus at 16 kHz + writes the manifest)

Because ``resampled_output_dir`` is intentionally NOT set, the reader keeps
``keep_waveform=True`` so the writer receives audio to encode. Resample-if-missing
is handled upstream (see ``build_rw_manifest.py``), so the reader input always
points at a 16 kHz WAV.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

from loguru import logger

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.io.nemo_speech_reader import NeMoSpeechAudioReader
from nemo_curator.stages.audio.io.nemo_speech_writer import NeMoSpeechWriterStage
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_TARGET_SR = 16000


@dataclass
class SegmentOffsetTagStage(ProcessingStage[AudioTask, AudioTask]):
    """Restore the per-segment writer fields from a pre-segmented manifest row.

    The stock writer names each clip ``<stem>_<offset_ms>ms.opus`` from
    ``start_ms`` and only records ``offset`` / ``original_end`` when
    ``start_ms`` / ``end_ms`` are present — normally set by the VAD stage. When
    reading a previous run's per-segment manifest directly (no VAD in this
    pipeline), that information lives on the ``offset`` / ``duration`` fields of
    the input row, so we translate it here:

    - ``start_ms = round(offset * 1000)`` and ``end_ms = round((offset + duration) * 1000)``
      so the opus filename + ``offset`` / ``original_end`` match the first run.
    - ``original_file = original_audio_filepath`` so the writer derives the clip
      stem (and records ``original_audio_filepath``) from the true source, not
      from the intermediate WAV path the reader loaded.
    """

    name: str = "SegmentOffsetTag"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: AudioTask) -> AudioTask:
        data = task.data
        offset = data.get("offset")
        duration = data.get("duration")
        if offset is not None:
            data["start_ms"] = int(round(float(offset) * 1000))
            if duration is not None:
                data["end_ms"] = int(round((float(offset) + float(duration)) * 1000))
        original = data.get("original_audio_filepath")
        if original:
            data["original_file"] = original
        return task

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        return [self.process(task) for task in tasks]


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Reader->Writer opus backfill (no GPU stages)")
    ap.add_argument("--data_config", type=str, required=True, help="Path to input_cfg YAML (segment-level type: nemo).")
    ap.add_argument("--output_dir", type=str, required=True, help="Output root for opus files + manifest copy.")
    ap.add_argument("--corpus", type=str, default=None, help="Filter to a specific corpus in the YAML.")
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Comma-separated language filter matching the YAML 'language' field(s).",
    )
    ap.add_argument("--target_sample_rate", type=int, default=_TARGET_SR, help="Output/opus sample rate.")
    ap.add_argument(
        "--max_io_threads",
        type=int,
        default=8,
        help="Max concurrent threads per reader batch for loading WAVs.",
    )
    ap.add_argument("--read_concurrency", type=int, default=4, help="Max parallel Ray reader tasks.")
    ap.add_argument("--writer_concurrency", type=int, default=8, help="Parallel Ray writer actors for opus output.")
    ap.add_argument(
        "--executor",
        choices=["ray_data", "xenna"],
        default="ray_data",
        help="Backend executor.",
    )
    ap.add_argument(
        "--execution_mode",
        choices=["streaming", "batch"],
        default="streaming",
        help="Xenna execution mode (only used with --executor xenna).",
    )
    return ap


def _build_stages(args: argparse.Namespace, language_filter: list[str] | None) -> list:
    corpus_filter = [args.corpus] if args.corpus else None
    return [
        NeMoSpeechAudioReader(
            yaml_path=args.data_config,
            corpus_filter=corpus_filter,
            language_filter=language_filter,
            output_dir=args.output_dir,
            max_io_threads=args.max_io_threads,
            read_concurrency=args.read_concurrency,
            # No resampled_output_dir on purpose: keeps keep_waveform=True so the
            # writer receives audio to encode. The input WAVs are already 16 kHz.
            keep_waveform=True,
        ),
        SegmentOffsetTagStage(),
        NeMoSpeechWriterStage(
            output_dir=args.output_dir,
            target_sample_rate=args.target_sample_rate,
            writer_concurrency=args.writer_concurrency,
            save_audio=True,
        ),
    ]


def main() -> None:
    args = _build_arg_parser().parse_args()

    language_filter = [lang.strip() for lang in args.language.split(",")] if args.language else None
    stages = _build_stages(args, language_filter)

    pipeline = Pipeline(name="reader_writer_opus_backfill", stages=stages)

    if args.executor == "xenna":
        from nemo_curator.backends.xenna import XennaExecutor

        executor = XennaExecutor(config={"execution_mode": args.execution_mode})
        logger.info(f"Executor: XennaExecutor (execution_mode={args.execution_mode})")
    else:
        from nemo_curator.backends.ray_data import RayDataExecutor

        executor = RayDataExecutor()
        logger.info("Executor: RayDataExecutor (streaming)")

    logger.info(f"Reader->Writer opus backfill: {len(stages)} stages")
    logger.info(f"  Input:  {args.data_config}")
    logger.info(f"  Output: {args.output_dir}")
    if language_filter:
        logger.info(f"  Language filter: {language_filter}")
    logger.info(
        f"  target_sample_rate={args.target_sample_rate}Hz, read_concurrency={args.read_concurrency}, "
        f"writer_concurrency={args.writer_concurrency}"
    )

    t0 = time.time()
    pipeline.run(executor=executor)
    logger.info(f"Pipeline finished in {(time.time() - t0) / 60:.1f} min. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
