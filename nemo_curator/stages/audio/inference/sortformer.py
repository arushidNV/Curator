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

from __future__ import annotations

import contextlib
import os
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import soundfile as sf
from huggingface_hub import snapshot_download
from loguru import logger
from nemo.collections.asr.models import SortformerEncLabelModel

from nemo_curator.stages.base import ProcessingStage

if TYPE_CHECKING:
    import numpy as np

    from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


def _parse_sortformer_segments(raw_segments: list) -> list[dict[str, Any]]:
    """Convert Sortformer output segments to list of {start, end, speaker} dicts.

    Handles both string format ("start end speaker") and objects with
    start/end/speaker attributes.
    """
    segments: list[dict[str, Any]] = []
    for seg in raw_segments:
        if isinstance(seg, str):
            parts = seg.strip().split()
            segments.append(
                {
                    "start": float(parts[0]),
                    "end": float(parts[1]),
                    "speaker": parts[2] if len(parts) > 2 else "unknown",  # noqa: PLR2004
                }
            )
        elif hasattr(seg, "start") and hasattr(seg, "end"):
            segments.append(
                {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "speaker": str(getattr(seg, "speaker", getattr(seg, "label", "unknown"))),
                }
            )
        elif isinstance(seg, (tuple, list)) and len(seg) >= 3:  # noqa: PLR2004
            segments.append(
                {
                    "start": float(seg[0]),
                    "end": float(seg[1]),
                    "speaker": str(seg[2]),
                }
            )
        else:
            logger.warning(f"Unrecognised segment format: {seg!r}")
    return segments


def _write_rttm(segments: list[dict[str, Any]], sess_name: str, rttm_out_dir: str) -> None:
    """Write diarization segments to an RTTM file."""
    os.makedirs(rttm_out_dir, exist_ok=True)
    rttm_path = os.path.join(rttm_out_dir, f"{sess_name}.rttm")
    with open(rttm_path, "w") as f:
        for seg in segments:
            duration = seg["end"] - seg["start"]
            if duration <= 0:
                logger.warning(f"Skipping degenerate segment with non-positive duration: {seg!r}")
                continue
            f.write(f"SPEAKER {sess_name} 1 {seg['start']:.3f} {duration:.3f} <NA> <NA> {seg['speaker']} <NA> <NA>\n")


def _waveform_to_tempfile(waveform: np.ndarray, sample_rate: int) -> str:
    """Write an in-memory waveform (numpy float32) to a temporary WAV file.

    Returns the path to the temp file.  Caller is responsible for cleanup.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        try:
            sf.write(tmp.name, waveform, sample_rate)
        except Exception:
            os.unlink(tmp.name)
            raise
        return tmp.name


@dataclass
class InferenceSortformerStage(ProcessingStage[AudioTask, AudioTask]):
    """Speaker diarization inference using Streaming Sortformer (NeMo).

    Uses the NeMo SortformerEncLabelModel for end-to-end neural speaker
    diarization with streaming support. See:
    https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2

    Supports two input modes:

    1. **File path mode** (default): reads audio from
       ``task.data[filepath_key]`` (e.g. ``"audio_filepath"``).
    2. **In-memory waveform mode**: when ``task.data[waveform_key]``
       exists, the stage writes the waveform to a temp file, runs
       diarization, and cleans up.  This is the mode used by the
       NeMo tarred audio pipeline where audio lives in memory.

    Args:
        model_name: Hugging Face model id.
        model_path: Local path to a .nemo checkpoint file; overrides model_name.
        cache_dir: Directory for caching downloaded model weights.
        diar_model: Pre-loaded SortformerEncLabelModel; if provided, setup() is a no-op.
        filepath_key: Key in data for path to audio file.
        waveform_key: Key in data for in-memory waveform (numpy float32).
        sample_rate_key: Key in data for sample rate (int).
        diar_segments_key: Key in output data for diarization segments list.
        num_speakers_key: Key in output data for the number of distinct speakers.
        store_segments: Whether to store the full diar_segments in task.data.
        rttm_out_dir: Optional directory to write RTTM files.
        chunk_len: Streaming chunk size in 80 ms frames.
        chunk_right_context: Right context frames.
        fifo_len: FIFO queue size in frames.
        spkcache_update_period: Speaker cache update period in frames.
        spkcache_len: Speaker cache size in frames.
        inference_batch_size: Batch size passed to diarize().
        name: Stage name.
    """

    model_name: str = "nvidia/diar_streaming_sortformer_4spk-v2"
    model_path: str | None = None
    cache_dir: str | None = None
    diar_model: Any | None = None
    filepath_key: str = "audio_filepath"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    diar_segments_key: str = "diar_segments"
    num_speakers_key: str = "num_speakers"
    store_segments: bool = True
    rttm_out_dir: str | None = None
    chunk_len: int = 340
    chunk_right_context: int = 40
    fifo_len: int = 40
    spkcache_update_period: int = 300
    spkcache_len: int = 188
    inference_batch_size: int = 1
    name: str = "Sortformer_inference"
    batch_size: int = 8
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpu_memory_gb=8.0))

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        """Pre-download model weights on the node so actors load from cache."""
        if self.model_path is not None:
            return
        try:
            repo_dir = snapshot_download(repo_id=self.model_name, cache_dir=self.cache_dir)
            nemo_files = [f for f in os.listdir(repo_dir) if f.endswith(".nemo")]
            if nemo_files:
                self.model_path = os.path.join(repo_dir, nemo_files[0])
            else:
                logger.warning(f"No .nemo file found in {repo_dir}; setup() will fail")
        except Exception:  # noqa: BLE001
            logger.info(f"Could not pre-cache {self.model_name}; actors will download on first use")

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Load Sortformer model from Hugging Face or a local .nemo file."""
        if self.diar_model is not None:
            self.diar_model.eval()
            self._configure_streaming()
            return

        restore_path = self.model_path
        if not restore_path and self.model_name.endswith(".nemo"):
            restore_path = self.model_name

        if restore_path:
            self.diar_model = SortformerEncLabelModel.restore_from(
                restore_path=restore_path,
                map_location="cuda",
                strict=False,
            )
        else:
            self.diar_model = SortformerEncLabelModel.from_pretrained(
                model_name=self.model_name,
                map_location="cuda",
            )

        self.diar_model.eval()
        self._configure_streaming()

    def _configure_streaming(self) -> None:
        """Apply streaming configuration to the loaded model."""
        sm = self.diar_model.sortformer_modules
        sm.chunk_len = self.chunk_len
        sm.chunk_right_context = self.chunk_right_context
        sm.fifo_len = self.fifo_len
        sm.spkcache_update_period = self.spkcache_update_period
        sm.spkcache_len = self.spkcache_len

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        out = [self.num_speakers_key]
        if self.store_segments:
            out.append(self.diar_segments_key)
        return ["data"], out

    def diarize(self, audio_paths: list[str]) -> list[list[dict[str, Any]]]:
        """Run Sortformer on a list of audio files.

        Returns a list (one entry per file) of segment lists [{start, end, speaker}].
        """
        predicted_segments = self.diar_model.diarize(
            audio=audio_paths,
            batch_size=self.inference_batch_size,
        )
        return [_parse_sortformer_segments(segs) for segs in predicted_segments]

    def _resolve_audio_path(self, task: AudioTask) -> tuple[str, bool]:
        """Resolve the audio path for a task.

        Returns ``(path, is_temp)`` where *is_temp* indicates a temporary
        file that the caller must clean up.
        """
        waveform = task.data.get(self.waveform_key)
        if waveform is not None:
            tmp_path = _waveform_to_tempfile(waveform, task.data[self.sample_rate_key])
            return tmp_path, True
        return task.data[self.filepath_key], False

    def _apply_results(self, task: AudioTask, segments: list[dict[str, Any]]) -> None:
        """Write diarization results into *task.data*."""
        task.data[self.num_speakers_key] = len({seg["speaker"] for seg in segments})
        if self.store_segments:
            task.data[self.diar_segments_key] = segments

    def process(self, task: AudioTask) -> AudioTask:
        """Run speaker diarization on a single task."""
        audio_path, is_temp = self._resolve_audio_path(task)
        try:
            segments = self.diarize([audio_path])[0]
        finally:
            if is_temp:
                os.unlink(audio_path)

        if self.rttm_out_dir is not None:
            sess_name = task.data.get("session_name") or os.path.splitext(os.path.basename(audio_path))[0]
            _write_rttm(segments, sess_name, self.rttm_out_dir)

        self._apply_results(task, segments)
        return task

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Run batched speaker diarization across multiple tasks."""
        if not tasks:
            return []

        audio_paths: list[str] = []
        is_temp: list[bool] = []

        try:
            for t in tasks:
                path, tmp = self._resolve_audio_path(t)
                audio_paths.append(path)
                is_temp.append(tmp)

            all_segments = self.diarize(audio_paths)

            for task, segments, path in zip(tasks, all_segments, audio_paths, strict=True):
                self._apply_results(task, segments)

                if self.rttm_out_dir is not None:
                    sess_name = task.data.get("session_name") or os.path.splitext(os.path.basename(path))[0]
                    _write_rttm(segments, sess_name, self.rttm_out_dir)
        finally:
            for path, tmp in zip(audio_paths, is_temp, strict=True):
                if tmp:
                    with contextlib.suppress(OSError):
                        os.unlink(path)

        logger.info(f"Sortformer: diarized {len(tasks)} samples")
        return tasks
