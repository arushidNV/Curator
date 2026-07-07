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

"""NeMo Speech Writer — encodes audio segments to opus files with a JSONL manifest.

Produces one opus file per segment at the task's sample rate (typically 16kHz mono
from upstream downsampling), plus a per-shard JSONL manifest with metadata:

    output_dir/
        <shard_key>/
            <basename>_<offset_ms>ms.opus
        <shard_key>.jsonl
        <shard_key>.jsonl.done          (written when all inputs in shard are processed)
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import soundfile as sf
from loguru import logger

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask, FileGroupTask

try:
    from nemo_curator.backends.utils import RayStageSpecKeys
except (ImportError, ModuleNotFoundError):
    RayStageSpecKeys = None

_TARGET_SR = 16000


def _shard_progress_path(output_dir: str, shard_subdir: str) -> str:
    safe = shard_subdir.replace("/", "_")
    return os.path.join(output_dir, f".{safe}.shard_progress.json")


def _record_shard_input(output_dir: str, shard_subdir: str, input_id: str, shard_total: int) -> None:
    """Track unique source files per shard on disk (safe across writer actors)."""
    if not shard_subdir or not input_id or shard_total <= 0:
        return

    try:
        import fcntl
    except ImportError:
        fcntl = None

    progress_path = _shard_progress_path(output_dir, shard_subdir)
    os.makedirs(os.path.dirname(progress_path) or output_dir, exist_ok=True)

    with open(progress_path, "a+", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            raw = f.read().strip()
            data = json.loads(raw) if raw else {"seen": []}
            seen = data.setdefault("seen", [])
            if input_id not in seen:
                seen.append(input_id)
            f.seek(0)
            f.truncate()
            json.dump(data, f)
            f.flush()
            if fcntl is not None:
                os.fsync(f.fileno())
            if len(seen) >= shard_total:
                done_path = os.path.join(output_dir, f"{shard_subdir}.jsonl.done")
                os.makedirs(os.path.dirname(done_path), exist_ok=True)
                with open(done_path, "w", encoding="utf-8"):
                    pass
                logger.info(f"Shard {shard_subdir} complete: {len(seen)}/{shard_total} inputs recorded")
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _append_manifest_line(manifest_path: str, line: str) -> None:
    """Append one JSONL line after the matching opus file is on disk."""
    with open(manifest_path, "a", encoding="utf-8") as f:
        try:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except ImportError:
            f.write(line + "\n")
            f.flush()


def _source_output_stem(original_file: str) -> str:
    """Directory-preserving output stem for a source path.

    Strips the URI scheme and leading slashes but keeps intermediate directories, so
    distinct recordings that share a basename across directories (e.g.
    ``set_a/utt_001`` vs ``set_b/utt_001``) map to distinct output files instead of
    silently overwriting each other.
    """
    path = original_file.split("://", 1)[-1].lstrip("/")
    return os.path.splitext(path)[0]


def _write_opus_atomic(out_path: str, opus_bytes: bytes) -> None:
    """Write opus to a temp file, fsync, then rename so manifest never references a partial file."""
    parent = os.path.dirname(out_path)
    os.makedirs(parent, exist_ok=True)
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        f.write(opus_bytes)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, out_path)


@dataclass
class NeMoSpeechWriterStage(ProcessingStage[AudioTask, FileGroupTask]):
    """Write audio segments as individual opus files with a JSONL manifest.

    Encodes each segment at the task's current sample rate (typically 16kHz
    mono from upstream downsampling) and writes a per-shard JSONL manifest
    with duration, language, SED events, and speaker count metadata.

    Args:
        output_dir: Root directory for output.
        target_sample_rate: Expected sample rate (default 16000).
        waveform_key: Task data key for audio waveform.
        sample_rate_key: Task data key for sample rate.
    """

    name: str = "nemo_speech_writer"
    output_dir: str = ""
    target_sample_rate: int = _TARGET_SR
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    writer_concurrency: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    _total_written: int = field(default=0, init=False, repr=False)
    _shard_counts: dict = field(default_factory=dict, init=False, repr=False)

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        # Recover _shard_counts from existing .done markers on restart
        if os.path.isdir(self.output_dir):
            for root, _dirs, files in os.walk(self.output_dir):
                for fname in files:
                    if fname.endswith(".jsonl.done"):
                        rel = os.path.relpath(os.path.join(root, fname), self.output_dir)
                        shard_key = rel[: -len(".jsonl.done")]
                        self._shard_counts[shard_key] = -1  # mark as already done

    def teardown(self) -> None:
        done_count = sum(
            1
            for k, v in self._shard_counts.items()
            if v == -1 or os.path.exists(os.path.join(self.output_dir, f"{k}.jsonl.done"))
        )
        logger.info(
            f"NeMoSpeechWriter: wrote {self._total_written} segments to {self.output_dir}, "
            f"{done_count}/{len(self._shard_counts)} shards completed with .done"
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.waveform_key, self.sample_rate_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def num_workers(self) -> int | None:
        return max(1, self.writer_concurrency)

    def ray_stage_spec(self) -> dict[str, Any]:
        if RayStageSpecKeys is not None:
            return {RayStageSpecKeys.IS_ACTOR_STAGE: True}
        return {"is_actor_stage": True}

    def _ensure_numpy(self, waveform: Any) -> np.ndarray:
        """Ensure waveform is a 1-D numpy float32 array.

        Upstream MonoDownsampleStage handles mono conversion and resampling;
        this is a lightweight safety net for type coercion only.
        """
        if not isinstance(waveform, np.ndarray):
            import torch

            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()
            else:
                waveform = np.asarray(waveform, dtype=np.float32)

        if waveform.ndim > 1:
            waveform = waveform.squeeze()

        return waveform.astype(np.float32)

    def _encode_opus(self, waveform: np.ndarray, sr: int) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, waveform, sr, format="OGG", subtype="OPUS")
        return buf.getvalue()

    def _shard_manifest_path(self, shard_subdir: str) -> str:
        name = f"{shard_subdir}.jsonl" if shard_subdir else "manifest.jsonl"
        return os.path.join(self.output_dir, name)

    def _emit_manifest_only(
        self,
        task: AudioTask,
        manifest_entry: dict[str, Any],
        shard_subdir: str,
        input_id: str,
        shard_total: int,
    ) -> FileGroupTask:
        """Write a manifest-only row (no opus) for placeholder tasks and record shard progress."""
        manifest_path = self._shard_manifest_path(shard_subdir)
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        _append_manifest_line(manifest_path, json.dumps(manifest_entry))
        _record_shard_input(self.output_dir, shard_subdir, input_id, shard_total)
        return FileGroupTask(task_id=task.task_id, dataset_name=task.dataset_name, data=[])

    def process_batch(self, tasks: list[AudioTask]) -> list[FileGroupTask]:
        return [self.process(task) for task in tasks]

    def process(self, task: AudioTask) -> FileGroupTask:  # noqa: C901
        # Skip segments from already-completed shards (resume support)
        shard_key = task._metadata.get("_shard_key", "")
        if shard_key and self._shard_counts.get(shard_key) == -1:
            return FileGroupTask(task_id=task.task_id, dataset_name=task.dataset_name, data=[])

        shard_subdir = shard_key or ""
        shard_total = int(task._metadata.get("_shard_total", 0))
        original_file = task.data.get("original_file", task.data.get("audio_filepath", ""))
        input_id = original_file or task.task_id

        if task.data.get("vad_empty"):
            manifest_entry: dict[str, Any] = {
                "audio_filepath": "",
                "duration": 0.0,
                "sample_rate": self.target_sample_rate,
                "sampling_rate": self.target_sample_rate,
                "vad_empty": True,
            }
            if original_file:
                manifest_entry["original_audio_filepath"] = original_file
            source_duration = task.data.get("duration_sec") or task.data.get("duration")
            if source_duration is not None:
                manifest_entry["source_duration"] = round(float(source_duration), 4)
            for key in ("language", "language_confidence", "sed_events", "num_speakers", "corpus", "shard_id"):
                if key in task.data:
                    manifest_entry[key] = task.data[key]
            return self._emit_manifest_only(task, manifest_entry, shard_subdir, input_id, shard_total)

        if task.data.get("read_error"):
            manifest_entry = {
                "audio_filepath": "",
                "duration": 0.0,
                "sample_rate": self.target_sample_rate,
                "sampling_rate": self.target_sample_rate,
                "read_error": True,
            }
            if original_file:
                manifest_entry["original_audio_filepath"] = original_file
            for key in ("corpus", "shard_id", "source_lang"):
                if key in task.data:
                    manifest_entry[key] = task.data[key]
            return self._emit_manifest_only(task, manifest_entry, shard_subdir, input_id, shard_total)

        waveform = task.data.get(self.waveform_key)
        sr = task.data.get(self.sample_rate_key, self.target_sample_rate)

        if waveform is None or (hasattr(waveform, "__len__") and len(waveform) == 0):
            return FileGroupTask(task_id=task.task_id, dataset_name=task.dataset_name, data=[])

        waveform = self._ensure_numpy(waveform)

        # Derive filename from the original audio path, preserving directory structure
        # so distinct recordings that share a basename across dirs don't collide.
        base_name = _source_output_stem(original_file) if original_file else str(self._total_written)
        offset_ms = int(task.data.get("start_ms", 0))
        if offset_ms == 0:
            filename = f"{base_name}.opus"
        else:
            filename = f"{base_name}_{offset_ms}ms.opus"

        # Use shard_key as subdirectory to mirror input structure
        segment_dir = os.path.join(self.output_dir, shard_subdir)
        os.makedirs(segment_dir, exist_ok=True)

        # Write opus (already at target SR from upstream ResampleStage), then manifest.
        opus_bytes = self._encode_opus(waveform, sr)
        out_path = os.path.join(segment_dir, filename)
        _write_opus_atomic(out_path, opus_bytes)

        # Build manifest entry
        duration = task.data.get("duration_sec") or (len(waveform) / sr if sr > 0 else 0)
        original_sr = task.data.get("original_sampling_rate", sr)
        original_channels = task.data.get("original_channels", 1)
        rel_path = os.path.join(shard_subdir, filename) if shard_subdir else filename
        manifest_entry = {
            "audio_filepath": rel_path,
            "duration": round(duration, 4),
            "sample_rate": sr,
            "sampling_rate": sr,
            "original_sampling_rate": original_sr,
            "original_channels": original_channels,
        }

        original_file = task.data.get("original_file", task.data.get("audio_filepath", ""))
        if original_file:
            manifest_entry["original_audio_filepath"] = original_file
        if "start_ms" in task.data:
            manifest_entry["offset"] = task.data["start_ms"] / 1000.0
        if "end_ms" in task.data:
            manifest_entry["original_end"] = task.data["end_ms"] / 1000.0
        if "language" in task.data:
            manifest_entry["language"] = task.data["language"]
        if "language_confidence" in task.data:
            manifest_entry["language_confidence"] = round(task.data["language_confidence"], 4)
        if "sed_events" in task.data:
            manifest_entry["sed_events"] = task.data["sed_events"]
        if "num_speakers" in task.data:
            manifest_entry["num_speakers"] = task.data["num_speakers"]

        # Forward all remaining text/metadata fields from upstream stages.
        # Recording-level / internal keys are excluded so per-segment rows stay small
        # and accurate: e.g. diar_segments is the whole recording's diarization and
        # must NOT be copied into every clip row (O(N*M) bloat + misleading metadata).
        _INTERNAL_KEYS = {
            self.waveform_key, self.sample_rate_key,
            "waveform", "sampling_rate", "sample_rate", "num_channels",
            "original_file", "audio_filepath", "start_ms", "end_ms",
            "language", "language_confidence", "sed_events", "num_speakers",
            "duration", "duration_sec", "original_sampling_rate", "original_channels",
            "corpus", "shard_id",
            "diar_segments", "session_name", "segment_num", "vad_empty", "read_error",
        }
        for key, value in task.data.items():
            if key not in _INTERNAL_KEYS and key not in manifest_entry:
                manifest_entry[key] = value

        # Write to per-shard manifest
        shard_manifest_path = self._shard_manifest_path(shard_subdir)
        os.makedirs(os.path.dirname(shard_manifest_path), exist_ok=True)
        _append_manifest_line(shard_manifest_path, json.dumps(manifest_entry))

        self._total_written += 1
        _record_shard_input(self.output_dir, shard_subdir, input_id, shard_total)

        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=[out_path],
        )
