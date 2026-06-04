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

"""NeMo Speech Writer — encodes audio segments to opus files at multiple sample rates.

Produces two versions of each segment:
    output_dir/
        16k/
            0_0.opus, 0_1.opus, ...    (16kHz mono)
        original/
            0_0.opus, 0_1.opus, ...    (original sample rate, mono)
        manifest.jsonl                  (one line per segment with metadata)
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


@dataclass
class NeMoSpeechWriterStage(ProcessingStage[AudioTask, FileGroupTask]):
    """Write audio segments as individual opus files at two sample rates.

    Produces 16kHz mono and original-rate mono versions of each segment,
    plus a single manifest referencing both.

    Args:
        output_dir: Root directory for output.
        target_sample_rate: Downsampled rate (default 16000).
        waveform_key: Task data key for audio waveform.
        sample_rate_key: Task data key for sample rate.
    """

    name: str = "nemo_speech_writer"
    output_dir: str = ""
    target_sample_rate: int = _TARGET_SR
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
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
            1 for k, v in self._shard_counts.items()
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
        return 1

    def ray_stage_spec(self) -> dict[str, Any]:
        if RayStageSpecKeys is not None:
            return {RayStageSpecKeys.IS_ACTOR_STAGE: True}
        return {"is_actor_stage": True}

    def _to_numpy_mono(self, waveform: Any) -> np.ndarray:
        if not isinstance(waveform, np.ndarray):
            import torch

            if isinstance(waveform, torch.Tensor):
                waveform = waveform.numpy()
            else:
                waveform = np.asarray(waveform, dtype=np.float32)

        if waveform.ndim > 1:
            waveform = waveform.squeeze()
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)

        return waveform.astype(np.float32)

    def _encode_opus(self, waveform: np.ndarray, sr: int) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, waveform, sr, format="OGG", subtype="OPUS")
        return buf.getvalue()

    def _resample(self, waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return waveform
        import librosa

        return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)

    def process(self, task: AudioTask) -> FileGroupTask:  # noqa: C901
        # Skip segments from already-completed shards (resume support)
        shard_key = task._metadata.get("_shard_key", "")
        if shard_key and self._shard_counts.get(shard_key) == -1:
            return FileGroupTask(task_id=task.task_id, dataset_name=task.dataset_name, data=[])

        waveform = task.data.get(self.waveform_key)
        sr = task.data.get(self.sample_rate_key, self.target_sample_rate)

        if waveform is None or (hasattr(waveform, "__len__") and len(waveform) == 0):
            return FileGroupTask(task_id=task.task_id, dataset_name=task.dataset_name, data=[])

        waveform = self._to_numpy_mono(waveform)

        # Derive filename from original audio path + offset
        original_file = task.data.get("original_file", task.data.get("audio_filepath", ""))
        base_name = os.path.splitext(os.path.basename(original_file))[0] if original_file else str(self._total_written)
        offset_ms = int(task.data.get("start_ms", 0))
        filename = f"{base_name}_{offset_ms}ms.opus"

        # Use shard_key as subdirectory to mirror input structure
        shard_subdir = shard_key or ""
        segment_dir = os.path.join(self.output_dir, shard_subdir)
        os.makedirs(segment_dir, exist_ok=True)

        # Write opus (already at target SR from upstream ResampleStage)
        opus_bytes = self._encode_opus(waveform, sr)
        out_path = os.path.join(segment_dir, filename)
        with open(out_path, "wb") as f:
            f.write(opus_bytes)

        # Build manifest entry
        duration = task.data.get("duration_sec") or (len(waveform) / sr if sr > 0 else 0)
        original_sr = task.data.get("original_sampling_rate", sr)
        rel_path = os.path.join(shard_subdir, filename) if shard_subdir else filename
        manifest_entry = {
            "audio_filepath": rel_path,
            "duration": round(duration, 4),
            "sample_rate": sr,
            "sampling_rate": sr,
            "original_sampling_rate": original_sr,
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

        # Write to per-shard manifest
        shard_manifest_path = os.path.join(self.output_dir, f"{shard_subdir}.jsonl") if shard_subdir else os.path.join(self.output_dir, "manifest.jsonl")
        os.makedirs(os.path.dirname(shard_manifest_path), exist_ok=True)
        with open(shard_manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(manifest_entry) + "\n")

        self._total_written += 1

        # Checkpointing: track per-shard progress by unique input files (not segments).
        # After VAD, one input audio produces many segments — count unique original files
        # to compare against shard_total (number of input entries in the manifest).
        shard_total = task._metadata.get("_shard_total", 0)
        if shard_key:
            input_id = task.data.get("original_file") or task.data.get("audio_filepath") or task.task_id
            if "_seen_inputs" not in self.__dict__:
                self._seen_inputs: dict[str, set] = {}
            if shard_key not in self._seen_inputs:
                self._seen_inputs[shard_key] = set()
            self._seen_inputs[shard_key].add(input_id)
            if shard_total > 0 and len(self._seen_inputs[shard_key]) >= shard_total:
                done_path = os.path.join(self.output_dir, f"{shard_subdir}.jsonl.done")
                os.makedirs(os.path.dirname(done_path), exist_ok=True)
                open(done_path, "w").close()
                logger.info(
                    f"Shard {shard_key} complete: {len(self._seen_inputs[shard_key])} inputs processed, "
                    f"{self._total_written} total segments written"
                )

        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=[out_path],
        )
