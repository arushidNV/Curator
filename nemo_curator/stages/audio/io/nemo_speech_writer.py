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

"""Tarred dataset writer — encodes audio segments to opus and packs into tar shards.

Produces NeMo-compatible tarred datasets:
    output_dir/
        audio_0.tar          (opus-encoded audio files)
        audio_1.tar
        ...
        manifest.jsonl       (one line per segment with metadata)
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from collections import defaultdict
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
except ImportError:
    RayStageSpecKeys = None

_TARGET_SR = 16000


@dataclass
class TarredDatasetWriterStage(ProcessingStage[AudioTask, FileGroupTask]):
    """Write audio segments as opus files packed into tar shards.

    Downsamples to 16kHz mono and encodes to opus before writing.
    Generates a NeMo-format JSONL manifest alongside the tars.

    Args:
        output_dir: Root directory for output tars and manifest.
        samples_per_tar: Number of audio segments per tar shard.
        target_sample_rate: Output sample rate (default 16000).
        waveform_key: Task data key for audio waveform.
        sample_rate_key: Task data key for sample rate.
    """

    name: str = "tarred_dataset_writer"
    output_dir: str = ""
    samples_per_tar: int = 1000
    target_sample_rate: int = _TARGET_SR
    waveform_key: str = "waveform"
    sample_rate_key: str = "sampling_rate"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    _current_tar: Any = field(default=None, init=False, repr=False)
    _current_shard_id: int = field(default=0, init=False, repr=False)
    _samples_in_current_tar: int = field(default=0, init=False, repr=False)
    _manifest_file: Any = field(default=None, init=False, repr=False)
    _total_written: int = field(default=0, init=False, repr=False)

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        manifest_path = os.path.join(self.output_dir, "manifest.jsonl")
        self._manifest_file = open(manifest_path, "a", encoding="utf-8")  # noqa: SIM115
        self._open_new_tar()

    def teardown(self) -> None:
        self._close_current_tar()
        if self._manifest_file is not None:
            self._manifest_file.close()
            self._manifest_file = None
        logger.info(
            f"TarredDatasetWriter: wrote {self._total_written} segments "
            f"across {self._current_shard_id + 1} tar shards to {self.output_dir}"
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

    def _open_new_tar(self) -> None:
        tar_path = os.path.join(self.output_dir, f"audio_{self._current_shard_id}.tar")
        self._current_tar = tarfile.open(tar_path, "w")
        self._samples_in_current_tar = 0

    def _close_current_tar(self) -> None:
        if self._current_tar is not None:
            self._current_tar.close()
            self._current_tar = None

    def _encode_opus(self, waveform: np.ndarray, sr: int) -> bytes:
        if sr != self.target_sample_rate:
            import librosa

            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.target_sample_rate)

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)

        buf = io.BytesIO()
        sf.write(buf, waveform, self.target_sample_rate, format="OGG", subtype="OPUS")
        return buf.getvalue()

    def process(self, task: AudioTask) -> FileGroupTask:
        waveform = task.data.get(self.waveform_key)
        sr = task.data.get(self.sample_rate_key, self.target_sample_rate)

        if waveform is None or len(waveform) == 0:
            return FileGroupTask(task_id=task.task_id, dataset_name=task.dataset_name, data=[])

        opus_bytes = self._encode_opus(waveform, sr)

        segment_id = f"{self._current_shard_id}_{self._samples_in_current_tar}"
        filename = f"{segment_id}.opus"

        info = tarfile.TarInfo(name=filename)
        info.size = len(opus_bytes)
        self._current_tar.addfile(info, io.BytesIO(opus_bytes))

        duration = len(waveform) / sr
        manifest_entry = {
            "audio_filepath": filename,
            "duration": round(duration, 4),
            "shard_id": self._current_shard_id,
            "sampling_rate": self.target_sample_rate,
        }

        if "original_file" in task.data:
            manifest_entry["original_audio_filepath"] = task.data["original_file"]
        if "start_ms" in task.data:
            manifest_entry["offset"] = task.data["start_ms"] / 1000.0
        if "end_ms" in task.data:
            manifest_entry["original_end"] = task.data["end_ms"] / 1000.0
        if "language" in task.data:
            manifest_entry["language"] = task.data["language"]
        if "sed_events" in task.data:
            manifest_entry["sed_events"] = task.data["sed_events"]

        self._manifest_file.write(json.dumps(manifest_entry) + "\n")
        self._manifest_file.flush()

        self._samples_in_current_tar += 1
        self._total_written += 1

        if self._samples_in_current_tar >= self.samples_per_tar:
            self._close_current_tar()
            self._current_shard_id += 1
            self._open_new_tar()

        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=[os.path.join(self.output_dir, f"audio_{self._current_shard_id}.tar")],
        )
