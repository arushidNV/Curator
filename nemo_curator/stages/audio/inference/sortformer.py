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

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

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
        elif isinstance(seg, dict) and {"start", "end", "speaker"} <= seg.keys():
            segments.append(
                {
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "speaker": str(seg["speaker"]),
                }
            )
        else:
            logger.warning(f"Unrecognised segment format: {seg!r}")
    return segments


def _safe_rttm_basename(sess_name: str) -> str:
    """Filesystem-safe RTTM filename stem (task ids may contain shard path slashes)."""
    return sess_name.replace("\\", "_").replace("/", "_")


def _write_rttm(segments: list[dict[str, Any]], sess_name: str, rttm_out_dir: str) -> None:
    """Write diarization segments to an RTTM file.

    Called once per full-audio recording (before VAD fan-out), so one file per input —
    not a per-segment bottleneck.
    """
    os.makedirs(rttm_out_dir, exist_ok=True)
    rttm_path = os.path.join(rttm_out_dir, f"{_safe_rttm_basename(sess_name)}.rttm")
    lines: list[str] = []
    for seg in segments:
        duration = seg["end"] - seg["start"]
        if duration <= 0:
            logger.warning(f"Skipping degenerate segment with non-positive duration: {seg!r}")
            continue
        lines.append(f"SPEAKER {sess_name} 1 {seg['start']:.3f} {duration:.3f} <NA> <NA> {seg['speaker']} <NA> <NA>\n")
    with open(rttm_path, "w") as f:
        f.writelines(lines)


@dataclass
class InferenceSortformerStage(ProcessingStage[AudioTask, AudioTask]):
    """Speaker diarization inference using Streaming Sortformer (NeMo).

    Uses the NeMo SortformerEncLabelModel for end-to-end neural speaker
    diarization with streaming support. See:
    https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2

    Supports two input modes:

    1. **File path mode**: reads audio from ``task.data[filepath_key]``.
    2. **In-memory waveform mode**: when ``task.data[waveform_key]``
       exists, numpy arrays are passed directly to NeMo's ``diarize()``
       API (requires NeMo >= 2.7).

    Args:
        model_name: Hugging Face model id or local ``.nemo`` path.
        model_path: Local path to a .nemo checkpoint file; overrides model_name.
        cache_dir: Directory for caching downloaded model weights.
        diar_model: Pre-loaded SortformerEncLabelModel; if provided, setup() is a no-op.
        filepath_key: Key in data for path to audio file.
        waveform_key: Key in data for in-memory waveform (numpy float32).
        sample_rate_key: Key in data for sample rate (int).
        duration_key: Key in data for audio duration in seconds.
        diar_segments_key: Key in output data for diarization segments list.
        num_speakers_key: Key in output data for the number of distinct speakers.
        store_segments: Whether to store the full diar_segments in task.data.
        rttm_out_dir: Optional directory to write RTTM files.
        chunk_len: Optional streaming chunk size override in encoder frames.
        chunk_right_context: Optional right context override in encoder frames.
        fifo_len: Optional FIFO queue size override in encoder frames.
        spkcache_update_period: Optional speaker cache update period override.
        spkcache_len: Optional speaker cache size override in encoder frames.
        inference_batch_size: Batch size passed to diarize().
        backend: ``"nemo"`` or ``"tensorrt"``.
        tensorrt_engine_path: TensorRT plan file.
        tensorrt_config_path: Runtime JSON stored with the TensorRT plan.
        tensorrt_runtime_module_path: Matching Riva ``sortformer_modules.py``.
        precision: Inference precision.
        compile_encoder: Whether to compile the 31-layer encoder with torch.compile.
        name: Stage name.
    """

    model_name: str = "nvidia/diar_streaming_sortformer_4spk-v2"
    model_path: str | None = None
    cache_dir: str | None = None
    diar_model: Any | None = None
    filepath_key: str = "audio_filepath"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    duration_key: str = "duration"
    diar_segments_key: str = "diar_segments"
    num_speakers_key: str = "num_speakers"
    store_segments: bool = True
    rttm_out_dir: str | None = None
    chunk_len: int | None = None
    chunk_right_context: int | None = None
    fifo_len: int | None = None
    spkcache_update_period: int | None = None
    spkcache_len: int | None = None
    inference_batch_size: int = 1
    backend: Literal["nemo", "tensorrt"] = "nemo"
    tensorrt_engine_path: str | None = None
    tensorrt_config_path: str | None = None
    tensorrt_runtime_module_path: str | None = None
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    compile_encoder: bool = False
    name: str = "Sortformer_inference"
    batch_size: int = 8
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpu_memory_gb=8.0))

    def __post_init__(self) -> None:
        if self.backend not in {"nemo", "tensorrt"}:
            msg = f"Unsupported Sortformer backend: {self.backend}"
            raise ValueError(msg)
        if self.precision not in {"fp32", "fp16", "bf16"}:
            msg = f"Unsupported Sortformer precision: {self.precision}"
            raise ValueError(msg)
        if self.backend == "tensorrt" and not all(
            (self.tensorrt_engine_path, self.tensorrt_config_path, self.tensorrt_runtime_module_path)
        ):
            msg = "Sortformer TensorRT requires engine, config, and runtime module paths"
            raise ValueError(msg)
        self._tensorrt_model = None

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        """Pre-download model weights on the node so actors load from cache."""
        if self.backend == "tensorrt" or self.model_path is not None:
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
        if self.backend == "tensorrt":
            from nemo_curator.stages.audio.inference.sortformer_tensorrt import TensorRTSortformer

            self._tensorrt_model = TensorRTSortformer(
                self.tensorrt_engine_path,
                self.tensorrt_config_path,
                self.tensorrt_runtime_module_path,
            )
            return

        if self.diar_model is not None:
            self.diar_model.eval()
            self._configure_streaming()
            self._compile_encoder()
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
        self._compile_encoder()

    def teardown(self) -> None:
        if self._tensorrt_model is not None:
            self._tensorrt_model.close()
            self._tensorrt_model = None

    def _configure_streaming(self) -> None:
        """Apply explicit streaming overrides to the loaded model."""
        sm = self.diar_model.sortformer_modules
        overrides = {
            "chunk_len": self.chunk_len,
            "chunk_right_context": self.chunk_right_context,
            "fifo_len": self.fifo_len,
            "spkcache_update_period": self.spkcache_update_period,
            "spkcache_len": self.spkcache_len,
        }
        for name, value in overrides.items():
            if value is not None:
                setattr(sm, name, value)
        if any(value is not None for value in overrides.values()) and hasattr(sm, "_check_streaming_parameters"):
            sm._check_streaming_parameters()

    def _compile_encoder(self) -> None:
        """Compile the fixed-shape encoder core."""
        if not self.compile_encoder:
            return
        import torch

        self.diar_model.encoder = torch.compile(self.diar_model.encoder, dynamic=False)
        logger.info("Sortformer: compiled encoder")

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        out = [self.num_speakers_key]
        if self.store_segments:
            out.append(self.diar_segments_key)
        return ["data"], out

    def _diarize(
        self, audio: list[np.ndarray] | list[str], sample_rate: int | None = None
    ) -> list[list[dict[str, Any]]]:
        """Run Sortformer diarization on a list of audio inputs.

        Accepts either file paths or numpy arrays (with sample_rate).
        """
        kwargs: dict[str, Any] = {"audio": audio, "batch_size": self.inference_batch_size}
        if sample_rate is not None:
            kwargs["sample_rate"] = sample_rate
        if self.backend == "tensorrt":
            kwargs.pop("batch_size")
            return self._tensorrt_model.diarize(**kwargs)
        if self.precision == "fp32":
            predicted_segments = self.diar_model.diarize(**kwargs)
        else:
            import torch

            dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
            with torch.autocast(device_type="cuda", dtype=dtype):
                predicted_segments = self.diar_model.diarize(**kwargs)
        return [_parse_sortformer_segments(segs) for segs in predicted_segments]

    def _apply_results(self, task: AudioTask, segments: list[dict[str, Any]]) -> None:
        """Write diarization results into *task.data*."""
        task.data[self.num_speakers_key] = len({seg["speaker"] for seg in segments})
        if self.store_segments:
            task.data[self.diar_segments_key] = segments

    def _session_name(self, task: AudioTask) -> str:
        """RTTM session/name key: explicit session_name, else the audio basename, else task_id.

        Keying on the audio basename (not task_id) keeps RTTMs matchable to their
        source recording; task_id defaults to "" and would collide across recordings.
        """
        sess_name = task.data.get("session_name")
        if sess_name:
            return sess_name
        filepath = task.data.get(self.filepath_key)
        if filepath:
            return os.path.splitext(os.path.basename(filepath))[0]
        return task.task_id

    def _duration(self, task: AudioTask) -> float:
        """Return duration for local batch ordering, with a waveform fallback."""
        duration = task.data.get(self.duration_key)
        if duration is not None:
            try:
                value = float(duration)
                if value > 0:
                    return value
            except (TypeError, ValueError):
                pass

        waveform = task.data.get(self.waveform_key)
        sample_rate = task.data.get(self.sample_rate_key)
        if waveform is not None and sample_rate:
            return float(waveform.shape[-1]) / float(sample_rate)
        return float("inf")

    def process(self, task: AudioTask) -> AudioTask:
        """Run speaker diarization on a single task."""
        if task.data.get("read_error"):
            return task

        waveform = task.data.get(self.waveform_key)
        if waveform is not None:
            sr = task.data.get(self.sample_rate_key)
            if sr is None:
                msg = f"Sortformer: waveform provided but '{self.sample_rate_key}' is missing in task {task.task_id}"
                raise ValueError(msg)
            segments = self._diarize([waveform], sample_rate=sr)[0]
        else:
            filepath = task.data.get(self.filepath_key)
            if filepath is None:
                msg = (
                    f"Sortformer: neither '{self.waveform_key}' nor '{self.filepath_key}' found in task {task.task_id}"
                )
                raise ValueError(msg)
            segments = self._diarize([filepath])[0]

        if self.rttm_out_dir is not None:
            _write_rttm(segments, self._session_name(task), self.rttm_out_dir)

        self._apply_results(task, segments)
        task.task_id = f"{task.task_id}_sortformer"
        return task

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Run batched speaker diarization across multiple tasks."""
        if len(tasks) == 0:
            return []

        to_process = sorted(
            (task for task in tasks if not task.data.get("read_error")),
            key=self._duration,
        )
        if not to_process:
            return tasks

        waveforms = [t.data.get(self.waveform_key) for t in to_process]
        has_waveform = [w is not None for w in waveforms]
        if any(has_waveform) and not all(has_waveform):
            msg = "Sortformer: batch contains a mix of waveform and filepath tasks; all tasks must use the same mode"
            raise ValueError(msg)
        use_waveform = has_waveform[0]

        if use_waveform:
            sr = to_process[0].data.get(self.sample_rate_key)
            if sr is None:
                msg = f"Sortformer: waveform provided but '{self.sample_rate_key}' is missing"
                raise ValueError(msg)
            all_segments = self._diarize(waveforms, sample_rate=sr)
        else:
            paths = [t.data[self.filepath_key] for t in to_process]
            all_segments = self._diarize(paths)

        for task, segments in zip(to_process, all_segments, strict=True):
            self._apply_results(task, segments)

        # Write RTTM files after all GPU results are applied (batch disk I/O)
        if self.rttm_out_dir is not None:
            for task, segments in zip(to_process, all_segments, strict=True):
                _write_rttm(segments, self._session_name(task), self.rttm_out_dir)

        for task in to_process:
            task.task_id = f"{task.task_id}_sortformer"

        logger.info(f"Sortformer: diarized {len(to_process)} samples")
        return tasks
