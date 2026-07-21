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

"""
VAD (Voice Activity Detection) segmentation stage.

Segments audio into speech chunks using the Silero VAD Torch or ONNX model,
filtering out silence and creating manageable segments for further processing.

Supports both CPU and GPU execution. GPU is used when available and requested
via _resources configuration.

Example:
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.audio.segmentation import VADSegmentationStage
    from nemo_curator.stages.resources import Resources

    # Default execution (CPU-only)
    pipeline.add_stage(VADSegmentationStage(min_duration_sec=2.0, threshold=0.5))

    # Opt into GPU if desired
    pipeline.add_stage(
        VADSegmentationStage(min_duration_sec=2.0)
        .with_(resources=Resources(gpus=0.3))
    )
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torchaudio
from loguru import logger
from silero_vad import get_speech_timestamps, load_silero_vad

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.audio.common import ensure_waveform_2d, load_audio_file
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

try:
    from nemo_curator.backends.utils import RayStageSpecKeys
except ImportError:
    from nemo_curator.backends.experimental.utils import RayStageSpecKeys

SILERO_SUPPORTED_RATES = {8000, 16000, 32000, 48000, 64000, 96000}
SILERO_TARGET_RATE = 16000


class _PrecomputedProbabilityModel:
    def __init__(self, probabilities: torch.Tensor) -> None:
        self.probabilities = probabilities
        self.index = 0

    def reset_states(self) -> None:
        self.index = 0

    def __call__(self, _audio: torch.Tensor, _sampling_rate: int) -> torch.Tensor:
        probability = self.probabilities[self.index]
        self.index += 1
        return probability.reshape(1, 1)


@dataclass
class VADSegmentationStage(ProcessingStage[AudioTask, AudioTask]):
    """
    Stage to segment audio using Voice Activity Detection (VAD).

    This stage takes a single AudioTask and segments it into speech chunks based on VAD,
    filtering out silence and creating manageable segments for further processing.
    Uses the model bundled by the official ``silero-vad`` package.

    Returns a list[AudioTask] with one AudioTask per detected speech segment (fan-out).

    Args:
        min_interval_ms: Minimum silence interval between speech segments in milliseconds.
        min_duration_sec: Minimum segment duration in seconds.
        max_duration_sec: Maximum segment duration in seconds.
        threshold: Voice activity detection threshold (0.0-1.0).
        speech_pad_ms: Padding in ms to add before/after speech segments.
        backend: Silero inference backend. ``"torch"`` uses TorchScript,
            ``"onnx"`` uses Silero's official ONNX Runtime wrapper, and
            ``"tensorrt"`` uses a persistent TensorRT engine on CUDA.
        tensorrt_engine_path: Serialized Silero TensorRT engine. Required when
            ``backend="tensorrt"``.
        waveform_key: Key to get waveform data.
        sample_rate_key: Key to get sample rate.

    Note:
        Default resources: cpus=1.0, gpus=0.0 (CPU). Silero VAD is lightweight.
        Use .with_(resources=Resources(gpus=X)) to opt into GPU execution.
    """

    min_interval_ms: int = 500
    min_duration_sec: float = 2.0
    max_duration_sec: float = 60.0
    threshold: float = 0.5
    speech_pad_ms: int = 300
    backend: Literal["torch", "onnx", "tensorrt"] = "torch"
    tensorrt_engine_path: str | None = None
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    nested: bool = False

    name: str = "VADSegmentation"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpus=0.0))

    def __post_init__(self):
        super().__init__()
        if self.backend not in {"torch", "onnx", "tensorrt"}:
            msg = f"Unsupported Silero backend: {self.backend!r}. Expected 'torch', 'onnx', or 'tensorrt'."
            raise ValueError(msg)
        if self.backend == "tensorrt" and not self.tensorrt_engine_path:
            msg = "tensorrt_engine_path is required for the Silero TensorRT backend"
            raise ValueError(msg)
        self._vad_model = None
        self._device = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], ["waveform", "sample_rate", "start_ms", "end_ms", "segment_num", "duration_sec"]

    def ray_stage_spec(self) -> dict[str, Any]:
        if self.nested:
            return {}
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def setup(self, _: WorkerMetadata | None = None) -> None:
        self._initialize_model()

    def teardown(self) -> None:
        if self._vad_model is not None:
            if hasattr(self._vad_model, "close"):
                self._vad_model.close()
            del self._vad_model
            self._vad_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _check_gpu_availability(gpus: float) -> None:
        if gpus > 0 and not torch.cuda.is_available():
            msg = (
                "Resources request GPU (gpus > 0) but CUDA is not available. "
                "Either set resources=Resources(gpus=0) for CPU-only or install CUDA."
            )
            raise RuntimeError(msg)

    def _initialize_model(self) -> None:
        if self._vad_model is not None:
            return
        self._check_gpu_availability(self._resources.gpus)
        if self.backend == "tensorrt" and self._resources.gpus <= 0:
            msg = "The Silero TensorRT backend requires resources=Resources(gpus=X) with X > 0"
            raise RuntimeError(msg)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Sampling rate is a multiple of 16000")
                if self.backend == "tensorrt":
                    from nemo_curator.stages.audio.segmentation.silero_tensorrt import TensorRTSileroModel

                    model = TensorRTSileroModel(self.tensorrt_engine_path)
                elif self.backend == "onnx":
                    model = load_silero_vad(onnx=True, opset_version=16)
                else:
                    model = load_silero_vad()

            use_gpu = self.backend in {"torch", "tensorrt"} and self._resources.gpus > 0 and torch.cuda.is_available()

            if self.backend == "tensorrt":
                self._device = torch.device("cuda")
                logger.info(f"Silero VAD TensorRT engine loaded on GPU: {self._device}")
            elif use_gpu:
                self._device = torch.device("cuda")
                model = model.to(self._device)
                logger.info(f"Silero VAD model loaded on GPU: {self._device}")
            else:
                self._device = torch.device("cpu")
                logger.info(f"Silero VAD model loaded on CPU ({self.backend} backend)")

            self._vad_model = model
        except Exception as e:
            logger.error(f"Failed to load VAD model: {e}")
            raise

    def _build_segment_item(
        self,
        item: dict[str, Any],
        waveform: torch.Tensor,
        sample_rate: int,
        segment: dict[str, float],
        segment_num: int,
    ) -> dict[str, Any]:
        """Build a single segment item dict from a VAD result."""
        start_ms = int(segment["start"] * 1000)
        end_ms = int(segment["end"] * 1000)
        start_sample = int(segment["start"] * sample_rate)
        end_sample = int(segment["end"] * sample_rate)

        if waveform.dim() == 1:
            segment_waveform = waveform[start_sample:end_sample].unsqueeze(0).clone()
        else:
            segment_waveform = waveform[:, start_sample:end_sample].clone()

        segment_data: dict[str, Any] = {
            k: v
            for k, v in item.items()
            if k
            not in (
                self.waveform_key,
                self.sample_rate_key,
                "start_ms",
                "end_ms",
                "segment_num",
                "duration_sec",
                "duration",
                "num_samples",
            )
        }
        segment_data.update(
            {
                "waveform": segment_waveform,
                "sample_rate": sample_rate,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "segment_num": segment_num,
                "duration_sec": (end_ms - start_ms) / 1000.0,
                "original_file": item.get("original_file", item.get("audio_filepath", "unknown")),
            }
        )
        return segment_data

    def _resolve_audio(self, item: dict[str, Any]) -> tuple[torch.Tensor, int] | None:
        """Resolve waveform and sample_rate from task data. Returns None on failure."""
        waveform = item.get(self.waveform_key)
        sample_rate = item.get(self.sample_rate_key)

        if waveform is None:
            audio_filepath = item.get("audio_filepath")
            if audio_filepath and os.path.exists(audio_filepath):
                try:
                    waveform, sample_rate = load_audio_file(audio_filepath)
                    item[self.waveform_key] = waveform
                    item[self.sample_rate_key] = sample_rate
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to load audio file {audio_filepath}: {e}")
                    return None
            else:
                logger.error("Missing waveform and no valid audio_filepath provided")
                return None
        elif sample_rate is None:
            logger.warning("Waveform present but sample_rate missing - task skipped")
            return None

        return ensure_waveform_2d(waveform), sample_rate

    def _as_read_error(self, task: AudioTask) -> AudioTask:
        """Turn a task into a read_error placeholder and drop its waveform.

        Read errors flow straight to the writer for the manifest audit trail; dropping
        the waveform keeps downstream GPU stages (SED/LangID) from touching them.
        """
        task.data["read_error"] = True
        task.data.pop(self.waveform_key, None)
        return task

    def _emit_segments(
        self,
        task: AudioTask,
        waveform: torch.Tensor,
        sample_rate: int,
        segments: list[dict[str, float]],
    ) -> AudioTask | list[AudioTask]:
        """Create fan-out tasks from timestamps shared by every inference backend."""
        if not segments:
            logger.warning("No speech segments detected by VAD")
            if self.nested:
                task.data["segments"] = []
                return task
            task.data["vad_empty"] = True
            if "duration_sec" not in task.data:
                n_samples = waveform.shape[-1] if waveform.dim() > 0 else 0
                task.data["duration_sec"] = n_samples / sample_rate if sample_rate else 0.0
            task.data.pop(self.waveform_key, None)
            return [task]

        original_file = task.data.get("audio_filepath", "unknown")
        file_name = os.path.basename(original_file) if original_file != "unknown" else task.task_id
        total_duration = sum(segment["end"] - segment["start"] for segment in segments)
        logger.info(
            f"[VADSegmentation] {file_name}: {len(segments)} segments extracted ({total_duration:.1f}s total speech)"
        )
        if self.nested:
            task.data["segments"] = [
                self._build_segment_item(task.data, waveform, sample_rate, segment, index)
                for index, segment in enumerate(segments)
            ]
            del task.data[self.waveform_key]
            return task

        output_tasks = []
        for index, segment in enumerate(segments):
            segment_data = self._build_segment_item(task.data, waveform, sample_rate, segment, index)
            segment_task = AudioTask(
                data=segment_data,
                task_id=f"{task.task_id}_seg_{index}",
                dataset_name=task.dataset_name,
            )
            if task._metadata:
                segment_task._metadata = dict(task._metadata)
            output_tasks.append(segment_task)
        return output_tasks

    def process(self, task: AudioTask) -> AudioTask | list[AudioTask]:
        """
        Process a single AudioTask.

        When ``nested=False`` (default), returns ``list[AudioTask]`` with one
        task per speech segment (fan-out).

        When ``nested=True``, returns a single ``AudioTask`` with all segment
        dicts stored in ``task.data["segments"]`` (no fan-out).
        """
        if self.backend == "tensorrt":
            return self.process_batch([task])
        if self._vad_model is None:
            msg = "VAD model failed to initialize. Cannot process audio."
            raise RuntimeError(msg)

        # Read failures from the reader flow straight to the writer (manifest audit trail).
        # Drop any residual waveform so downstream GPU stages skip them.
        if task.data.get("read_error"):
            return [self._as_read_error(task)]

        audio_result = self._resolve_audio(task.data)
        if audio_result is None:
            return []
        waveform, sample_rate = audio_result

        try:
            segments = self._get_vad_segments(waveform, sample_rate)
            return self._emit_segments(task, waveform, sample_rate, segments)
        except Exception as e:  # noqa: BLE001
            # A crash here (e.g. a corrupt/degenerate waveform) must not silently drop
            # the recording — that would leave its shard one input short forever and
            # ``.jsonl.done`` would never be written. Forward a read_error placeholder.
            logger.exception(f"Error during VAD segmentation: {e}")
            return [self._as_read_error(task)]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Batch independent recordings through the recurrent TensorRT engine."""
        if self.backend != "tensorrt":
            return super().process_batch(tasks)
        if self._vad_model is None:
            msg = "VAD model failed to initialize. Cannot process audio."
            raise RuntimeError(msg)

        resolved: list[tuple[int, AudioTask, torch.Tensor, int, torch.Tensor, int]] = []
        results_by_index: list[list[AudioTask] | None] = [None] * len(tasks)
        for index, task in enumerate(tasks):
            if task.data.get("read_error"):
                results_by_index[index] = [self._as_read_error(task)]
                continue
            try:
                audio_result = self._resolve_audio(task.data)
                if audio_result is None:
                    results_by_index[index] = []
                    continue
                waveform, sample_rate = audio_result
                vad_waveform, vad_sample_rate = self._prepare_vad_waveform(waveform, sample_rate)
                resolved.append((index, task, waveform, sample_rate, vad_waveform, vad_sample_rate))
            except Exception as e:  # noqa: BLE001
                logger.exception(f"Error preparing audio for VAD segmentation: {e}")
                results_by_index[index] = [self._as_read_error(task)]

        if not resolved:
            return [result for task_results in results_by_index if task_results for result in task_results]
        probabilities = self._vad_model.infer_probabilities([item[4] for item in resolved])
        for (index, task, waveform, sample_rate, vad_waveform, vad_sample_rate), recording_probs in zip(
            resolved, probabilities, strict=True
        ):
            try:
                timestamps = get_speech_timestamps(
                    vad_waveform,
                    _PrecomputedProbabilityModel(recording_probs),
                    sampling_rate=vad_sample_rate,
                    threshold=self.threshold,
                    min_speech_duration_ms=self.min_duration_sec * 1000,
                    max_speech_duration_s=self.max_duration_sec,
                    min_silence_duration_ms=self.min_interval_ms,
                    speech_pad_ms=self.speech_pad_ms,
                )
                segments = [
                    {"start": timestamp["start"] / vad_sample_rate, "end": timestamp["end"] / vad_sample_rate}
                    for timestamp in timestamps
                ]
                result = self._emit_segments(task, waveform, sample_rate, segments)
                results_by_index[index] = result if isinstance(result, list) else [result]
            except Exception as e:  # noqa: BLE001, PERF203
                logger.exception(f"Error during VAD segmentation: {e}")
                results_by_index[index] = [self._as_read_error(task)]
        return [result for task_results in results_by_index if task_results for result in task_results]

    def _prepare_vad_waveform(self, waveform: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, int]:
        """Convert one waveform to mono and the backend's required sample rate/device."""
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform.squeeze(0)

        target_rate = SILERO_TARGET_RATE
        needs_resample = (
            sample_rate != target_rate if self.backend == "tensorrt" else sample_rate not in SILERO_SUPPORTED_RATES
        )
        if needs_resample:
            logger.debug(f"Resampling audio from {sample_rate}Hz to {target_rate}Hz for VAD")
            waveform_cpu = waveform.cpu() if waveform.device.type != "cpu" else waveform
            if waveform_cpu.dim() == 1:
                waveform_cpu = waveform_cpu.unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_rate)
            waveform = resampler(waveform_cpu).squeeze(0)
            sample_rate = target_rate

        if self._device is not None and waveform.device != self._device:
            waveform = waveform.to(self._device)
        return waveform, sample_rate

    def _get_vad_segments(self, waveform: torch.Tensor, sample_rate: int) -> list[dict[str, float]]:
        """Get speech segments using VAD."""
        vad_waveform, vad_sample_rate = self._prepare_vad_waveform(waveform, sample_rate)

        with torch.inference_mode():
            speech_timestamps = get_speech_timestamps(
                vad_waveform,
                self._vad_model,
                sampling_rate=vad_sample_rate,
                threshold=self.threshold,
                min_speech_duration_ms=self.min_duration_sec * 1000,
                max_speech_duration_s=self.max_duration_sec,
                min_silence_duration_ms=self.min_interval_ms,
                speech_pad_ms=self.speech_pad_ms,
            )

        segments = []
        for ts in speech_timestamps:
            start_sec = ts["start"] / vad_sample_rate
            end_sec = ts["end"] / vad_sample_rate
            segments.append(
                {
                    "start": start_sec,
                    "end": end_sec,
                }
            )

        return segments
