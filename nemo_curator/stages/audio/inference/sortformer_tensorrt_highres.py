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

"""High-resolution streaming Sortformer inference with TensorRT."""

from __future__ import annotations

import gc
import math
from typing import TYPE_CHECKING, Any

import torch
from nemo.collections.asr.models import SortformerEncLabelModel

from nemo_curator.stages.audio.common import ensure_waveform_2d, load_audio_file
from nemo_curator.stages.audio.inference.sortformer_tensorrt import _binarize
from nemo_curator.stages.audio.segmentation.silero_tensorrt import TensorRTSession

if TYPE_CHECKING:
    import numpy as np

_SPEECH_THRESHOLD = 0.5


class HighResolutionTensorRTSortformer:
    """NeMo preprocessing and streaming state around cold/steady TensorRT cores."""

    def __init__(self, model_path: str, cold_engine_path: str, steady_engine_path: str) -> None:
        model = SortformerEncLabelModel.restore_from(model_path, map_location="cpu")
        model.eval()
        model.freeze()
        model = model.cuda()
        if not model.high_resolution:
            msg = "The Sortformer checkpoint must have high-resolution output enabled"
            raise RuntimeError(msg)
        if int(model.sortformer_modules.fifo_len) != 0:
            msg = "The high-resolution TensorRT runtime currently requires fifo_len=0"
            raise RuntimeError(msg)

        self.preprocessor = model.preprocessor.eval()
        self.modules = model.sortformer_modules.eval()
        self.sample_rate = int(model.preprocessor._cfg.sample_rate)
        self.subsampling = int(model.encoder.subsampling_factor)
        self.output_subsampling = int(model.output_subsampling_factor)
        self.upsample_factor = int(model.upsample_factor)
        self.chunk_frames = int(self.modules.chunk_len) * self.subsampling
        self.left_frames = int(self.modules.chunk_left_context) * self.subsampling
        self.right_frames = int(self.modules.chunk_right_context) * self.subsampling
        self.max_feature_frames = self.chunk_frames + self.left_frames + self.right_frames
        self.num_speakers = int(self.modules.n_spk)
        self.cold = TensorRTSession(cold_engine_path)
        self.steady = TensorRTSession(steady_engine_path)
        self.engine_batch_size = self.cold.tensor_shape("chunk")[0]
        if self.engine_batch_size != 1 or self.steady.tensor_shape("chunk")[0] != 1:
            msg = "High-resolution Sortformer TensorRT requires batch-1 cold and steady engines"
            raise RuntimeError(msg)
        expected_chunk = (1, self.max_feature_frames, int(model.cfg.encoder.feat_in))
        if self.cold.tensor_shape("chunk") != expected_chunk or self.steady.tensor_shape("chunk") != expected_chunk:
            msg = f"TensorRT chunk shape must match the NeMo model: {expected_chunk}"
            raise RuntimeError(msg)
        expected_cache = (1, int(self.modules.spkcache_len), int(self.modules.fc_d_model))
        if self.steady.tensor_shape("spkcache") != expected_cache:
            msg = f"TensorRT speaker-cache shape must match the NeMo model: {expected_cache}"
            raise RuntimeError(msg)

        model.preprocessor = None
        model.sortformer_modules = None
        del model
        gc.collect()
        torch.cuda.empty_cache()

    def _prepare_waveform(self, waveform: np.ndarray | torch.Tensor, sample_rate: int) -> torch.Tensor:
        tensor = ensure_waveform_2d(torch.as_tensor(waveform, dtype=torch.float32)).mean(dim=0)
        if sample_rate != self.sample_rate:
            import torchaudio

            tensor = torchaudio.functional.resample(tensor, sample_rate, self.sample_rate)
        return tensor

    def _load_inputs(
        self,
        audio: list[np.ndarray] | list[torch.Tensor] | list[str],
        sample_rate: int | None,
    ) -> list[torch.Tensor]:
        if sample_rate is not None:
            return [self._prepare_waveform(waveform, sample_rate) for waveform in audio]
        waveforms = []
        for path in audio:
            waveform, file_sample_rate = load_audio_file(str(path))
            waveforms.append(self._prepare_waveform(waveform, file_sample_rate))
        return waveforms

    def _features(self, waveforms: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[int]]:
        features = []
        logical_lengths = []
        with torch.inference_mode():
            for waveform in waveforms:
                signal = waveform.reshape(1, -1).to(device="cuda", non_blocking=True)
                length = torch.tensor([waveform.numel()], dtype=torch.int64, device="cuda")
                item_features, item_lengths = self.preprocessor(input_signal=signal, length=length)
                features.append(item_features[0].transpose(0, 1).contiguous())
                logical_lengths.append(int(item_lengths[0]))
        return features, logical_lengths

    def _chunk(
        self,
        feature: torch.Tensor,
        logical_length: int,
        start: int,
    ) -> tuple[torch.Tensor, int, int, int]:
        if start >= feature.shape[0]:
            return feature[:0], 0, 0, 0
        left = min(self.left_frames, start)
        center_end = min(start + self.chunk_frames, feature.shape[0])
        right = min(self.right_frames, feature.shape[0] - center_end)
        chunk = feature[start - left : center_end + right]
        logical_frames = max(0, min(logical_length - start + left, chunk.shape[0]))
        return chunk, logical_frames, left, right

    def _infer_group(
        self,
        features: list[torch.Tensor],
        logical_lengths: list[int],
    ) -> list[torch.Tensor]:
        if len(features) != 1:
            msg = f"Expected one recording, received {len(features)}"
            raise ValueError(msg)
        feature = features[0]
        logical_length = logical_lengths[0]
        steps = math.ceil(feature.shape[0] / self.chunk_frames)
        state = self.modules.init_streaming_state(
            batch_size=1,
            async_streaming=False,
            device=torch.device("cuda"),
        )
        prediction_chunks = []

        with torch.inference_mode():
            for step in range(steps):
                start = step * self.chunk_frames
                chunk, valid_frames, left, right = self._chunk(feature, logical_length, start)
                padded = torch.zeros(
                    (1, self.max_feature_frames, feature.shape[1]),
                    dtype=torch.float32,
                    device="cuda",
                )
                padded[0, : chunk.shape[0]] = chunk
                physical_embedding_length = math.ceil(chunk.shape[0] / self.subsampling)
                logical_embedding_length = math.ceil(valid_frames / self.subsampling)
                left_embedding = round(left / self.subsampling)
                right_embedding = math.ceil(right / self.subsampling)
                saved_cache_length = state.spkcache.shape[1]
                values = {
                    "chunk": padded,
                    "chunk_lengths": torch.tensor([valid_frames], dtype=torch.int64, device="cuda"),
                }
                session = self.cold if step == 0 else self.steady
                if step:
                    values["spkcache"] = state.spkcache
                outputs = session.infer(values)
                packed_physical_length = saved_cache_length + physical_embedding_length
                packed_logical_length = saved_cache_length + logical_embedding_length
                low_resolution = outputs["cache_resolution_preds"][:, :packed_physical_length]
                high_resolution = outputs["high_resolution_preds"][:, : packed_physical_length * self.upsample_factor]
                chunk_embeddings = outputs["chunk_embeddings"][:, :physical_embedding_length]

                if state.spk_perm is not None:
                    inverse = torch.argsort(state.spk_perm[0])
                    high_resolution = high_resolution[:, :, inverse]

                packed_lengths = torch.tensor([packed_logical_length], dtype=torch.int64, device="cuda")
                low_resolution = self.modules.apply_mask_to_preds(low_resolution, packed_lengths)
                state, _ = self.modules.streaming_update(
                    streaming_state=state,
                    chunk=chunk_embeddings,
                    preds=low_resolution,
                    lc=left_embedding,
                    rc=right_embedding,
                )

                current_embeddings = physical_embedding_length - left_embedding - right_embedding
                output_start = (saved_cache_length + left_embedding) * self.upsample_factor
                current = high_resolution[:, output_start : output_start + current_embeddings * self.upsample_factor]
                if self.output_subsampling > 1:
                    current = self.modules.downsample_preds(current, self.output_subsampling)
                prediction_chunks.append(current[0].float().cpu())

        probabilities = torch.cat(prediction_chunks)[: math.ceil(feature.shape[0] / self.output_subsampling)]
        return [probabilities]

    def _infer_probabilities(
        self,
        features: list[torch.Tensor],
        logical_lengths: list[int],
    ) -> list[torch.Tensor]:
        probabilities = []
        for start in range(0, len(features), self.engine_batch_size):
            end = start + self.engine_batch_size
            probabilities.extend(self._infer_group(features[start:end], logical_lengths[start:end]))
        return probabilities

    def _segments(self, probabilities: torch.Tensor) -> list[dict[str, Any]]:
        frame_step = 0.01 * self.output_subsampling
        segments = []
        active_speakers = (probabilities > _SPEECH_THRESHOLD).any(dim=0).nonzero().flatten().tolist()
        for speaker in active_speakers:
            for start, end in _binarize(probabilities[:, speaker], frame_step).tolist():
                segments.append(
                    {
                        "start": round(float(start), 2),
                        "end": round(float(end), 2),
                        "speaker": f"speaker_{speaker}",
                    }
                )
        return sorted(segments, key=lambda segment: (segment["start"], segment["speaker"]))

    def diarize(
        self,
        audio: list[np.ndarray] | list[torch.Tensor] | list[str],
        sample_rate: int | None = None,
    ) -> list[list[dict[str, Any]]]:
        waveforms = self._load_inputs(audio, sample_rate)
        features, logical_lengths = self._features(waveforms)
        probabilities = self._infer_probabilities(features, logical_lengths)
        return [self._segments(item) for item in probabilities]

    def close(self) -> None:
        self.cold.close()
        self.steady.close()
