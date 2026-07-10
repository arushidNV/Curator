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

"""TensorRT neural-network execution for NeMo Streaming Sortformer.

The NeMo model remains responsible for audio preprocessing, streaming cache
semantics, and diarization postprocessing.  Only the expensive exported
Sortformer streaming graph is replaced by a persistent TensorRT engine.  This
keeps the Curator stage native and avoids a Triton/Riva service dependency.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from nemo_curator.utils.tensorrt_session import TensorRTSession

if TYPE_CHECKING:
    from nemo.collections.asr.models import SortformerEncLabelModel


class TensorRTSortformerRunner:
    """Run the exported streaming graph while retaining NeMo state updates."""

    def __init__(
        self,
        model: SortformerEncLabelModel,
        engine_path: str,
        *,
        model_path: str | None = None,
        metadata_path: str | None = None,
        validate_metadata: bool = True,
    ) -> None:
        if getattr(model, "async_streaming", False):
            message = "The native TensorRT Sortformer backend currently supports synchronous batched inference only"
            raise ValueError(message)
        if validate_metadata:
            if model_path is None:
                message = "model_path is required to validate a Sortformer TensorRT engine"
                raise ValueError(message)
            self._validate_metadata(engine_path, model_path, metadata_path)
        self.model = model
        self.session = TensorRTSession(engine_path, device=model.device)
        expected_inputs = {"chunk", "chunk_lengths", "spkcache", "spkcache_lengths", "fifo", "fifo_lengths"}
        expected_outputs = {"predictions", "pred_lengths", "chunk_embs", "chunk_emb_lengths"}
        missing_inputs = expected_inputs - set(self.session.input_names)
        missing_outputs = expected_outputs - set(self.session.output_names)
        if missing_inputs or missing_outputs:
            message = (
                f"Sortformer TensorRT engine has incompatible I/O; missing inputs={sorted(missing_inputs)}, "
                f"missing outputs={sorted(missing_outputs)}"
            )
            raise ValueError(message)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as input_file:
            for block in iter(lambda: input_file.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    @classmethod
    def _validate_metadata(cls, engine_path: str, model_path: str, metadata_path: str | None) -> None:
        engine = Path(engine_path)
        model = Path(model_path)
        metadata = Path(metadata_path) if metadata_path else engine.with_suffix(f"{engine.suffix}.json")
        if not metadata.is_file():
            message = f"Sortformer TensorRT metadata sidecar not found: {metadata}"
            raise FileNotFoundError(message)
        values = json.loads(metadata.read_text())
        expected_model_hash = values.get("model_sha256")
        expected_engine_hash = values.get("engine_sha256")
        if cls._sha256(model) != expected_model_hash:
            message = f"Sortformer TensorRT engine was not exported from model {model}"
            raise ValueError(message)
        if cls._sha256(engine) != expected_engine_hash:
            message = f"Sortformer TensorRT engine hash does not match {metadata}"
            raise ValueError(message)

    @staticmethod
    def _nonempty_state(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return TRT-compatible state and its unpadded per-item lengths."""
        batch_size, state_length, embedding_size = tensor.shape
        lengths = torch.full((batch_size,), state_length, dtype=torch.int64, device=tensor.device)
        if state_length:
            return tensor, lengths
        padded = torch.zeros((batch_size, 1, embedding_size), dtype=tensor.dtype, device=tensor.device)
        return padded, lengths

    def forward_streaming(
        self,
        processed_signal: torch.Tensor,
        processed_signal_length: torch.Tensor,
    ) -> torch.Tensor:
        """TensorRT counterpart of ``SortformerEncLabelModel.forward_streaming``."""
        modules = self.model.sortformer_modules
        batch_size = processed_signal.shape[0]
        streaming_state = modules.init_streaming_state(
            batch_size=batch_size,
            async_streaming=False,
            device=self.model.device,
        )
        processed_signal_offset = torch.zeros((batch_size,), dtype=torch.long, device=self.model.device)
        subsampling_factor = self.model.encoder.subsampling_factor
        expected_prediction_frames = math.ceil(processed_signal.shape[2] / subsampling_factor)
        chunk_input_frames = modules.chunk_len * modules.subsampling_factor
        num_chunks = math.ceil(processed_signal.shape[2] / chunk_input_frames)
        # Convolutional subsampling and left/right context can add a rounding
        # frame at chunk boundaries. Reserve a small deterministic cushion and
        # return only frames actually written.
        max_prediction_frames = expected_prediction_frames + 2 * num_chunks
        num_speakers = getattr(modules, "n_spk", 4)
        total_predictions = torch.empty(
            (batch_size, max_prediction_frames, num_speakers),
            dtype=torch.float32,
            device=self.model.device,
        )
        prediction_offset = 0

        streaming_loader = modules.streaming_feat_loader(
            feat_seq=processed_signal,
            feat_seq_length=processed_signal_length,
            feat_seq_offset=processed_signal_offset,
        )
        for _chunk_index, chunk, chunk_lengths, left_offset, right_offset in streaming_loader:
            spkcache, spkcache_lengths = self._nonempty_state(streaming_state.spkcache)
            fifo, fifo_lengths = self._nonempty_state(streaming_state.fifo)
            outputs = self.session.infer(
                {
                    "chunk": chunk.contiguous(),
                    "chunk_lengths": chunk_lengths.to(torch.int64),
                    "spkcache": spkcache,
                    "spkcache_lengths": spkcache_lengths,
                    "fifo": fifo,
                    "fifo_lengths": fifo_lengths,
                }
            )
            predictions = modules.apply_mask_to_preds(outputs["predictions"], outputs["pred_lengths"])
            streaming_state, chunk_predictions = modules.streaming_update(
                streaming_state=streaming_state,
                chunk=outputs["chunk_embs"],
                preds=predictions,
                lc=round(left_offset / subsampling_factor),
                rc=math.ceil(right_offset / subsampling_factor),
            )
            chunk_frames = chunk_predictions.shape[1]
            next_offset = prediction_offset + chunk_frames
            if next_offset > max_prediction_frames:
                message = (
                    f"TensorRT Sortformer produced {next_offset} frames for an allocation of {max_prediction_frames}"
                )
                raise RuntimeError(message)
            total_predictions[:, prediction_offset:next_offset].copy_(chunk_predictions)
            prediction_offset = next_offset

        return total_predictions[:, :prediction_offset]

    def close(self) -> None:
        self.session.close()
