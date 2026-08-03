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

"""Indic Parakeet RNN-T inference with a TensorRT encoder and NeMo decoder."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from loguru import logger

from nemo_curator.stages.audio.inference.asr_nemo import NemoASRModel
from nemo_curator.stages.audio.inference.audio_chunking import (
    engine_chunk_duration,
    has_audio_longer_than,
    merge_chunk_texts,
    split_waveforms,
)
from nemo_curator.stages.audio.inference.tensorrt_encoder import (
    ENGINE_FILENAME,
    MODEL_FILENAME,
    TensorRTEncoder,
)
from nemo_curator.stages.audio.inference.tensorrt_encoder import (
    load_engine_metadata as _load_engine_metadata,
)

if TYPE_CHECKING:
    import numpy as np

_MAX_CHUNK_DURATION_SEC = 40.0


def load_engine_metadata(engine_dir: str | Path) -> dict[str, Any]:
    """Load and validate an Indic Parakeet RNN-T engine bundle manifest."""
    return _load_engine_metadata(
        engine_dir,
        model_type="indic_parakeet_rnnt",
        required_positive_ints=(
            "sample_rate",
            "feature_count",
            "subsampling_factor",
            "vocabulary_size",
            "max_symbols_per_step",
        ),
    )


class TensorRTParakeetRNNTModel(NemoASRModel):
    """NeMo RNN-T wrapper that replaces only the Conformer encoder with TensorRT."""

    def __init__(
        self,
        engine_dir: str | Path,
        *,
        inference_batch_size: int = 16,
        chunking_mode: Literal["engine", "none"] = "engine",
    ) -> None:
        if chunking_mode not in {"engine", "none"}:
            msg = f"Unsupported Indic Parakeet chunking mode: {chunking_mode!r}"
            raise ValueError(msg)
        self.engine_dir = Path(engine_dir)
        self.metadata = load_engine_metadata(self.engine_dir)
        engine_path = self.engine_dir / ENGINE_FILENAME
        model_path = self.engine_dir / MODEL_FILENAME
        if not engine_path.is_file():
            msg = f"TensorRT encoder engine not found: {engine_path}"
            raise FileNotFoundError(msg)
        if not model_path.is_file():
            msg = f"Bundled NeMo model not found: {model_path}"
            raise FileNotFoundError(msg)

        super().__init__(model_name=str(model_path), inference_batch_size=inference_batch_size)
        self.chunking_mode = chunking_mode
        self._engine_path = engine_path
        self._trt_encoder: TensorRTEncoder | None = None
        self._chunk_duration_sec: float | None = None

    def _disable_cuda_graphs(self) -> None:
        # This backend deliberately uses NeMo's batched CUDA-graph RNN-T decoder.
        return

    def setup(self, device: torch.device | str | None = None) -> None:
        if self._trt_encoder is not None:
            return
        if not torch.cuda.is_available():
            msg = "Indic Parakeet TensorRT inference requires CUDA"
            raise RuntimeError(msg)
        if device is not None and torch.device(device).type != "cuda":
            msg = "Indic Parakeet TensorRT inference requires a CUDA device"
            raise ValueError(msg)

        super().setup(device=torch.device("cuda"))
        if self.asr_model is None:
            msg = "NeMo ASR model did not load"
            raise RuntimeError(msg)
        self._validate_model()
        self.asr_model.to(dtype=torch.float16)
        self._enable_batched_greedy_decoder()

        original_encoder = self.asr_model.encoder
        self.asr_model.encoder = None
        del original_encoder
        gc.collect()
        torch.cuda.empty_cache()
        self._trt_encoder = TensorRTEncoder(
            self._engine_path,
            subsampling_factor=int(self.metadata["subsampling_factor"]),
        )
        max_input_shape = self._trt_encoder.max_input_shape("audio_signal")
        self.inference_batch_size = min(self.inference_batch_size, max_input_shape[0])
        max_feature_frames = max_input_shape[2]
        if self.chunking_mode == "engine":
            engine_duration = engine_chunk_duration(self.asr_model, max_feature_frames)
            if engine_duration < _MAX_CHUNK_DURATION_SEC:
                msg = (
                    "Indic Parakeet TensorRT engine does not support 40-second audio: "
                    f"max_feature_frames={max_feature_frames}; rebuild with --max-frames 4001"
                )
                raise ValueError(msg)
            self._chunk_duration_sec = _MAX_CHUNK_DURATION_SEC
        self.asr_model.encoder = self._trt_encoder
        logger.info(f"Indic Parakeet TensorRT encoder loaded: {self._engine_path}")

    def _validate_model(self) -> None:
        model = self.asr_model
        if model is None:
            msg = "NeMo ASR model not loaded; call setup() first."
            raise RuntimeError(msg)
        if not hasattr(model, "decoder") or not hasattr(model, "joint"):
            msg = "Indic Parakeet TensorRT backend requires a NeMo RNN-T model"
            raise TypeError(msg)

        encoder = getattr(model, "encoder", None)
        actual_subsampling = int(getattr(encoder, "subsampling_factor", -1))
        expected_subsampling = int(self.metadata["subsampling_factor"])
        if actual_subsampling != expected_subsampling:
            msg = (
                "Bundled NeMo model does not match the TensorRT engine: "
                f"subsampling_factor={actual_subsampling}, expected={expected_subsampling}"
            )
            raise ValueError(msg)

        actual_feature_count = int(getattr(encoder, "_feat_in", getattr(model.cfg.encoder, "feat_in", -1)))
        expected_feature_count = int(self.metadata["feature_count"])
        if actual_feature_count != expected_feature_count:
            msg = (
                "Bundled NeMo model does not match the TensorRT engine: "
                f"feature_count={actual_feature_count}, expected={expected_feature_count}"
            )
            raise ValueError(msg)

        preprocessor_cfg = getattr(model.cfg, "preprocessor", None)
        actual_sample_rate = int(getattr(preprocessor_cfg, "sample_rate", -1))
        expected_sample_rate = int(self.metadata["sample_rate"])
        if actual_sample_rate != expected_sample_rate:
            msg = (
                "Bundled NeMo model does not match the TensorRT engine: "
                f"sample_rate={actual_sample_rate}, expected={expected_sample_rate}"
            )
            raise ValueError(msg)

        actual_vocabulary_size = int(
            getattr(model.joint, "_vocab_size", getattr(model.cfg.joint, "num_classes", -1))
        )
        expected_vocabulary_size = int(self.metadata["vocabulary_size"])
        if actual_vocabulary_size != expected_vocabulary_size:
            msg = (
                "Bundled NeMo model does not match the TensorRT engine: "
                f"vocabulary_size={actual_vocabulary_size}, expected={expected_vocabulary_size}"
            )
            raise ValueError(msg)

    def _enable_batched_greedy_decoder(self) -> None:
        model = self.asr_model
        if model is None:
            msg = "NeMo ASR model not loaded; call setup() first."
            raise RuntimeError(msg)
        from omegaconf import OmegaConf, open_dict

        with open_dict(model.cfg):
            model.cfg.decoding.strategy = "greedy_batch"
            greedy_cfg = OmegaConf.select(model.cfg, "decoding.greedy")
            if greedy_cfg is None:
                model.cfg.decoding.greedy = OmegaConf.create({})
            model.cfg.decoding.greedy.max_symbols_per_step = int(self.metadata["max_symbols_per_step"])
            model.cfg.decoding.greedy.use_cuda_graph_decoder = True
        model.change_decoding_strategy(model.cfg.decoding)

    def transcribe_waveforms(
        self,
        waveforms: list[np.ndarray],
        sample_rates: list[int],
    ) -> list[str]:
        if self.asr_model is None:
            msg = "NeMo ASR model not loaded; call setup() first."
            raise RuntimeError(msg)
        if not waveforms:
            return []
        if self.chunking_mode == "none":
            return super().transcribe_waveforms(waveforms, sample_rates)
        if self._chunk_duration_sec is None:
            msg = "Indic Parakeet chunk duration was not initialized from the model"
            raise RuntimeError(msg)
        requires_merge = has_audio_longer_than(waveforms, sample_rates, self._chunk_duration_sec)

        chunks, chunk_sample_rates, owners = split_waveforms(
            waveforms,
            sample_rates,
            self._chunk_duration_sec,
        )
        if not chunks:
            return [""] * len(waveforms)
        duration_order = sorted(
            range(len(chunks)),
            key=lambda index: chunks[index].shape[0] / chunk_sample_rates[index],
        )
        ordered_chunks = [chunks[index] for index in duration_order]
        ordered_sample_rates = [chunk_sample_rates[index] for index in duration_order]
        ordered_texts = super().transcribe_waveforms(ordered_chunks, ordered_sample_rates)
        chunk_texts = [""] * len(chunks)
        for original_index, text in zip(duration_order, ordered_texts, strict=True):
            chunk_texts[original_index] = text
        if not requires_merge:
            texts = [""] * len(waveforms)
            for text, owner in zip(chunk_texts, owners, strict=True):
                texts[owner] = text
            return texts
        return merge_chunk_texts(chunk_texts, owners, len(waveforms))

    def teardown(self) -> None:
        if self._trt_encoder is not None:
            self._trt_encoder.close()
            self._trt_encoder = None
        self._chunk_duration_sec = None
        super().teardown()
