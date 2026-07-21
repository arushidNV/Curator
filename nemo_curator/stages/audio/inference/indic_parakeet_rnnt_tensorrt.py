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
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

from nemo_curator.stages.audio.inference.asr_nemo import NemoASRModel
from nemo_curator.stages.audio.inference.audio_chunking import (
    merge_chunk_texts,
    model_chunk_duration,
    split_waveforms,
)
from nemo_curator.stages.audio.inference.tensorrt_encoder import TensorRTEncoder, TensorRTEncoderSession

if TYPE_CHECKING:
    import numpy as np

_ENGINE_FILENAME = "encoder.plan"
_METADATA_FILENAME = "metadata.json"
_MODEL_FILENAME = "model.nemo"
_INPUT_NAMES = {"audio_signal", "length"}
_OUTPUT_NAMES = {"outputs", "encoded_lengths"}


def load_engine_metadata(engine_dir: str | Path) -> dict[str, Any]:  # noqa: C901
    """Load and validate an Indic Parakeet RNN-T engine bundle manifest."""
    path = Path(engine_dir) / _METADATA_FILENAME
    if not path.is_file():
        msg = f"TensorRT engine metadata not found: {path}"
        raise FileNotFoundError(msg)
    try:
        metadata = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        msg = f"Could not read TensorRT engine metadata: {path}"
        raise ValueError(msg) from error

    if metadata.get("schema_version") != 1:
        msg = f"Unsupported TensorRT engine metadata schema: {metadata.get('schema_version')!r}"
        raise ValueError(msg)
    if metadata.get("model_type") != "indic_parakeet_rnnt":
        msg = f"Unsupported TensorRT engine model type: {metadata.get('model_type')!r}"
        raise ValueError(msg)
    if metadata.get("precision") != "fp16":
        msg = f"Unsupported TensorRT engine precision: {metadata.get('precision')!r}"
        raise ValueError(msg)
    if metadata.get("engine_file") != _ENGINE_FILENAME or metadata.get("model_file") != _MODEL_FILENAME:
        msg = "TensorRT engine metadata does not describe the expected bundle filenames"
        raise ValueError(msg)
    if set(metadata.get("input_names", [])) != _INPUT_NAMES:
        msg = f"Unexpected TensorRT encoder inputs: {metadata.get('input_names')!r}"
        raise ValueError(msg)
    if set(metadata.get("output_names", [])) != _OUTPUT_NAMES:
        msg = f"Unexpected TensorRT encoder outputs: {metadata.get('output_names')!r}"
        raise ValueError(msg)
    for key in ("sample_rate", "feature_count", "subsampling_factor", "vocabulary_size", "max_symbols_per_step"):
        if not isinstance(metadata.get(key), int) or metadata[key] < 1:
            msg = f"Invalid TensorRT engine metadata value for {key}: {metadata.get(key)!r}"
            raise ValueError(msg)
    return metadata


class TensorRTParakeetEncoder(TensorRTEncoder):
    """Conformer encoder adapter backed by Curator's persistent TensorRT session."""

    def __init__(
        self,
        engine_path: str | Path,
        *,
        subsampling_factor: int,
        session: TensorRTEncoderSession | None = None,
    ) -> None:
        super().__init__(engine_path, subsampling_factor=subsampling_factor, session=session)


TensorRTParakeetEncoderSession = TensorRTEncoderSession


class TensorRTParakeetRNNTModel(NemoASRModel):
    """NeMo RNN-T wrapper that replaces only the Conformer encoder with TensorRT."""

    def __init__(
        self,
        engine_dir: str | Path,
        *,
        inference_batch_size: int = 16,
    ) -> None:
        self.engine_dir = Path(engine_dir)
        self.metadata = load_engine_metadata(self.engine_dir)
        engine_path = self.engine_dir / _ENGINE_FILENAME
        model_path = self.engine_dir / _MODEL_FILENAME
        if not engine_path.is_file():
            msg = f"TensorRT encoder engine not found: {engine_path}"
            raise FileNotFoundError(msg)
        if not model_path.is_file():
            msg = f"Bundled NeMo model not found: {model_path}"
            raise FileNotFoundError(msg)

        super().__init__(model_name=str(model_path), inference_batch_size=inference_batch_size)
        self._engine_path = engine_path
        self._trt_encoder: TensorRTParakeetEncoder | None = None
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
        self._trt_encoder = TensorRTParakeetEncoder(
            self._engine_path,
            subsampling_factor=int(self.metadata["subsampling_factor"]),
        )
        max_feature_frames = self._trt_encoder.max_input_shape("audio_signal")[2]
        self._chunk_duration_sec = model_chunk_duration(self.asr_model, max_feature_frames)
        self.asr_model.encoder = self._trt_encoder
        del original_encoder
        gc.collect()
        torch.cuda.empty_cache()
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
        if self._chunk_duration_sec is None:
            msg = "Indic Parakeet chunk duration was not initialized from the model"
            raise RuntimeError(msg)

        chunks, chunk_sample_rates, owners = split_waveforms(
            waveforms,
            sample_rates,
            self._chunk_duration_sec,
        )
        if not chunks:
            return [""] * len(waveforms)
        chunk_texts = super().transcribe_waveforms(chunks, chunk_sample_rates)
        return merge_chunk_texts(chunk_texts, owners, len(waveforms))

    def teardown(self) -> None:
        if self._trt_encoder is not None:
            self._trt_encoder.close()
            self._trt_encoder = None
        self._chunk_duration_sec = None
        super().teardown()
