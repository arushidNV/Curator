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

if TYPE_CHECKING:
    from collections.abc import Mapping

_ENGINE_FILENAME = "encoder.plan"
_METADATA_FILENAME = "metadata.json"
_MODEL_FILENAME = "model.nemo"
_INPUT_NAMES = {"audio_signal", "length"}
_OUTPUT_NAMES = {"outputs", "encoded_lengths"}


def _trt_dtype_to_torch(dtype: object) -> torch.dtype:
    name = str(dtype).upper()
    mappings = (
        (("FP16", "FLOAT16", ".HALF"), torch.float16),
        (("FP32", "FLOAT32", ".FLOAT"), torch.float32),
        (("INT64",), torch.int64),
        (("INT32",), torch.int32),
        (("INT8",), torch.int8),
        (("BOOL",), torch.bool),
    )
    for aliases, torch_dtype in mappings:
        if any(alias in name for alias in aliases):
            return torch_dtype
    msg = f"Unsupported TensorRT tensor dtype: {dtype!r}"
    raise TypeError(msg)


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


class TensorRTParakeetEncoderSession:
    """Persistent TensorRT execution context for the Parakeet encoder."""

    def __init__(self, engine_path: str | Path) -> None:
        if not torch.cuda.is_available():
            msg = "Indic Parakeet TensorRT inference requires CUDA"
            raise RuntimeError(msg)
        try:
            import tensorrt as trt
        except ImportError as error:
            msg = "TensorRT Python bindings are required for the TensorRT backend"
            raise RuntimeError(msg) from error

        path = Path(engine_path)
        if not path.is_file():
            msg = f"TensorRT encoder engine not found: {path}"
            raise FileNotFoundError(msg)

        self.device = torch.device("cuda")
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(path.read_bytes())
        if self._engine is None:
            msg = f"Could not deserialize TensorRT encoder engine: {path}"
            raise RuntimeError(msg)
        self._context = self._engine.create_execution_context()
        if self._context is None:
            msg = f"Could not create TensorRT encoder execution context: {path}"
            raise RuntimeError(msg)

        self._stream = torch.cuda.Stream(device=self.device)
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        self._output_buffers: dict[tuple[str, tuple[int, ...], torch.dtype], torch.Tensor] = {}

    def _prepare_input(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type != "cuda":
            tensor = tensor.to(self.device, non_blocking=True)
        expected_dtype = _trt_dtype_to_torch(self._engine.get_tensor_dtype(name))
        if tensor.dtype != expected_dtype:
            tensor = tensor.to(expected_dtype)
        return tensor.contiguous()

    def _output_buffer(self, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        if any(dimension < 0 for dimension in shape):
            msg = f"TensorRT did not resolve encoder output shape for {name!r}: {shape}"
            raise RuntimeError(msg)
        dtype = _trt_dtype_to_torch(self._engine.get_tensor_dtype(name))
        key = (name, shape, dtype)
        if key not in self._output_buffers:
            self._output_buffers[key] = torch.empty(shape, dtype=dtype, device=self.device)
        return self._output_buffers[key]

    def infer(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        missing = set(self.input_names) - set(inputs)
        if missing:
            msg = f"Missing TensorRT encoder inputs: {sorted(missing)}"
            raise KeyError(msg)
        prepared = {name: self._prepare_input(name, inputs[name]) for name in self.input_names}
        current_stream = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(current_stream)

        with torch.cuda.stream(self._stream):
            for name, tensor in prepared.items():
                if self._context.set_input_shape(name, tuple(tensor.shape)) is False:
                    msg = f"Input shape {tuple(tensor.shape)} is outside the TensorRT profile for {name!r}"
                    raise RuntimeError(msg)
                self._context.set_tensor_address(name, tensor.data_ptr())
                tensor.record_stream(self._stream)

            outputs = {}
            for name in self.output_names:
                output = self._output_buffer(name, tuple(self._context.get_tensor_shape(name)))
                self._context.set_tensor_address(name, output.data_ptr())
                outputs[name] = output
            if not self._context.execute_async_v3(self._stream.cuda_stream):
                msg = "TensorRT encoder execute_async_v3 failed"
                raise RuntimeError(msg)
            for output in outputs.values():
                output.record_stream(self._stream)

        current_stream.wait_stream(self._stream)
        return outputs

    def close(self) -> None:
        self._output_buffers.clear()
        self._context = None
        self._engine = None
        self._runtime = None


class TensorRTParakeetEncoder(torch.nn.Module):
    """Conformer encoder adapter backed by Curator's persistent TensorRT session."""

    def __init__(
        self,
        engine_path: str | Path,
        *,
        subsampling_factor: int,
        session: TensorRTParakeetEncoderSession | None = None,
    ) -> None:
        super().__init__()
        if session is None:
            session = TensorRTParakeetEncoderSession(engine_path)
        self.session = session
        self.subsampling_factor = int(subsampling_factor)

        missing_inputs = _INPUT_NAMES - set(self.session.input_names)
        if missing_inputs:
            msg = f"Indic Parakeet TensorRT engine is missing inputs: {sorted(missing_inputs)}"
            raise ValueError(msg)
        missing_outputs = _OUTPUT_NAMES - set(self.session.output_names)
        if missing_outputs:
            msg = f"Indic Parakeet TensorRT engine is missing outputs: {sorted(missing_outputs)}"
            raise ValueError(msg)

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.session.infer({"audio_signal": audio_signal, "length": length})
        return outputs["outputs"], outputs["encoded_lengths"]

    def freeze(self) -> None:
        self.eval()

    def unfreeze(self, partial: bool = False) -> None:  # noqa: ARG002
        self.eval()

    def close(self) -> None:
        self.session.close()


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

    def teardown(self) -> None:
        if self._trt_encoder is not None:
            self._trt_encoder.close()
            self._trt_encoder = None
        super().teardown()
