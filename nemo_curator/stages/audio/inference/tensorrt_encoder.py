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

"""Shared TensorRT execution adapter for exported NeMo encoders."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Mapping

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


class TensorRTEncoderSession:
    """Persistent TensorRT execution context for an exported NeMo encoder."""

    def __init__(self, engine_path: str | Path) -> None:
        if not torch.cuda.is_available():
            msg = "TensorRT encoder inference requires CUDA"
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


class TensorRTEncoder(torch.nn.Module):
    """NeMo Conformer encoder adapter backed by a TensorRT session."""

    def __init__(
        self,
        engine_path: str | Path,
        *,
        subsampling_factor: int,
        session: TensorRTEncoderSession | None = None,
    ) -> None:
        super().__init__()
        if session is None:
            session = TensorRTEncoderSession(engine_path)
        self.session = session
        self.subsampling_factor = int(subsampling_factor)

        missing_inputs = _INPUT_NAMES - set(self.session.input_names)
        if missing_inputs:
            msg = f"TensorRT encoder is missing inputs: {sorted(missing_inputs)}"
            raise ValueError(msg)
        missing_outputs = _OUTPUT_NAMES - set(self.session.output_names)
        if missing_outputs:
            msg = f"TensorRT encoder is missing outputs: {sorted(missing_outputs)}"
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
