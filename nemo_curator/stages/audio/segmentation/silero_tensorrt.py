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

"""Batched Silero VAD inference backed by a persistent TensorRT engine."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Mapping

_WINDOW_SIZE = 512
_CONTEXT_SIZE = 64
_STATE_SIZE = 128
_INPUT_NAME = "input"
_STATE_INPUT_NAME = "state"
_OUTPUT_NAME = "output"
_STATE_OUTPUT_NAME = "stateN"


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


class TensorRTSession:
    """Persistent TensorRT execution context for PyTorch CUDA tensors."""

    def __init__(self, engine_path: str | Path) -> None:
        if not torch.cuda.is_available():
            msg = "TensorRT inference requires CUDA"
            raise RuntimeError(msg)

        path = Path(engine_path)
        if not path.is_file():
            msg = f"TensorRT engine not found: {path}"
            raise FileNotFoundError(msg)

        try:
            import tensorrt as trt
        except ImportError as error:
            msg = "TensorRT Python bindings are required for the TensorRT backend"
            raise RuntimeError(msg) from error

        self.device = torch.device("cuda")
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(path.read_bytes())
        if self._engine is None:
            msg = f"Could not deserialize TensorRT engine: {path}"
            raise RuntimeError(msg)

        self._context = self._engine.create_execution_context()
        if self._context is None:
            msg = f"Could not create TensorRT execution context: {path}"
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
        if name not in self.input_names:
            msg = f"{name!r} is not an input of this TensorRT engine"
            raise KeyError(msg)
        if tensor.device.type != "cuda":
            tensor = tensor.to(self.device, non_blocking=True)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        expected_dtype = _trt_dtype_to_torch(self._engine.get_tensor_dtype(name))
        return tensor if tensor.dtype == expected_dtype else tensor.to(expected_dtype)

    def _output_buffer(self, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        if any(dimension < 0 for dimension in shape):
            msg = f"TensorRT did not resolve output shape for {name!r}: {shape}"
            raise RuntimeError(msg)
        dtype = _trt_dtype_to_torch(self._engine.get_tensor_dtype(name))
        key = (name, shape, dtype)
        if key not in self._output_buffers:
            self._output_buffers[key] = torch.empty(shape, dtype=dtype, device=self.device)
        return self._output_buffers[key]

    def infer(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        missing = set(self.input_names) - set(inputs)
        if missing:
            msg = f"Missing TensorRT engine inputs: {sorted(missing)}"
            raise KeyError(msg)

        prepared = {name: self._prepare_input(name, inputs[name]) for name in self.input_names}
        current_stream = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(current_stream)

        with torch.cuda.stream(self._stream):
            for name, tensor in prepared.items():
                shape_status = self._context.set_input_shape(name, tuple(tensor.shape))
                if shape_status is False:
                    msg = f"Input shape {tuple(tensor.shape)} is outside the profile for {name!r}"
                    raise RuntimeError(msg)
                self._context.set_tensor_address(name, tensor.data_ptr())
                tensor.record_stream(self._stream)

            outputs = {}
            for name in self.output_names:
                output = self._output_buffer(name, tuple(self._context.get_tensor_shape(name)))
                self._context.set_tensor_address(name, output.data_ptr())
                outputs[name] = output

            if not self._context.execute_async_v3(self._stream.cuda_stream):
                msg = "TensorRT execute_async_v3 failed"
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


class TensorRTSileroModel:
    """Persistent 16 kHz Silero engine with Riva-style recurrent batching.

    The engine consumes a 576-sample tensor (64 samples of left context plus
    512 new samples) and batch-first recurrent state. ``infer_probabilities``
    advances many independent recordings in lockstep and transfers the final
    probability matrix to CPU only once.
    """

    def __init__(self, engine_path: str) -> None:
        self.session = TensorRTSession(engine_path)

        required = {_INPUT_NAME, _STATE_INPUT_NAME}
        missing = required - set(self.session.input_names)
        if missing:
            message = f"Silero TensorRT engine is missing inputs: {sorted(missing)}"
            raise ValueError(message)
        required_outputs = {_OUTPUT_NAME, _STATE_OUTPUT_NAME}
        missing_outputs = required_outputs - set(self.session.output_names)
        if missing_outputs:
            message = f"Silero TensorRT engine is missing outputs: {sorted(missing_outputs)}"
            raise ValueError(message)

    def infer_step(self, model_input: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute one 576-sample recurrent step for an active recording batch."""
        if model_input.ndim != 2 or model_input.shape[1] != _CONTEXT_SIZE + _WINDOW_SIZE:  # noqa: PLR2004
            message = f"Silero TensorRT input must have shape [batch, 576], got {tuple(model_input.shape)}"
            raise ValueError(message)
        expected_state = (model_input.shape[0], 2, _STATE_SIZE)
        if tuple(state.shape) != expected_state:
            message = f"Silero TensorRT state must have shape {expected_state}, got {tuple(state.shape)}"
            raise ValueError(message)
        outputs = self.session.infer(
            {
                _INPUT_NAME: model_input.contiguous(),
                _STATE_INPUT_NAME: state.contiguous(),
            }
        )
        return outputs[_OUTPUT_NAME], outputs[_STATE_OUTPUT_NAME]

    def infer_probabilities(self, waveforms: list[torch.Tensor]) -> list[torch.Tensor]:
        """Infer frame probabilities for independent 16 kHz recordings as one batch."""
        if not waveforms:
            return []
        device = self.session.device
        prepared = [
            waveform.reshape(-1).to(device=device, dtype=torch.float32, non_blocking=True) for waveform in waveforms
        ]
        lengths = [int(waveform.numel()) for waveform in prepared]
        steps = [(length + _WINDOW_SIZE - 1) // _WINDOW_SIZE for length in lengths]
        max_steps = max(steps, default=0)
        if max_steps == 0:
            return [torch.empty(0, dtype=torch.float32) for _ in prepared]

        batch_size = len(prepared)
        state = torch.zeros((batch_size, 2, _STATE_SIZE), dtype=torch.float32, device=device)
        context = torch.zeros((batch_size, _CONTEXT_SIZE), dtype=torch.float32, device=device)
        probabilities = torch.zeros((batch_size, max_steps), dtype=torch.float32, device=device)

        for step in range(max_steps):
            active = [index for index, count in enumerate(steps) if step < count]
            active_index = torch.tensor(active, dtype=torch.int64, device=device)
            chunks = []
            start = step * _WINDOW_SIZE
            for index in active:
                chunk = prepared[index][start : start + _WINDOW_SIZE]
                if chunk.numel() < _WINDOW_SIZE:
                    chunk = torch.nn.functional.pad(chunk, (0, _WINDOW_SIZE - chunk.numel()))
                chunks.append(chunk)
            audio = torch.stack(chunks, dim=0)
            active_context = context.index_select(0, active_index)
            active_state = state.index_select(0, active_index)
            output, next_state = self.infer_step(torch.cat((active_context, audio), dim=1), active_state)
            probabilities[active_index, step] = output[:, 0]
            state.index_copy_(0, active_index, next_state)
            context.index_copy_(0, active_index, audio[:, -_CONTEXT_SIZE:])

        probabilities_cpu = probabilities.cpu()
        return [probabilities_cpu[index, :count].clone() for index, count in enumerate(steps)]

    def close(self) -> None:
        self.session.close()
