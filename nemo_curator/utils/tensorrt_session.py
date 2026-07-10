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

"""Small, persistent TensorRT session for PyTorch CUDA tensors.

TensorRT is imported lazily so importing NeMo Curator does not make it a hard
dependency.  The session deliberately keeps the engine, execution context,
CUDA stream, and output allocations alive across calls.  Inputs and outputs
stay on the GPU; there is no PyCUDA or host staging buffer in the hot path.
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Mapping


def _trt_dtype_to_torch(dtype: object) -> torch.dtype:
    """Map a TensorRT data type without depending on a particular TRT release."""
    name = str(dtype).upper()
    mappings = (
        (("BF16", "BFLOAT16"), torch.bfloat16),
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
    message = f"Unsupported TensorRT tensor dtype: {dtype!r}"
    raise TypeError(message)


class TensorRTSession:
    """Execute one TensorRT engine repeatedly with persistent CUDA resources.

    Returned tensors reference reusable buffers.  A caller that needs to retain
    an output after the next :meth:`infer` call must clone it.
    """

    def __init__(
        self,
        engine_path: str,
        *,
        device: torch.device | str = "cuda",
        stream: torch.cuda.Stream | None = None,
        max_cached_shapes_per_output: int = 8,
    ) -> None:
        if max_cached_shapes_per_output < 1:
            message = "max_cached_shapes_per_output must be at least 1"
            raise ValueError(message)
        if not torch.cuda.is_available():
            message = "TensorRT inference requires CUDA"
            raise RuntimeError(message)

        path = Path(engine_path)
        if not path.is_file():
            message = f"TensorRT engine not found: {path}"
            raise FileNotFoundError(message)

        try:
            import tensorrt as trt
        except ImportError as e:
            msg = "TensorRT Python bindings are required for the TensorRT backend"
            raise RuntimeError(msg) from e

        self._trt = trt
        self.device = torch.device(device)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with path.open("rb") as engine_file:
            self.engine = self.runtime.deserialize_cuda_engine(engine_file.read())
        if self.engine is None:
            message = f"Could not deserialize TensorRT engine: {path}"
            raise RuntimeError(message)

        self.context = self.engine.create_execution_context()
        if self.context is None:
            message = f"Could not create TensorRT execution context: {path}"
            raise RuntimeError(message)

        self.stream = stream or torch.cuda.Stream(device=self.device)
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        self.max_cached_shapes_per_output = max_cached_shapes_per_output
        self._output_cache: dict[tuple[str, tuple[int, ...], torch.dtype], torch.Tensor] = {}
        self._shape_lru: defaultdict[str, deque[tuple[str, tuple[int, ...], torch.dtype]]] = defaultdict(deque)
        self.inference_count = 0

    def _validate_input(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if name not in self.input_names:
            message = f"{name!r} is not an input of this TensorRT engine"
            raise KeyError(message)
        if tensor.device.type != "cuda":
            tensor = tensor.to(self.device, non_blocking=True)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        expected_dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
        if tensor.dtype != expected_dtype:
            tensor = tensor.to(expected_dtype)
        return tensor

    def _output_buffer(self, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        if any(dimension < 0 for dimension in shape):
            message = f"TensorRT did not resolve output shape for {name!r}: {shape}"
            raise RuntimeError(message)
        dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
        key = (name, shape, dtype)
        cached = self._output_cache.get(key)
        if cached is not None:
            lru = self._shape_lru[name]
            lru.remove(key)
            lru.append(key)
            return cached

        output = torch.empty(shape, dtype=dtype, device=self.device)
        self._output_cache[key] = output
        lru = self._shape_lru[name]
        lru.append(key)
        while len(lru) > self.max_cached_shapes_per_output:
            del self._output_cache[lru.popleft()]
        return output

    def infer(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Enqueue inference and return GPU output tensors by engine tensor name."""
        missing = set(self.input_names) - set(inputs)
        if missing:
            message = f"Missing TensorRT engine inputs: {sorted(missing)}"
            raise KeyError(message)

        prepared = {name: self._validate_input(name, inputs[name]) for name in self.input_names}
        current_stream = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current_stream)

        with torch.cuda.stream(self.stream):
            for name, tensor in prepared.items():
                shape_status = self.context.set_input_shape(name, tuple(tensor.shape))
                if shape_status is False:
                    message = f"Input shape {tuple(tensor.shape)} is outside the profile for {name!r}"
                    raise RuntimeError(message)
                self.context.set_tensor_address(name, tensor.data_ptr())
                tensor.record_stream(self.stream)

            outputs: dict[str, torch.Tensor] = {}
            for name in self.output_names:
                shape = tuple(self.context.get_tensor_shape(name))
                output = self._output_buffer(name, shape)
                self.context.set_tensor_address(name, output.data_ptr())
                outputs[name] = output

            if not self.context.execute_async_v3(self.stream.cuda_stream):
                message = "TensorRT execute_async_v3 failed"
                raise RuntimeError(message)
            self.inference_count += 1
            for output in outputs.values():
                output.record_stream(self.stream)

        current_stream.wait_stream(self.stream)
        return outputs

    @property
    def cached_output_buffer_count(self) -> int:
        """Number of currently retained output allocations (for diagnostics/tests)."""
        return len(self._output_cache)

    def close(self) -> None:
        """Release TensorRT objects and reusable buffers."""
        self._output_cache.clear()
        self._shape_lru.clear()
        self.context = None
        self.engine = None
        self.runtime = None
