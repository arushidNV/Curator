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

"""Silero VAD model protocol backed by a persistent TensorRT engine."""

from __future__ import annotations

import torch

from nemo_curator.utils.tensorrt_session import TensorRTSession


class TensorRTSileroModel:
    """Callable compatible with ``silero_vad.get_speech_timestamps``.

    The engine is expected to implement the standard Silero recurrent graph:
    ``input`` + ``state`` (and optionally ``sr``) -> ``output`` + ``stateN``.
    The default 64-sample context and 512-sample window match the 16 kHz Silero
    model used by Riva and by the official ``silero-vad`` package.
    """

    def __init__(  # noqa: PLR0913
        self,
        engine_path: str,
        *,
        sample_rate: int = 16000,
        context_size: int = 64,
        state_size: int = 128,
        input_name: str = "input",
        state_input_name: str = "state",
        sample_rate_input_name: str = "sr",
        output_name: str = "output",
        state_output_name: str = "stateN",
    ) -> None:
        self.session = TensorRTSession(engine_path)
        self.sample_rate = sample_rate
        self.context_size = context_size
        self.state_size = state_size
        self.input_name = input_name
        self.state_input_name = state_input_name
        self.sample_rate_input_name = sample_rate_input_name
        self.output_name = output_name
        self.state_output_name = state_output_name
        self._state: torch.Tensor | None = None
        self._context: torch.Tensor | None = None
        self._last_batch_size = 0
        self.reset_states()

        required = {self.input_name, self.state_input_name}
        missing = required - set(self.session.input_names)
        if missing:
            message = f"Silero TensorRT engine is missing inputs: {sorted(missing)}"
            raise ValueError(message)
        required_outputs = {self.output_name, self.state_output_name}
        missing_outputs = required_outputs - set(self.session.output_names)
        if missing_outputs:
            message = f"Silero TensorRT engine is missing outputs: {sorted(missing_outputs)}"
            raise ValueError(message)

    def reset_states(self, batch_size: int = 1) -> None:
        """Reset recurrent and input-context state between recordings."""
        device = self.session.device
        self._state = torch.zeros((2, batch_size, self.state_size), dtype=torch.float32, device=device)
        self._context = torch.zeros((batch_size, self.context_size), dtype=torch.float32, device=device)
        self._last_batch_size = batch_size

    def __call__(self, audio: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        if sampling_rate != self.sample_rate:
            message = f"Silero TensorRT engine expects {self.sample_rate} Hz audio, received {sampling_rate} Hz"
            raise ValueError(message)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        if audio.ndim != 2:  # noqa: PLR2004
            message = f"Silero TensorRT input must have shape [batch, samples], got {tuple(audio.shape)}"
            raise ValueError(message)

        audio = audio.to(device=self.session.device, dtype=torch.float32, non_blocking=True)
        batch_size = audio.shape[0]
        if batch_size != self._last_batch_size:
            self.reset_states(batch_size)
        context = self._context
        state = self._state
        if context is None or state is None:
            message = "Silero TensorRT state was released; create a new model after close()"
            raise RuntimeError(message)

        model_input = torch.cat((context, audio), dim=1).contiguous()
        inputs = {self.input_name: model_input, self.state_input_name: state}
        if self.sample_rate_input_name in self.session.input_names:
            inputs[self.sample_rate_input_name] = torch.tensor(
                sampling_rate, dtype=torch.int64, device=self.session.device
            )

        outputs = self.session.infer(inputs)
        # Keep recurrent input and engine output in distinct allocations.  Some
        # TensorRT engines do not permit input/output aliasing on the next call.
        state.copy_(outputs[self.state_output_name])
        self._context = audio[:, -self.context_size :]
        return outputs[self.output_name]

    def close(self) -> None:
        self._state = None
        self._context = None
        self.session.close()
