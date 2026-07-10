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

import torch

from nemo_curator.utils.tensorrt_session import TensorRTSession


def probabilities_to_speech_timestamps(  # noqa: C901, PLR0912, PLR0913, PLR0915
    speech_probabilities: torch.Tensor,
    audio_length_samples: int,
    *,
    sampling_rate: int = 16000,
    threshold: float = 0.5,
    min_speech_duration_ms: float = 250,
    max_speech_duration_s: float = float("inf"),
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    neg_threshold: float | None = None,
    min_silence_at_max_speech: int = 98,
) -> list[dict[str, int]]:
    """Apply Silero 6.2.1 timestamp semantics to precomputed 16 kHz probabilities."""
    window_size_samples = 512
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = sampling_rate * max_speech_duration_s - window_size_samples - 2 * speech_pad_samples
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * min_silence_at_max_speech / 1000
    if neg_threshold is None:
        neg_threshold = max(threshold - 0.15, 0.01)

    triggered = False
    speeches: list[dict[str, int]] = []
    current_speech: dict[str, int] = {}
    temp_end = 0
    prev_end = 0
    next_start = 0
    possible_ends: list[tuple[int, int]] = []

    for index, probability in enumerate(speech_probabilities.tolist()):
        current_sample = window_size_samples * index
        if probability >= threshold and temp_end:
            silence_duration = current_sample - temp_end
            if silence_duration > min_silence_samples_at_max_speech:
                possible_ends.append((temp_end, silence_duration))
            temp_end = 0
            if next_start < prev_end:
                next_start = current_sample

        if probability >= threshold and not triggered:
            triggered = True
            current_speech["start"] = current_sample
            continue

        if triggered and current_sample - current_speech["start"] > max_speech_samples:
            if possible_ends:
                prev_end, duration = max(possible_ends, key=lambda item: item[1])
                current_speech["end"] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                next_start = prev_end + duration
                if next_start < prev_end + current_sample:
                    current_speech["start"] = next_start
                else:
                    triggered = False
                prev_end = next_start = temp_end = 0
                possible_ends = []
            else:
                current_speech["end"] = current_sample
                speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                possible_ends = []
                continue

        if probability < neg_threshold and triggered:
            if not temp_end:
                temp_end = current_sample
            if current_sample - temp_end < min_silence_samples:
                continue
            current_speech["end"] = temp_end
            if current_speech["end"] - current_speech["start"] > min_speech_samples:
                speeches.append(current_speech)
            current_speech = {}
            prev_end = next_start = temp_end = 0
            triggered = False
            possible_ends = []

    if current_speech and audio_length_samples - current_speech["start"] > min_speech_samples:
        current_speech["end"] = audio_length_samples
        speeches.append(current_speech)

    for index, speech in enumerate(speeches):
        if index == 0:
            speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
        if index != len(speeches) - 1:
            silence_duration = speeches[index + 1]["start"] - speech["end"]
            if silence_duration < 2 * speech_pad_samples:
                speech["end"] += int(silence_duration // 2)
                speeches[index + 1]["start"] = int(max(0, speeches[index + 1]["start"] - silence_duration // 2))
            else:
                speech["end"] = int(min(audio_length_samples, speech["end"] + speech_pad_samples))
                speeches[index + 1]["start"] = int(
                    max(0, speeches[index + 1]["start"] - speech_pad_samples)
                )
        else:
            speech["end"] = int(min(audio_length_samples, speech["end"] + speech_pad_samples))
    return speeches


class TensorRTSileroModel:
    """Persistent 16 kHz Silero engine with Riva-style recurrent batching.

    The engine consumes a 576-sample tensor (64 samples of left context plus
    512 new samples) and batch-first recurrent state. ``infer_probabilities``
    advances many independent recordings in lockstep and transfers the final
    probability matrix to CPU only once.
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
        output_name: str = "output",
        state_output_name: str = "stateN",
    ) -> None:
        self.session = TensorRTSession(engine_path)
        self.sample_rate = sample_rate
        self.context_size = context_size
        self.state_size = state_size
        self.input_name = input_name
        self.state_input_name = state_input_name
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
        self._state = torch.zeros((batch_size, 2, self.state_size), dtype=torch.float32, device=device)
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
        outputs = self.infer_step(model_input, state)
        # Keep recurrent input and engine output in distinct allocations.  Some
        # TensorRT engines do not permit input/output aliasing on the next call.
        state.copy_(outputs[1])
        self._context = audio[:, -self.context_size :]
        return outputs[0]

    def infer_step(self, model_input: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute one 576-sample recurrent step for an active recording batch."""
        if model_input.ndim != 2 or model_input.shape[1] != self.context_size + 512:  # noqa: PLR2004
            message = f"Silero TensorRT input must have shape [batch, 576], got {tuple(model_input.shape)}"
            raise ValueError(message)
        expected_state = (model_input.shape[0], 2, self.state_size)
        if tuple(state.shape) != expected_state:
            message = f"Silero TensorRT state must have shape {expected_state}, got {tuple(state.shape)}"
            raise ValueError(message)
        outputs = self.session.infer(
            {
                self.input_name: model_input.contiguous(),
                self.state_input_name: state.contiguous(),
            }
        )
        return outputs[self.output_name], outputs[self.state_output_name]

    def infer_probabilities(self, waveforms: list[torch.Tensor]) -> list[torch.Tensor]:
        """Infer frame probabilities for independent 16 kHz recordings as one batch."""
        if not waveforms:
            return []
        device = self.session.device
        prepared = [
            waveform.reshape(-1).to(device=device, dtype=torch.float32, non_blocking=True)
            for waveform in waveforms
        ]
        lengths = [int(waveform.numel()) for waveform in prepared]
        steps = [(length + 511) // 512 for length in lengths]
        max_steps = max(steps, default=0)
        if max_steps == 0:
            return [torch.empty(0, dtype=torch.float32) for _ in prepared]

        batch_size = len(prepared)
        state = torch.zeros((batch_size, 2, self.state_size), dtype=torch.float32, device=device)
        context = torch.zeros((batch_size, self.context_size), dtype=torch.float32, device=device)
        probabilities = torch.zeros((batch_size, max_steps), dtype=torch.float32, device=device)

        for step in range(max_steps):
            active = [index for index, count in enumerate(steps) if step < count]
            active_index = torch.tensor(active, dtype=torch.int64, device=device)
            chunks = []
            start = step * 512
            for index in active:
                chunk = prepared[index][start : start + 512]
                if chunk.numel() < 512:  # noqa: PLR2004
                    chunk = torch.nn.functional.pad(chunk, (0, 512 - chunk.numel()))
                chunks.append(chunk)
            audio = torch.stack(chunks, dim=0)
            active_context = context.index_select(0, active_index)
            active_state = state.index_select(0, active_index)
            output, next_state = self.infer_step(torch.cat((active_context, audio), dim=1), active_state)
            probabilities[active_index, step] = output[:, 0]
            state.index_copy_(0, active_index, next_state)
            context.index_copy_(0, active_index, audio[:, -self.context_size :])

        probabilities_cpu = probabilities.cpu()
        return [probabilities_cpu[index, :count].clone() for index, count in enumerate(steps)]

    def close(self) -> None:
        self._state = None
        self._context = None
        self.session.close()

    @property
    def inference_count(self) -> int:
        """Number of TensorRT enqueues completed by this model."""
        return self.session.inference_count
