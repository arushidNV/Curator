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

"""Batched streaming Sortformer inference with TensorRT."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import soundfile
import torch

from nemo_curator.stages.audio.common import ensure_waveform_2d, load_audio_file
from nemo_curator.stages.audio.segmentation.silero_tensorrt import TensorRTSession

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

_SPEECH_THRESHOLD = 0.5
_LONG_AUDIO_SECONDS = 60 * 60
_STFT_BLOCK_SECONDS = 60 * 60
_STFT_CONTEXT_HOPS = 2


def _binarize(sequence: torch.Tensor, frame_step: float) -> torch.Tensor:
    if sequence.numel() == 0:
        return torch.empty((0, 2), device=sequence.device)
    active = sequence > _SPEECH_THRESHOLD
    padded = torch.nn.functional.pad(active.float(), (1, 1))
    transitions = padded[1:] - padded[:-1]
    starts = torch.where(transitions > _SPEECH_THRESHOLD)[0]
    ends = torch.where(transitions < -_SPEECH_THRESHOLD)[0]
    if active[-1]:
        ends[-1] = len(sequence) - 1
    start_times = starts.float() * frame_step
    end_times = ends.float() * frame_step
    valid = end_times > start_times
    if active[-1]:
        valid[-1] = end_times[-1] >= start_times[-1]
    return torch.stack((start_times[valid], end_times[valid]), dim=1)


def _load_state_module(path: Path) -> ModuleType:
    if not path.is_file():
        msg = f"Sortformer TensorRT runtime module not found: {path}"
        raise FileNotFoundError(msg)
    spec = importlib.util.spec_from_file_location("riva_sortformer_modules", path)
    if spec is None or spec.loader is None:
        msg = f"Could not load Sortformer TensorRT runtime module: {path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TensorRTSortformer:
    """Persistent TensorRT engine and streaming state manager."""

    def __init__(
        self,
        engine_path: str,
        config_path: str,
        runtime_module_path: str,
        inference_batch_size: int | None = None,
    ) -> None:
        config_file = Path(config_path)
        self.config = json.loads(config_file.read_text())
        engine_max_batch_size = int(self.config["max_batch_size"])
        self.inference_batch_size = (
            engine_max_batch_size if inference_batch_size is None else int(inference_batch_size)
        )
        if not 1 <= self.inference_batch_size <= engine_max_batch_size:
            msg = (
                f"Sortformer TensorRT inference batch size must be between 1 and "
                f"{engine_max_batch_size}, got {self.inference_batch_size}"
            )
            raise ValueError(msg)
        self.session = TensorRTSession(engine_path)
        state_module = _load_state_module(Path(runtime_module_path))
        self.modules = state_module.SortformerModules(
            spkcache_refresh_rate=int(self.config["spkcache_refresh_rate"]),
            spkcache_len=int(self.config["spkcache_len"]),
            fifo_len=int(self.config["fifo_len"]),
            fc_d_model=int(self.config["emb_dim"]),
            num_spks=int(self.config["num_speakers"]),
            dtype=torch.float32,
        )

        mel_path = Path(self.config["mel_basis"])
        if not mel_path.is_absolute():
            mel_path = config_file.parent / mel_path
        self.mel_basis = torch.from_numpy(np.load(mel_path)).to(device="cuda", dtype=torch.float32)
        self.window = torch.hann_window(
            int(self.config["win_length"]),
            periodic=False,
            dtype=torch.float32,
            device="cuda",
        )
        hop_length = int(self.config["hop_length"])
        block_samples = int(self.config["sample_rate"]) * _STFT_BLOCK_SECONDS
        self._stft_block_samples = block_samples - block_samples % hop_length

    def _prepare_waveform(self, waveform: np.ndarray | torch.Tensor, sample_rate: int) -> torch.Tensor:
        tensor = torch.as_tensor(waveform, dtype=torch.float32)
        tensor = ensure_waveform_2d(tensor).mean(dim=0)
        target_rate = int(self.config["sample_rate"])
        if sample_rate != target_rate:
            import torchaudio

            tensor = torchaudio.functional.resample(tensor, sample_rate, target_rate)
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

    def _extract_features(
        self,
        waveform: torch.Tensor,
        logical_length: int,
        frame_offset: int = 0,
    ) -> torch.Tensor:
        preemphasis = float(self.config["preemphasis"])
        hop_length = int(self.config["hop_length"])
        with torch.inference_mode():
            signal = waveform.reshape(1, -1).to(device=self.window.device, non_blocking=True)
            signal = torch.cat((signal[:, :1], signal[:, 1:] - preemphasis * signal[:, :-1]), dim=1)
            spectrum = torch.stft(
                signal,
                n_fft=int(self.config["n_fft"]),
                hop_length=hop_length,
                win_length=int(self.config["win_length"]),
                window=self.window,
                center=True,
                pad_mode="constant",
                return_complex=True,
            )
            mel = torch.log(
                torch.matmul(self.mel_basis.unsqueeze(0), spectrum.abs().square()) + self.config["log_guard"]
            )
            end = frame_offset + logical_length
            return mel[0, :, frame_offset:end].transpose(0, 1).to(device="cpu").contiguous()

    def _features(self, waveforms: list[torch.Tensor]) -> list[torch.Tensor]:
        hop_length = int(self.config["hop_length"])
        return [self._extract_features(waveform, max(1, waveform.numel() // hop_length)) for waveform in waveforms]

    def _is_long_audio(self, audio: np.ndarray | torch.Tensor | str, sample_rate: int | None) -> bool:
        if sample_rate is None:
            info = soundfile.info(str(audio))
            return info.frames > info.samplerate * _LONG_AUDIO_SECONDS
        waveform = ensure_waveform_2d(torch.as_tensor(audio))
        return waveform.shape[-1] > sample_rate * _LONG_AUDIO_SECONDS

    def _waveform_feature_blocks(self, waveform: torch.Tensor) -> Iterator[torch.Tensor]:
        hop_length = int(self.config["hop_length"])
        context_samples = _STFT_CONTEXT_HOPS * hop_length
        total_samples = waveform.numel()
        for start in range(0, total_samples, self._stft_block_samples):
            end = min(start + self._stft_block_samples, total_samples)
            logical_length = (end - start) // hop_length
            if logical_length == 0:
                continue
            read_start = max(0, start - context_samples)
            read_end = min(total_samples, end + context_samples)
            signal = waveform[read_start:read_end]
            if start < context_samples:
                signal = torch.nn.functional.pad(signal, (context_samples - start, 0))
            yield self._extract_features(signal, logical_length, _STFT_CONTEXT_HOPS)

    def _file_feature_blocks(self, path: str) -> Iterator[torch.Tensor]:
        target_rate = int(self.config["sample_rate"])
        hop_length = int(self.config["hop_length"])
        context_samples = _STFT_CONTEXT_HOPS * hop_length
        with soundfile.SoundFile(path) as audio_file:
            if audio_file.samplerate != target_rate:
                msg = (
                    f"Long Sortformer inputs must use the engine sample rate "
                    f"({target_rate} Hz), got {audio_file.samplerate} Hz for {path}"
                )
                raise ValueError(msg)
            total_samples = len(audio_file)
            for start in range(0, total_samples, self._stft_block_samples):
                end = min(start + self._stft_block_samples, total_samples)
                logical_length = (end - start) // hop_length
                if logical_length == 0:
                    continue
                read_start = max(0, start - context_samples)
                read_end = min(total_samples, end + context_samples)
                audio_file.seek(read_start)
                data = audio_file.read(read_end - read_start, dtype="float32", always_2d=True)
                signal = torch.from_numpy(data).mean(dim=1)
                if start < context_samples:
                    signal = torch.nn.functional.pad(signal, (context_samples - start, 0))
                yield self._extract_features(signal, logical_length, _STFT_CONTEXT_HOPS)

    def _streaming_feature_blocks(
        self,
        audio: np.ndarray | torch.Tensor | str,
        sample_rate: int | None,
    ) -> Iterator[torch.Tensor]:
        if sample_rate is None:
            yield from self._file_feature_blocks(str(audio))
            return
        waveform = self._prepare_waveform(audio, sample_rate)
        yield from self._waveform_feature_blocks(waveform)

    def _infer_batch(
        self,
        batch_states: list[Any],
        feature_windows: list[torch.Tensor],
        left_embeddings: list[int],
        right_embeddings: list[int],
        end_flags: list[int],
    ) -> tuple[list[Any], list[torch.Tensor]]:
        chunk_len = int(self.config["chunk_len"])
        subsampling = int(self.config["subsampling_factor"])
        emb_dim = int(self.config["emb_dim"])
        self.modules.sync_pending_compression_batched(batch_states)

        chunks = torch.zeros((len(feature_windows), chunk_len, 128), dtype=torch.float32, device="cuda")
        chunk_lengths = [min(window.shape[0], chunk_len) for window in feature_windows]
        for index, (window, length) in enumerate(zip(feature_windows, chunk_lengths, strict=True)):
            chunks[index, :length] = window[:length].to(device="cuda", non_blocking=True)
        embedding_lengths = [(length - 1) // subsampling + 1 for length in chunk_lengths]

        speaker_lengths = [state.spkcache_len_cached for state in batch_states]
        max_speaker_length = max(1, *speaker_lengths)
        outputs = self.session.infer(
            {
                "chunk": chunks,
                "chunk_lengths": torch.tensor(chunk_lengths, dtype=torch.int64, device="cuda"),
                "spkcache": torch.stack(
                    [state.spkcache[0, :max_speaker_length] for state in batch_states],
                ),
                "spkcache_lengths": torch.tensor(speaker_lengths, dtype=torch.int64, device="cuda"),
                "fifo": torch.zeros((len(feature_windows), 1, emb_dim), dtype=torch.float32, device="cuda"),
                "fifo_lengths": torch.zeros(len(feature_windows), dtype=torch.int64, device="cuda"),
            }
        )
        predictions = self.modules.apply_mask_to_preds(outputs["predictions"], outputs["pred_lengths"])
        updated_states, chunk_predictions, _ = self.modules.streaming_update_batched(
            batch_states=batch_states,
            chunk_embs=outputs["chunk_embs"],
            chunk_emb_lengths=embedding_lengths,
            preds=predictions,
            lc_list=left_embeddings,
            rc_list=right_embeddings,
            end_flags=end_flags,
        )
        probability_chunks = []
        for index, embedding_length in enumerate(embedding_lengths):
            output_length = max(0, embedding_length - left_embeddings[index] - right_embeddings[index])
            probability_chunks.append(chunk_predictions[index, :output_length].float().cpu())
        return updated_states, probability_chunks

    def _infer_probabilities(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        count = len(features)
        chunk_len = int(self.config["chunk_len"])
        center_frames = int(self.config["center_chunk_frames"])
        left_frames = int(self.config["left_context_frames"])
        right_frames = int(self.config["right_context_frames"])
        subsampling = int(self.config["subsampling_factor"])

        states = [self.modules.init_streaming_state(torch.device("cuda")) for _ in features]
        positions = [0] * count
        probabilities: list[list[torch.Tensor]] = [[] for _ in features]

        while True:
            active = [index for index in range(count) if positions[index] < features[index].shape[0]]
            if not active:
                break
            for batch_start in range(0, len(active), self.inference_batch_size):
                batch_active = active[batch_start : batch_start + self.inference_batch_size]
                batch_states = [states[index] for index in batch_active]
                feature_windows = []
                left_embeddings = []
                right_embeddings = []
                end_flags = []
                for item_index in batch_active:
                    feature = features[item_index]
                    center_start = positions[item_index]
                    center_end = min(center_start + center_frames, feature.shape[0])
                    window_start = max(0, center_start - left_frames)
                    window_end = min(feature.shape[0], center_end + right_frames)
                    chunk = feature[window_start:window_end]
                    valid_frames = min(chunk.shape[0], chunk_len)
                    feature_windows.append(chunk[:valid_frames])
                    left_embeddings.append((center_start - window_start + subsampling - 1) // subsampling)
                    right_embeddings.append((window_end - center_end + subsampling - 1) // subsampling)
                    end_flags.append(int(center_end == feature.shape[0]))
                    positions[item_index] = center_end

                updated_states, probability_chunks = self._infer_batch(
                    batch_states,
                    feature_windows,
                    left_embeddings,
                    right_embeddings,
                    end_flags,
                )
                for batch_index, item_index in enumerate(batch_active):
                    states[item_index] = updated_states[batch_index]
                    probabilities[item_index].append(probability_chunks[batch_index])

        num_speakers = int(self.config["num_speakers"])
        return [
            torch.cat(parts) if parts else torch.empty((0, num_speakers), dtype=torch.float32)
            for parts in probabilities
        ]

    def _infer_streaming_probabilities(self, feature_blocks: Iterator[torch.Tensor]) -> torch.Tensor:
        center_frames = int(self.config["center_chunk_frames"])
        left_frames = int(self.config["left_context_frames"])
        right_frames = int(self.config["right_context_frames"])
        subsampling = int(self.config["subsampling_factor"])
        num_speakers = int(self.config["num_speakers"])
        state = self.modules.init_streaming_state(torch.device("cuda"))
        buffer = torch.empty((0, 128), dtype=torch.float32)
        center_start = 0
        probabilities = []

        def consume(end_of_input: bool) -> None:
            nonlocal buffer, center_start, state
            while center_start < buffer.shape[0]:
                remaining = buffer.shape[0] - center_start
                if not end_of_input and remaining < center_frames + right_frames:
                    break
                center_end = min(center_start + center_frames, buffer.shape[0])
                if not end_of_input and center_end - center_start < center_frames:
                    break
                window_start = max(0, center_start - left_frames)
                window_end = min(buffer.shape[0], center_end + right_frames)
                left_embedding = (center_start - window_start + subsampling - 1) // subsampling
                right_embedding = (window_end - center_end + subsampling - 1) // subsampling
                updated_states, probability_chunks = self._infer_batch(
                    [state],
                    [buffer[window_start:window_end]],
                    [left_embedding],
                    [right_embedding],
                    [int(end_of_input and center_end == buffer.shape[0])],
                )
                state = updated_states[0]
                probabilities.append(probability_chunks[0])
                keep_start = max(0, center_end - left_frames)
                buffer = buffer[keep_start:]
                center_start = center_end - keep_start

        for feature_block in feature_blocks:
            buffer = torch.cat((buffer, feature_block))
            consume(end_of_input=False)
        consume(end_of_input=True)
        return torch.cat(probabilities) if probabilities else torch.empty((0, num_speakers), dtype=torch.float32)

    def _segments(self, probabilities: torch.Tensor) -> list[dict[str, Any]]:
        frame_step = float(self.config["output_step_ms"]) / 1000
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
        results: list[list[dict[str, Any]] | None] = [None] * len(audio)
        regular_indices = [index for index, item in enumerate(audio) if not self._is_long_audio(item, sample_rate)]
        if regular_indices:
            regular_audio = [audio[index] for index in regular_indices]
            waveforms = self._load_inputs(regular_audio, sample_rate)
            probabilities = self._infer_probabilities(self._features(waveforms))
            for index, item_probabilities in zip(regular_indices, probabilities, strict=True):
                results[index] = self._segments(item_probabilities)

        for index, item in enumerate(audio):
            if results[index] is not None:
                continue
            feature_blocks = self._streaming_feature_blocks(item, sample_rate)
            results[index] = self._segments(self._infer_streaming_probabilities(feature_blocks))
        return [result if result is not None else [] for result in results]

    def close(self) -> None:
        self.session.close()
