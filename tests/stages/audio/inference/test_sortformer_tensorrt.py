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

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from nemo_curator.stages.audio.inference.sortformer_tensorrt import (
    TensorRTSortformer,
    _create_state_modules,
)


def _state_config() -> dict:
    return {
        "spkcache_refresh_rate": 188,
        "spkcache_len": 264,
        "fifo_len": 0,
        "emb_dim": 4,
        "num_speakers": 8,
    }


def _cpu_feature_runtime() -> TensorRTSortformer:
    runtime = object.__new__(TensorRTSortformer)
    runtime.config = {
        "sample_rate": 16,
        "n_fft": 4,
        "hop_length": 2,
        "win_length": 4,
        "preemphasis": 0.97,
        "log_guard": 2**-24,
    }
    runtime.window = torch.hann_window(4, periodic=False)
    runtime.mel_basis = torch.tensor(
        [
            [1.0, 0.5, 0.0],
            [0.0, 0.5, 1.0],
        ],
    )
    runtime._stft_block_samples = 20
    return runtime


def test_streaming_stft_matches_full_stft_across_blocks() -> None:
    runtime = _cpu_feature_runtime()
    waveform = torch.sin(torch.arange(47, dtype=torch.float32) * 0.2)

    expected = runtime._extract_features(waveform, waveform.numel() // 2)
    actual = torch.cat(list(runtime._waveform_feature_blocks(waveform)))

    torch.testing.assert_close(actual, expected)


def test_passes_learned_silence_to_updated_runtime_module() -> None:
    class UpdatedModules:
        def __init__(self, learnable_sil_emb: torch.Tensor | None = None, **_kwargs) -> None:
            self.learnable_sil_emb = learnable_sil_emb

    learned_silence = torch.arange(4, dtype=torch.float32)
    modules = _create_state_modules(
        SimpleNamespace(SortformerModules=UpdatedModules),
        _state_config(),
        learned_silence,
    )

    assert modules.learnable_sil_emb is learned_silence


def test_applies_learned_silence_to_legacy_runtime_module() -> None:
    class LegacyModules:
        def __init__(self, **_kwargs) -> None:
            pass

    learned_silence = torch.arange(4, dtype=torch.float32)
    modules = _create_state_modules(
        SimpleNamespace(SortformerModules=LegacyModules),
        _state_config(),
        learned_silence,
    )

    actual = modules._get_silence_profile(torch.zeros((3, 2, 4)), torch.zeros((3, 2, 8)))
    torch.testing.assert_close(actual, learned_silence.expand(3, -1))


def test_streaming_inference_preserves_chunk_grid_and_context() -> None:
    runtime = object.__new__(TensorRTSortformer)
    runtime.config = {
        "center_chunk_frames": 4,
        "left_context_frames": 2,
        "right_context_frames": 1,
        "subsampling_factor": 1,
        "num_speakers": 2,
    }
    state = SimpleNamespace()
    runtime.modules = SimpleNamespace(init_streaming_state=MagicMock(return_value=state))
    calls = []

    def infer_batch(
        batch_states: list,
        windows: list[torch.Tensor],
        left_embeddings: list[int],
        right_embeddings: list[int],
        end_flags: list[int],
    ) -> tuple[list, list[torch.Tensor]]:
        calls.append(
            (
                windows[0][:, 0].tolist(),
                left_embeddings[0],
                right_embeddings[0],
                end_flags[0],
            )
        )
        output_length = windows[0].shape[0] - left_embeddings[0] - right_embeddings[0]
        return batch_states, [torch.zeros((output_length, 2))]

    runtime._infer_batch = infer_batch
    features = torch.arange(11, dtype=torch.float32).reshape(-1, 1).repeat(1, 128)

    probabilities = runtime._infer_streaming_probabilities(iter((features[:6], features[6:])))

    assert probabilities.shape == (11, 2)
    assert calls == [
        ([0.0, 1.0, 2.0, 3.0, 4.0], 0, 1, 0),
        ([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 2, 1, 0),
        ([6.0, 7.0, 8.0, 9.0, 10.0], 2, 0, 1),
    ]


def test_diarize_streams_only_long_outliers() -> None:
    runtime = object.__new__(TensorRTSortformer)
    runtime._is_long_audio = MagicMock(side_effect=lambda item, _sample_rate: item == "long.wav")
    runtime._load_inputs = MagicMock(return_value=[torch.zeros(8)])
    runtime._features = MagicMock(return_value=[torch.zeros((4, 128))])
    runtime._infer_probabilities = MagicMock(return_value=[torch.tensor([[1.0]])])
    runtime._streaming_feature_blocks = MagicMock(return_value=iter((torch.zeros((4, 128)),)))
    runtime._infer_streaming_probabilities = MagicMock(return_value=torch.tensor([[2.0]]))
    runtime._segments = MagicMock(side_effect=lambda probabilities: [float(probabilities[0, 0])])

    results = runtime.diarize(["short.wav", "long.wav"])

    assert results == [[1.0], [2.0]]
    runtime._load_inputs.assert_called_once_with(["short.wav"], None)
    runtime._streaming_feature_blocks.assert_called_once_with("long.wav", None)
