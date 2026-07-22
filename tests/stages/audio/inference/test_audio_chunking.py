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

import numpy as np

from nemo_curator.stages.audio.inference.audio_chunking import (
    merge_chunk_texts,
    model_chunk_duration,
    model_training_max_duration,
    split_waveforms,
)


def test_model_training_max_duration_reads_checkpoint_config() -> None:
    model = SimpleNamespace(cfg=SimpleNamespace(train_ds=SimpleNamespace(max_duration=20)))
    assert model_training_max_duration(model) == 20.0


def test_model_chunk_duration_respects_actual_encoder_shape() -> None:
    model = SimpleNamespace(
        cfg=SimpleNamespace(
            train_ds=SimpleNamespace(max_duration=30),
            preprocessor=SimpleNamespace(sample_rate=16000, window_stride=0.01),
        )
    )
    assert model_chunk_duration(model, max_feature_frames=3000) == 479999 / 16000


def test_split_waveforms_uses_consecutive_non_overlapping_windows() -> None:
    waveform = np.arange(45, dtype=np.float32)
    chunks, sample_rates, owners = split_waveforms(
        [waveform, np.array([], dtype=np.float32)],
        [10, 10],
        max_duration_sec=2.0,
    )

    assert [chunk.shape[0] for chunk in chunks] == [20, 20, 5]
    np.testing.assert_array_equal(np.concatenate(chunks), waveform)
    assert sample_rates == [10, 10, 10]
    assert owners == [0, 0, 0]


def test_split_waveforms_zero_pads_tiny_final_chunk() -> None:
    waveform = np.arange(21, dtype=np.float32)

    chunks, sample_rates, owners = split_waveforms([waveform], [100], max_duration_sec=0.2)

    assert [chunk.shape[0] for chunk in chunks] == [20, 10]
    np.testing.assert_array_equal(chunks[0], waveform[:20])
    np.testing.assert_array_equal(chunks[1][:1], waveform[20:])
    np.testing.assert_array_equal(chunks[1][1:], np.zeros(9, dtype=np.float32))
    assert sample_rates == [100, 100]
    assert owners == [0, 0]


def test_merge_chunk_texts_preserves_input_order_and_empty_audio() -> None:
    texts = merge_chunk_texts([" first ", "", "second", "third"], [0, 0, 0, 2], 3)
    assert texts == ["first second", "", "third"]
