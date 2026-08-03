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

from unittest.mock import MagicMock

import pytest
import torch

from nemo_curator.stages.audio.inference.tensorrt_encoder import TensorRTEncoder


def _session(
    min_shape: tuple[int, int, int] = (1, 80, 8),
    max_shape: tuple[int, int, int] = (16, 80, 3000),
) -> MagicMock:
    session = MagicMock(
        input_names=["audio_signal", "length"],
        output_names=["outputs", "encoded_lengths"],
    )
    session.input_shape_range.return_value = (min_shape, (8, 80, 800), max_shape)
    return session


def test_encoder_pads_inputs_to_profile_minimum_and_discards_padding_rows() -> None:
    session = _session((4, 80, 16))
    session.infer.return_value = {
        "outputs": torch.randn(4, 512, 4),
        "encoded_lengths": torch.tensor([2, 2, 0, 0]),
    }
    encoder = TensorRTEncoder("unused.plan", subsampling_factor=8, session=session)

    outputs, encoded_lengths = encoder(torch.randn(2, 80, 8), torch.tensor([8, 7]))

    inputs = session.infer.call_args.args[0]
    assert inputs["audio_signal"].shape == (4, 80, 16)
    assert inputs["length"].tolist() == [8, 7, 16, 16]
    assert outputs.shape[0] == 2
    assert encoded_lengths.tolist() == [2, 2]


def test_encoder_rejects_input_larger_than_profile() -> None:
    encoder = TensorRTEncoder("unused.plan", subsampling_factor=8, session=_session())

    with pytest.raises(ValueError, match="exceeds profile maximum"):
        encoder(torch.randn(17, 80, 8), torch.full((17,), 8))


def test_encoder_splits_large_batches() -> None:
    session = _session(max_shape=(8, 80, 3000))
    output_buffer = torch.empty((8, 512, 2))
    length_buffer = torch.empty((8,), dtype=torch.long)

    def infer(_inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output_buffer.fill_(session.infer.call_count - 1)
        length_buffer.fill_(2)
        return {"outputs": output_buffer, "encoded_lengths": length_buffer}

    session.infer.side_effect = infer
    encoder = TensorRTEncoder(
        "unused.plan",
        subsampling_factor=8,
        max_batch_size=8,
        session=session,
    )

    outputs, encoded_lengths = encoder(torch.randn(64, 80, 8), torch.full((64,), 8))

    assert session.infer.call_count == 8
    assert [call.args[0]["audio_signal"].shape for call in session.infer.call_args_list] == [(8, 80, 8)] * 8
    assert outputs[:, 0, 0].tolist() == [float(call) for call in range(8) for _ in range(8)]
    assert encoded_lengths.tolist() == [2] * 64


def test_encoder_pads_final_split_to_profile_minimum() -> None:
    session = _session((4, 80, 8))
    session.infer.side_effect = [
        {
            "outputs": torch.full((4, 512, 2), float(call)),
            "encoded_lengths": torch.full((4,), 2),
        }
        for call in range(2)
    ]
    encoder = TensorRTEncoder(
        "unused.plan",
        subsampling_factor=8,
        max_batch_size=4,
        session=session,
    )

    outputs, encoded_lengths = encoder(torch.randn(6, 80, 8), torch.full((6,), 8))

    assert [call.args[0]["audio_signal"].shape for call in session.infer.call_args_list] == [
        (4, 80, 8),
        (4, 80, 8),
    ]
    assert outputs[:, 0, 0].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    assert encoded_lengths.tolist() == [2] * 6
