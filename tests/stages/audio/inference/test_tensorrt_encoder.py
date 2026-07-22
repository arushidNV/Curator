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


def _session(min_shape: tuple[int, int, int] = (1, 80, 8)) -> MagicMock:
    session = MagicMock(
        input_names=["audio_signal", "length"],
        output_names=["outputs", "encoded_lengths"],
    )
    session.input_shape_range.return_value = (min_shape, (8, 80, 800), (16, 80, 3000))
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
