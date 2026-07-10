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

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nemo_curator.stages.audio.inference.sortformer_tensorrt import TensorRTSortformerRunner


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_engine_metadata_binds_engine_to_exact_checkpoint(tmp_path: Path) -> None:
    model_bytes = b"exact nemo checkpoint"
    engine_bytes = b"target h100 engine"
    model = tmp_path / "model.nemo"
    engine = tmp_path / "model.plan"
    metadata = tmp_path / "model.plan.json"
    model.write_bytes(model_bytes)
    engine.write_bytes(engine_bytes)
    metadata.write_text(json.dumps({"model_sha256": _sha256(model_bytes), "engine_sha256": _sha256(engine_bytes)}))

    TensorRTSortformerRunner._validate_metadata(str(engine), str(model), None)

    engine.write_bytes(b"different engine")
    with pytest.raises(ValueError, match="engine hash"):
        TensorRTSortformerRunner._validate_metadata(str(engine), str(model), None)


def test_nonempty_state_pads_zero_length_for_tensorrt() -> None:
    state = torch.empty((2, 0, 512))

    padded, lengths = TensorRTSortformerRunner._nonempty_state(state)

    assert padded.shape == (2, 1, 512)
    assert lengths.tolist() == [0, 0]


def test_forward_streaming_uses_engine_and_nemo_state_update() -> None:
    batch_size = 2
    state = SimpleNamespace(
        spkcache=torch.empty((batch_size, 0, 512)),
        fifo=torch.empty((batch_size, 0, 512)),
    )
    modules = MagicMock()
    modules.init_streaming_state.return_value = state
    chunk = torch.zeros((batch_size, 24, 128))
    chunk_lengths = torch.tensor([24, 20])
    modules.streaming_feat_loader.return_value = [(0, chunk, chunk_lengths, 8, 8)]
    modules.apply_mask_to_preds.side_effect = lambda predictions, _lengths: predictions
    expected = torch.ones((batch_size, 2, 4))
    modules.streaming_update.return_value = (state, expected)
    modules.n_spk = 4
    modules.chunk_len = 340
    modules.subsampling_factor = 8

    model = SimpleNamespace(
        sortformer_modules=modules,
        encoder=SimpleNamespace(subsampling_factor=8),
        device=torch.device("cpu"),
    )
    session = MagicMock()
    session.infer.return_value = {
        "predictions": torch.zeros((batch_size, 4, 4)),
        "pred_lengths": torch.tensor([4, 4]),
        "chunk_embs": torch.zeros((batch_size, 4, 512)),
        "chunk_emb_lengths": torch.tensor([4, 3]),
    }
    runner = object.__new__(TensorRTSortformerRunner)
    runner.model = model
    runner.session = session

    result = runner.forward_streaming(torch.zeros((batch_size, 128, 24)), chunk_lengths)

    assert torch.equal(result, expected)
    engine_inputs = session.infer.call_args.args[0]
    assert engine_inputs["chunk"].shape == (batch_size, 24, 128)
    assert engine_inputs["spkcache"].shape == (batch_size, 1, 512)
    assert engine_inputs["spkcache_lengths"].tolist() == [0, 0]
    modules.streaming_update.assert_called_once()
