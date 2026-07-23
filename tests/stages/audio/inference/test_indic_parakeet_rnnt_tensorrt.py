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

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from nemo_curator.stages.audio.inference.asr_nemo import NemoASRModel
from nemo_curator.stages.audio.inference.indic_parakeet_rnnt_tensorrt import (
    TensorRTParakeetRNNTModel,
    load_engine_metadata,
)
from nemo_curator.stages.audio.inference.parakeet import InferenceParakeetStage
from nemo_curator.stages.audio.inference.tensorrt_encoder import TensorRTEncoder


def _metadata() -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_type": "indic_parakeet_rnnt",
        "precision": "fp16",
        "engine_file": "encoder.plan",
        "model_file": "model.nemo",
        "sample_rate": 16000,
        "feature_count": 80,
        "subsampling_factor": 8,
        "vocabulary_size": 958,
        "max_symbols_per_step": 10,
        "input_names": ["audio_signal", "length"],
        "output_names": ["outputs", "encoded_lengths"],
        "profile": {
            "min": {"batch": 1, "feature_frames": 8},
            "opt": {"batch": 8, "feature_frames": 800},
            "max": {"batch": 16, "feature_frames": 3000},
        },
    }


def _engine_bundle(tmp_path: Path) -> Path:
    (tmp_path / "encoder.plan").touch()
    (tmp_path / "model.nemo").touch()
    (tmp_path / "metadata.json").write_text(json.dumps(_metadata()))
    return tmp_path


def test_load_engine_metadata(tmp_path: Path) -> None:
    engine_dir = _engine_bundle(tmp_path)
    assert load_engine_metadata(engine_dir)["subsampling_factor"] == 8


def test_load_engine_metadata_rejects_wrong_model_type(tmp_path: Path) -> None:
    metadata = _metadata()
    metadata["model_type"] = "canary"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="model type"):
        load_engine_metadata(tmp_path)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("input_names", None, "input names"),
        ("profile", {"min": None, "opt": {}, "max": {}}, "profile points"),
    ],
)
def test_load_engine_metadata_rejects_malformed_fields(
    tmp_path: Path,
    key: str,
    value: object,
    message: str,
) -> None:
    metadata = _metadata()
    metadata[key] = value
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))

    with pytest.raises((TypeError, ValueError), match=message):
        load_engine_metadata(tmp_path)


def test_tensorrt_encoder_forwards_one_batch() -> None:
    expected_outputs = torch.randn(2, 1024, 4)
    expected_lengths = torch.tensor([4, 3])
    session = MagicMock(
        input_names=["audio_signal", "length"],
        output_names=["outputs", "encoded_lengths"],
    )
    session.input_shape_range.return_value = ((1, 80, 8), (8, 80, 800), (16, 80, 3000))
    session.infer.return_value = {
        "outputs": expected_outputs,
        "encoded_lengths": expected_lengths,
    }
    encoder = TensorRTEncoder("unused.plan", subsampling_factor=8, session=session)
    audio_signal = torch.randn(2, 80, 32)
    length = torch.tensor([32, 24])

    outputs, encoded_lengths = encoder(audio_signal, length)

    session.infer.assert_called_once()
    inputs = session.infer.call_args.args[0]
    assert inputs["audio_signal"] is audio_signal
    assert inputs["length"] is length
    assert outputs is expected_outputs
    assert encoded_lengths is expected_lengths
    assert encoder.subsampling_factor == 8
    encoder.train()
    encoder.freeze()
    assert encoder.training is False
    encoder.train()
    encoder.unfreeze(partial=True)
    assert encoder.training is False


def test_tensorrt_wrapper_enables_batched_greedy_decoder(tmp_path: Path) -> None:
    wrapper = TensorRTParakeetRNNTModel(_engine_bundle(tmp_path))
    model = SimpleNamespace(
        cfg=OmegaConf.create({"decoding": {"strategy": "greedy", "greedy": {}}}),
        change_decoding_strategy=MagicMock(),
    )
    wrapper.asr_model = model

    wrapper._enable_batched_greedy_decoder()

    assert model.cfg.decoding.strategy == "greedy_batch"
    assert model.cfg.decoding.greedy.max_symbols_per_step == 10
    assert model.cfg.decoding.greedy.use_cuda_graph_decoder is True
    model.change_decoding_strategy.assert_called_once_with(model.cfg.decoding)


def test_tensorrt_wrapper_replaces_only_encoder(tmp_path: Path) -> None:
    wrapper = TensorRTParakeetRNNTModel(_engine_bundle(tmp_path))
    original_encoder = SimpleNamespace(subsampling_factor=8, _feat_in=80)
    model = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "encoder": {"feat_in": 80},
                "preprocessor": {"sample_rate": 16000, "window_stride": 0.01},
                "train_ds": {"max_duration": 20},
                "joint": {"num_classes": 958},
                "decoding": {"strategy": "greedy", "greedy": {}},
            }
        ),
        encoder=original_encoder,
        decoder=object(),
        joint=SimpleNamespace(_vocab_size=958),
        to=MagicMock(),
        change_decoding_strategy=MagicMock(),
    )
    wrapper.asr_model = model
    optimized_encoder = MagicMock()
    optimized_encoder.max_input_shape.return_value = (16, 80, 3000)

    with (
        patch.object(NemoASRModel, "setup"),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.empty_cache"),
        patch(
            "nemo_curator.stages.audio.inference.indic_parakeet_rnnt_tensorrt.TensorRTEncoder",
            return_value=optimized_encoder,
        ) as encoder_type,
    ):
        wrapper.setup()

    encoder_type.assert_called_once_with(tmp_path / "encoder.plan", subsampling_factor=8)
    assert model.encoder is optimized_encoder
    assert model.decoder is not None
    assert model.joint._vocab_size == 958
    model.to.assert_called_once_with(dtype=torch.float16)
    model.change_decoding_strategy.assert_called_once()
    assert wrapper._chunk_duration_sec == 20.0


def test_tensorrt_wrapper_chunks_long_audio_without_overlap(tmp_path: Path) -> None:
    wrapper = TensorRTParakeetRNNTModel(_engine_bundle(tmp_path))
    wrapper._chunk_duration_sec = 20.0
    transcribe = MagicMock(return_value=["third", "first", "second"])
    wrapper.asr_model = SimpleNamespace(transcribe=transcribe)
    waveform = np.zeros(45 * 16000, dtype=np.float32)

    texts = wrapper.transcribe_waveforms(
        [waveform, np.array([], dtype=np.float32)],
        [16000, 16000],
    )

    prepared = transcribe.call_args.args[0]
    assert [chunk.shape[0] for chunk in prepared] == [5 * 16000, 20 * 16000, 20 * 16000]
    assert texts == ["first second third", ""]


def test_tensorrt_wrapper_orders_chunks_by_duration_across_sample_rates(tmp_path: Path) -> None:
    wrapper = TensorRTParakeetRNNTModel(_engine_bundle(tmp_path))
    wrapper._chunk_duration_sec = 20.0
    transcribe = MagicMock(return_value=["second", "first"])
    wrapper.asr_model = SimpleNamespace(transcribe=transcribe)

    texts = wrapper.transcribe_waveforms(
        [
            np.zeros(8 * 16000, dtype=np.float32),
            np.zeros(6 * 48000, dtype=np.float32),
        ],
        [16000, 48000],
    )

    prepared = transcribe.call_args.args[0]
    assert [chunk.shape[0] for chunk in prepared] == [6 * 16000, 8 * 16000]
    assert texts == ["first", "second"]


def test_parakeet_stage_preserves_nemo_default() -> None:
    stage = InferenceParakeetStage()
    assert stage.backend == "nemo"
    assert isinstance(stage._create_wrapper(), NemoASRModel)


def test_parakeet_stage_requires_engine_directory() -> None:
    with pytest.raises(ValueError, match="tensorrt_engine_dir"):
        InferenceParakeetStage(backend="tensorrt")


def test_parakeet_stage_creates_tensorrt_wrapper() -> None:
    optimized_wrapper = MagicMock()
    with patch(
        "nemo_curator.stages.audio.inference.indic_parakeet_rnnt_tensorrt.TensorRTParakeetRNNTModel",
        return_value=optimized_wrapper,
    ) as wrapper_type:
        stage = InferenceParakeetStage(
            backend="tensorrt",
            tensorrt_engine_dir="/engines/indic-rnnt",
            inference_batch_size=12,
        )
        assert stage._create_wrapper() is optimized_wrapper

    wrapper_type.assert_called_once_with(
        engine_dir="/engines/indic-rnnt",
        inference_batch_size=12,
    )
