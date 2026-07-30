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
from unittest.mock import patch

import numpy as np
import pytest
import torch

from nemo_curator.stages.audio.inference.indic_conformer_hybrid import (
    _MAX_CHUNK_DURATION_SEC,
    IndicConformerHybridASR,
    InferenceIndicConformerHybridStage,
)
from nemo_curator.stages.audio.inference.indic_conformer_tensorrt import load_engine_metadata


def _metadata() -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_type": "indic_conformer_hybrid",
        "precision": "fp16",
        "engine_file": "encoder.plan",
        "model_file": "model.nemo",
        "sample_rate": 16000,
        "feature_count": 80,
        "subsampling_factor": 4,
        "encoder_dim": 512,
        "input_names": ["audio_signal", "length"],
        "output_names": ["outputs", "encoded_lengths"],
        "profile": {
            "min": {"batch": 1, "feature_frames": 8},
            "opt": {"batch": 2, "feature_frames": 800},
            "max": {"batch": 2, "feature_frames": 3000},
        },
    }


def _engine_bundle(tmp_path: Path) -> Path:
    (tmp_path / "encoder.plan").touch()
    (tmp_path / "model.nemo").touch()
    (tmp_path / "metadata.json").write_text(json.dumps(_metadata()))
    return tmp_path


def test_load_engine_metadata(tmp_path: Path) -> None:
    assert load_engine_metadata(_engine_bundle(tmp_path))["encoder_dim"] == 512


def test_load_engine_metadata_rejects_wrong_model_type(tmp_path: Path) -> None:
    metadata = _metadata()
    metadata["model_type"] = "indic_parakeet_rnnt"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="model type"):
        load_engine_metadata(tmp_path)


def test_stage_preserves_nemo_default() -> None:
    stage = InferenceIndicConformerHybridStage()
    model = stage._create_model()
    assert stage.backend == "nemo"
    assert model.tensorrt_engine_dir is None
    assert model.rnnt_precision == "fp32"


def test_stage_rejects_unknown_rnnt_precision() -> None:
    with pytest.raises(ValueError, match="RNNT precision"):
        InferenceIndicConformerHybridStage(rnnt_precision="bf16")


def test_stage_requires_engine_directory() -> None:
    with pytest.raises(ValueError, match="tensorrt_engine_dir"):
        InferenceIndicConformerHybridStage(backend="tensorrt")


def test_stage_creates_tensorrt_model(tmp_path: Path) -> None:
    stage = InferenceIndicConformerHybridStage(
        backend="tensorrt",
        tensorrt_engine_dir=str(tmp_path),
    )
    model = stage._create_model()
    assert model.tensorrt_engine_dir == str(tmp_path)


def test_tensorrt_path_batches_encoder_and_preserves_order() -> None:
    model = IndicConformerHybridASR("unused.nemo", decode_mode="ctc")
    model._device = torch.device("cpu")
    model._trt_encoder = object()
    model._trt_metadata = _metadata()
    model._chunk_duration_sec = 30.0
    encoder_batch_sizes: list[int] = []

    def preprocessor(*, input_signal: torch.Tensor, length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = input_signal.shape[0]
        return torch.ones(batch, 80, 4), torch.full((batch,), 4, dtype=length.dtype)

    def encoder(*, audio_signal: torch.Tensor, length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert audio_signal.shape[-1] == 4
        assert audio_signal.dtype == torch.float16
        encoder_batch_sizes.append(audio_signal.shape[0])
        batch = audio_signal.shape[0]
        return torch.ones(batch, 512, 2, dtype=torch.float16), torch.full((batch,), 2, dtype=length.dtype)

    model._model = SimpleNamespace(preprocessor=preprocessor, encoder=encoder)
    waveforms = [
        np.ones(160, dtype=np.float32),
        np.ones(80, dtype=np.float32),
        np.array([], dtype=np.float32),
        np.ones(120, dtype=np.float32),
    ]
    with patch.object(
        model,
        "_decode_ctc_batch",
        side_effect=lambda _encoded, _length, langs: [f"text-{lang}" for lang in langs],
    ) as decode_batch:
        texts, languages = model.generate(waveforms, [16000] * 4, ["hi", "ta", "bn", "mr"])

    assert encoder_batch_sizes == [3]
    assert [call.args[2] for call in decode_batch.call_args_list] == [["hi", "ta", "mr"]]
    assert texts == ["text-hi", "text-ta", "", "text-mr"]
    assert languages == ["hi", "ta", "bn", "mr"]


def test_indic_conformer_chunks_and_merges_before_inference() -> None:
    model = IndicConformerHybridASR("unused.nemo", decode_mode="ctc")
    model._model = object()
    model._chunk_duration_sec = 2.0
    waveform = np.zeros(5, dtype=np.float32)

    with patch.object(
        model,
        "_generate_chunks",
        return_value=(["first", "second", "third"], ["hi", "hi", "hi"]),
    ) as generate_chunks:
        texts, languages = model.generate([waveform], [1], ["hi"])

    chunks = generate_chunks.call_args.args[0]
    assert [chunk.shape[0] for chunk in chunks] == [2, 2, 1]
    assert texts == ["first second third"]
    assert languages == ["hi"]


def test_indic_conformer_only_chunks_audio_over_40_seconds() -> None:
    model = IndicConformerHybridASR("unused.nemo", decode_mode="ctc")
    model._model = object()
    model._chunk_duration_sec = _MAX_CHUNK_DURATION_SEC
    waveforms = [
        np.zeros(40, dtype=np.float32),
        np.zeros(45, dtype=np.float32),
    ]

    with patch.object(
        model,
        "_generate_chunks",
        return_value=(["whole", "first", "second"], ["hi", "hi", "hi"]),
    ) as generate_chunks:
        texts, languages = model.generate(waveforms, [1, 1], ["hi", "hi"])

    chunks = generate_chunks.call_args.args[0]
    assert [chunk.shape[0] for chunk in chunks] == [40, 40, 5]
    assert texts == ["whole", "first second"]
    assert languages == ["hi", "hi"]


def test_tensorrt_path_respects_inference_batch_size() -> None:
    model = IndicConformerHybridASR("unused.nemo", decode_mode="ctc", inference_batch_size=1)
    model._device = torch.device("cpu")
    model._trt_encoder = object()
    model._trt_metadata = _metadata()
    model._chunk_duration_sec = 30.0
    encoder_batch_sizes: list[int] = []

    def preprocessor(*, input_signal: torch.Tensor, length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = input_signal.shape[0]
        return torch.ones(batch, 80, 8), torch.full((batch,), 8, dtype=length.dtype)

    def encoder(*, audio_signal: torch.Tensor, length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_batch_sizes.append(audio_signal.shape[0])
        batch = audio_signal.shape[0]
        return torch.ones(batch, 512, 2), torch.full((batch,), 2, dtype=length.dtype)

    model._model = SimpleNamespace(preprocessor=preprocessor, encoder=encoder)
    with patch.object(model, "_decode_ctc_batch", side_effect=lambda _encoded, _length, langs: langs):
        model.generate(
            [np.ones(80, dtype=np.float32), np.ones(100, dtype=np.float32)],
            [16000, 16000],
            ["hi", "ta"],
        )

    assert encoder_batch_sizes == [1, 1]
