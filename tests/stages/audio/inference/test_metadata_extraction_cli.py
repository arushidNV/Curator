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

import importlib.util
from pathlib import Path

import pytest

from nemo_curator.stages.audio.inference.indic_canary_lid import IndicCanaryLangIDStage
from nemo_curator.stages.audio.inference.sed import SEDInferenceStage
from nemo_curator.stages.audio.inference.sortformer import InferenceSortformerStage
from nemo_curator.stages.audio.inference.speechbrain_langid import SpeechBrainLangIDStage
from nemo_curator.stages.audio.io.nemo_speech_reader import NeMoSpeechAudioReader
from nemo_curator.stages.audio.segmentation.vad_segmentation import VADSegmentationStage

_SCRIPT = Path(__file__).parents[4] / "examples/audio/metadata_extraction/run_metadata_extraction.py"
_SPEC = importlib.util.spec_from_file_location("run_metadata_extraction", _SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _parse(*extra: str):
    return _MODULE._build_arg_parser().parse_args(
        ["--data_config", "input.yaml", "--output_dir", "output", *extra]
    )


def test_worker_and_batch_window_wiring() -> None:
    args = _parse(
        "--sortformer_model",
        "model.nemo",
        "--sortformer_batch_size",
        "8",
        "--sortformer_num_workers",
        "3",
        "--vad_num_workers",
        "2",
        "--sed_checkpoint",
        "sed.pth",
        "--sed_num_workers",
        "4",
    )

    stages = _MODULE._build_stages(args, None)
    sortformer = next(stage for stage in stages if isinstance(stage, InferenceSortformerStage))
    vad = next(stage for stage in stages if isinstance(stage, VADSegmentationStage))
    sed = next(stage for stage in stages if isinstance(stage, SEDInferenceStage))

    assert sortformer.batch_size == 32
    assert sortformer.num_workers() == 3
    assert vad.num_workers() == 2
    assert sed.num_workers() == 4


def test_explicit_sortformer_batch_window() -> None:
    stages = _MODULE._build_stages(
        _parse("--sortformer_model", "model.nemo", "--sortformer_batch_size", "8", "--sortformer_batch_window", "20"),
        None,
    )
    sortformer = next(stage for stage in stages if isinstance(stage, InferenceSortformerStage))
    assert sortformer.batch_size == 20


def test_resampled_subtype_wiring() -> None:
    stages = _MODULE._build_stages(
        _parse("--resampled_output_dir", "resampled", "--resampled_subtype", "PCM_16"),
        None,
    )
    reader = next(stage for stage in stages if isinstance(stage, NeMoSpeechAudioReader))
    assert reader.resampled_subtype == "PCM_16"


def test_indic_lid_pipeline_wiring() -> None:
    stages = _MODULE._build_stages(
        _parse(
            "--indic",
            "--indic_canary_engine_dir",
            "engine",
            "--langid_batch_size",
            "12",
            "--langid_max_duration_sec",
            "8",
            "--langid_max_workers",
            "2",
            "--indic_canary_batch_size",
            "6",
            "--indic_canary_lid_max_duration_sec",
            "12",
            "--indic_canary_num_workers",
            "1",
        ),
        None,
    )

    primary = next(stage for stage in stages if isinstance(stage, SpeechBrainLangIDStage))
    canary = next(stage for stage in stages if isinstance(stage, IndicCanaryLangIDStage))

    assert primary.batch_size == 12
    assert primary.max_duration_sec == 8
    assert primary.num_workers() == 2
    assert canary.batch_size == 6
    assert canary.max_duration_sec == 12
    assert canary.num_workers() == 1


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--sortformer_model", "model.nemo", "--sortformer_batch_window", "0"), "must be positive"),
        (("--indic", "--skip_langid"), "cannot be combined"),
        (
            ("--indic", "--indic_canary_engine_dir", "engine", "--indic_canary_lid_max_duration_sec", "0"),
            "must be positive",
        ),
    ],
)
def test_invalid_stage_controls(extra: tuple[str, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _MODULE._build_stages(_parse(*extra), None)
