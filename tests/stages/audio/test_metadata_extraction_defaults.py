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

import runpy
from pathlib import Path

from nemo_curator.stages.audio.inference.sortformer import InferenceSortformerStage


def test_original_metadata_extraction_defaults_are_preserved() -> None:
    repository = Path(__file__).parents[3]
    script = repository / "examples/audio/metadata_extraction/run_metadata_extraction.py"
    parser = runpy.run_path(str(script))["_build_arg_parser"]()
    args = parser.parse_args(["--output_dir", "output"])

    assert args.vad_backend == "torch"
    assert args.vad_stage_batch_size == 1
    assert args.sortformer_batch_size == 1
    assert args.sortformer_stage_batch_size == 2
    assert args.read_concurrency == 2
    assert args.writer_concurrency == 1
    assert InferenceSortformerStage().inference_batch_size == 1
