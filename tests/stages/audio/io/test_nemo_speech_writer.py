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

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nemo_curator.stages.audio.io.nemo_speech_writer import NeMoSpeechWriterStage
from nemo_curator.tasks import AudioTask


class TestNeMoSpeechWriterStage:
    def test_writes_matching_opus_and_manifest_line(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "out"
        stage = NeMoSpeechWriterStage(output_dir=str(output_dir), writer_concurrency=1)
        stage.setup()

        waveform = np.zeros(16000, dtype=np.float32)
        task = AudioTask(
            task_id="clip_0",
            dataset_name="test",
            data={
                "waveform": waveform,
                "sample_rate": 16000,
                "duration_sec": 1.0,
                "original_file": "s3://bucket/audio/clip_0.wav",
                "start_ms": 500,
            },
            _metadata={"_shard_key": "shard_a", "_shard_total": 1},
        )

        stage.process(task)

        opus_files = list(output_dir.rglob("*.opus"))
        manifest_path = output_dir / "shard_a.jsonl"
        assert len(opus_files) == 1
        assert manifest_path.is_file()
        lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["audio_filepath"].endswith(".opus")
        assert Path(output_dir / entry["audio_filepath"]).is_file()

    def test_vad_empty_writes_manifest_without_opus(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "out"
        stage = NeMoSpeechWriterStage(output_dir=str(output_dir), writer_concurrency=1)
        stage.setup()

        task = AudioTask(
            task_id="clip_empty",
            dataset_name="test",
            data={
                "vad_empty": True,
                "duration_sec": 120.0,
                "original_file": "s3://bucket/audio/silent.wav",
            },
            _metadata={"_shard_key": "shard_a", "_shard_total": 1},
        )

        stage.process(task)

        assert list(output_dir.rglob("*.opus")) == []
        manifest_path = output_dir / "shard_a.jsonl"
        entry = json.loads(manifest_path.read_text(encoding="utf-8").strip())
        assert entry["vad_empty"] is True
        assert entry["audio_filepath"] == ""
        assert entry["source_duration"] == 120.0
        assert (output_dir / "shard_a.jsonl.done").is_file()

    def test_read_error_writes_manifest_and_done(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "out"
        stage = NeMoSpeechWriterStage(output_dir=str(output_dir), writer_concurrency=1)
        stage.setup()

        task = AudioTask(
            task_id="clip_bad",
            dataset_name="test",
            data={
                "read_error": True,
                "original_file": "s3://bucket/audio/broken.m4a",
                "audio_filepath": "s3://bucket/audio/broken.m4a",
            },
            _metadata={"_shard_key": "shard_b", "_shard_total": 2},
        )

        stage.process(task)

        manifest_path = output_dir / "shard_b.jsonl"
        entry = json.loads(manifest_path.read_text(encoding="utf-8").strip())
        assert entry["read_error"] is True
        assert not (output_dir / "shard_b.jsonl.done").is_file()

        task2 = AudioTask(
            task_id="clip_ok",
            dataset_name="test",
            data={
                "waveform": np.zeros(16000, dtype=np.float32),
                "sample_rate": 16000,
                "duration_sec": 1.0,
                "original_file": "s3://bucket/audio/clip_ok.wav",
            },
            _metadata={"_shard_key": "shard_b", "_shard_total": 2},
        )
        stage.process(task2)
        assert (output_dir / "shard_b.jsonl.done").is_file()

    def test_shard_done_tracks_unique_sources_across_segments(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "out"
        stage = NeMoSpeechWriterStage(output_dir=str(output_dir), writer_concurrency=1)
        stage.setup()
        metadata = {"_shard_key": "shard_c", "_shard_total": 1}
        original = "s3://bucket/audio/long.wav"

        for offset in (0, 5000):
            task = AudioTask(
                task_id=f"clip_{offset}",
                dataset_name="test",
                data={
                    "waveform": np.zeros(8000, dtype=np.float32),
                    "sample_rate": 16000,
                    "duration_sec": 0.5,
                    "original_file": original,
                    "start_ms": offset,
                },
                _metadata=metadata,
            )
            stage.process(task)

        assert (output_dir / "shard_c.jsonl.done").is_file()
        assert len((output_dir / "shard_c.jsonl").read_text(encoding="utf-8").strip().splitlines()) == 2
