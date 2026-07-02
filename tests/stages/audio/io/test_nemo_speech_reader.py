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

from nemo_curator.stages.audio.io.nemo_speech_reader import _dedup_entries_by_stem


class TestDedupEntriesByStem:
    def test_keeps_preferred_format_for_same_stem(self) -> None:
        entries = [
            {"audio_filepath": "s3://b/audios/vid1.wav"},
            {"audio_filepath": "s3://b/audios/vid1.opus"},
        ]
        result = _dedup_entries_by_stem(entries, "shard")
        assert len(result) == 1
        # .opus outranks .wav
        assert result[0]["audio_filepath"] == "s3://b/audios/vid1.opus"

    def test_preserves_order_and_distinct_stems(self) -> None:
        entries = [
            {"audio_filepath": "s3://b/audios/a.wav"},
            {"audio_filepath": "s3://b/audios/b.opus"},
            {"audio_filepath": "s3://b/audios/a.opus"},  # dup of a -> replaces, keeps a's position
            {"audio_filepath": "s3://b/audios/c.m4a"},
        ]
        result = _dedup_entries_by_stem(entries, "shard")
        paths = [e["audio_filepath"] for e in result]
        assert paths == [
            "s3://b/audios/a.opus",
            "s3://b/audios/b.opus",
            "s3://b/audios/c.m4a",
        ]

    def test_no_duplicates_is_identity(self) -> None:
        entries = [
            {"audio_filepath": "s3://b/audios/x.opus"},
            {"audio_filepath": "s3://b/audios/y.wav"},
        ]
        result = _dedup_entries_by_stem(entries, "shard")
        assert len(result) == 2

    def test_unknown_extension_ranks_last(self) -> None:
        entries = [
            {"audio_filepath": "s3://b/audios/z.xyz"},
            {"audio_filepath": "s3://b/audios/z.opus"},
        ]
        result = _dedup_entries_by_stem(entries, "shard")
        assert len(result) == 1
        assert result[0]["audio_filepath"] == "s3://b/audios/z.opus"

    def test_empty_paths_skipped(self) -> None:
        entries = [
            {"audio_filepath": ""},
            {"audio_filepath": "s3://b/audios/w.opus"},
        ]
        result = _dedup_entries_by_stem(entries, "shard")
        assert len(result) == 1
        assert result[0]["audio_filepath"] == "s3://b/audios/w.opus"
