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

from typing import ClassVar

import pytest

from nemo_curator.stages.audio.io.nemo_speech_reader import (
    NeMoSpeechAudioReader,
    NeMoSpeechDiscoveryStage,
    _dedup_entries_by_stem,
    _load_input_cfg,
    _parse_input_cfg,
)


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

    def test_keeps_distinct_recordings_sharing_basename_across_dirs(self) -> None:
        # Different directories, same basename -> genuinely distinct recordings.
        # Basename-only dedup would drop one; directory-aware dedup keeps both.
        entries = [
            {"audio_filepath": "s3://b/set_a/utt_001.wav"},
            {"audio_filepath": "s3://b/set_b/utt_001.wav"},
        ]
        result = _dedup_entries_by_stem(entries, "shard")
        paths = [e["audio_filepath"] for e in result]
        assert paths == ["s3://b/set_a/utt_001.wav", "s3://b/set_b/utt_001.wav"]

    def test_same_dir_same_stem_still_dedups(self) -> None:
        entries = [
            {"audio_filepath": "s3://b/set_a/utt_001.wav"},
            {"audio_filepath": "s3://b/set_a/utt_001.opus"},
        ]
        result = _dedup_entries_by_stem(entries, "shard")
        assert len(result) == 1
        assert result[0]["audio_filepath"] == "s3://b/set_a/utt_001.opus"


class TestInlineInputCfg:
    """The reader accepts an inline ``input_cfg`` so no separate wrapper YAML is needed."""

    _INLINE: ClassVar[list] = [
        {
            "input_cfg": [
                {"corpus": "hi", "language": "hi", "type": "nemo", "manifest_filepath": "/data/hi/m.jsonl"},
            ]
        }
    ]

    def test_load_from_yaml_file(self, tmp_path) -> None:  # noqa: ANN001
        import yaml

        p = tmp_path / "data_config.yaml"
        p.write_text(yaml.safe_dump(self._INLINE), encoding="utf-8")
        assert _load_input_cfg(str(p), None) == self._INLINE

    def test_inline_takes_precedence_over_yaml(self) -> None:
        # yaml_path is a bogus path; inline is used, so no file read happens.
        assert _load_input_cfg("/does/not/exist.yaml", self._INLINE) == self._INLINE

    def test_neither_source_raises(self) -> None:
        with pytest.raises(ValueError, match="input_cfg or yaml_path"):
            _load_input_cfg(None, None)

    def test_parse_inline_produces_shard_descriptor(self) -> None:
        shards = _parse_input_cfg(self._INLINE, corpus_filter=None)
        assert shards == [{"corpus": "hi", "manifest_path": "/data/hi/m.jsonl", "language": "hi"}]

    def test_parse_rejects_non_list(self) -> None:
        with pytest.raises(ValueError, match="input_cfg list"):
            _parse_input_cfg({"not": "a list"}, corpus_filter=None)

    def test_omegaconf_interpolation_is_resolved(self) -> None:
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "input_manifest": "/data/hi/m.jsonl",
                "data_config": [
                    {"input_cfg": [{"corpus": "hi", "language": "hi", "manifest_filepath": "${input_manifest}"}]}
                ],
            }
        )
        resolved = _load_input_cfg(None, cfg.data_config)
        shards = _parse_input_cfg(resolved, corpus_filter=None)
        assert shards[0]["manifest_path"] == "/data/hi/m.jsonl"

    def test_flat_cfg_without_input_cfg_key(self) -> None:
        # A plain list of cfg dicts (no wrapping ``input_cfg`` key) is also accepted.
        flat = [{"corpus": "hi", "language": "hi", "manifest_filepath": "/data/hi/m.jsonl"}]
        shards = _parse_input_cfg(flat, corpus_filter=None)
        assert shards == [{"corpus": "hi", "manifest_path": "/data/hi/m.jsonl", "language": "hi"}]

    def test_reader_requires_a_source(self) -> None:
        with pytest.raises(ValueError, match="input_cfg or yaml_path"):
            NeMoSpeechAudioReader()

    def test_reader_accepts_inline_cfg(self) -> None:
        reader = NeMoSpeechAudioReader(input_cfg=self._INLINE)
        discovery = reader.decompose()[0]
        assert isinstance(discovery, NeMoSpeechDiscoveryStage)
        assert discovery.input_cfg == self._INLINE
