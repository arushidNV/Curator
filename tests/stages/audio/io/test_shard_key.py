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

import pytest

from nemo_curator.stages.audio.io.shard_key import derive_manifest_shard_key


class TestDeriveManifestShardKey:
    def test_corpus_in_path_once(self) -> None:
        manifest = "/data/yodas/0_from_captions/en/sharded_manifests/manifest_42.jsonl"
        assert derive_manifest_shard_key(manifest, "yodas") == "yodas/0_from_captions/en/sharded_manifests/manifest_42"

    def test_s3_bucket_matches_corpus_once(self) -> None:
        manifest = "s3://audio-riva-originals/manifests_raw/nl/manifest_000000.jsonl"
        assert derive_manifest_shard_key(manifest, "audio-riva-originals") == (
            "audio-riva-originals/manifests_raw/nl/manifest_000000"
        )

    def test_shard_key_prefix_keeps_tail_after_anchor(self) -> None:
        # Anchor (prefix's last segment) is found in the path; everything after it
        # (bucket + sharded_manifests + manifest) is preserved so buckets don't collide.
        manifest = (
            "s3://asr/datasets/final/es/youtube/v1.1_wer10_whisper/"
            "yt_mixed_2024_12_18_083625/bucket_5/sharded_manifests/manifest_000042.json"
        )
        prefix = "yt_harvested/es/youtube/v1.1_wer10_whisper/yt_mixed_2024_12_18_083625"
        key = derive_manifest_shard_key(manifest, "yt_harvested", shard_key_prefix=prefix)
        assert key == f"{prefix}/bucket_5/sharded_manifests/manifest_000042"

    def test_shard_key_prefix_different_languages_do_not_collide(self) -> None:
        base = (
            "s3://asr/datasets/final/{locale}/youtube/v1.1_wer10_whisper/"
            "yt_mixed_2024_12_18_083625/bucket_1/sharded_manifests/manifest_000042.json"
        )
        prefix_de = "yt_harvested/de/youtube/v1.1_wer10_whisper/yt_mixed_2024_12_18_083625"
        prefix_es = "yt_harvested/es/youtube/v1.1_wer10_whisper/yt_mixed_2024_12_18_083625"
        de_key = derive_manifest_shard_key(base.format(locale="de"), "yt_harvested", shard_key_prefix=prefix_de)
        es_key = derive_manifest_shard_key(base.format(locale="es"), "yt_harvested", shard_key_prefix=prefix_es)
        assert de_key != es_key
        assert de_key.startswith("yt_harvested/de/")
        assert es_key.startswith("yt_harvested/es/")

    def test_shard_key_prefix_unknown_corpus_uuid_dataset(self) -> None:
        manifest = (
            "s3://asr/datasets/final/de-DE/78e842f6-eac6-11ee-a616-03e701f9bfe1/"
            "sharded_manifests/manifest__OP_0..255_CL_.json"
        )
        catalog = "riva_de_batch"
        prefix = f"{catalog}/de-DE/78e842f6-eac6-11ee-a616-03e701f9bfe1"
        key = derive_manifest_shard_key(manifest, catalog, shard_key_prefix=prefix)
        assert key == f"{prefix}/sharded_manifests/manifest__OP_0..255_CL_"

    def test_shard_key_prefix_different_buckets_do_not_collide(self) -> None:
        prefix = "yt_harvested/es/youtube/v1.1_wer10_whisper/yt_mixed_2024_12_18_083625"
        base = "s3://asr/datasets/final/es/youtube/v1.1_wer10_whisper/yt_mixed_2024_12_18_083625"
        k1 = derive_manifest_shard_key(
            f"{base}/bucket_1/sharded_manifests/manifest_000042.json", "yt_harvested", shard_key_prefix=prefix
        )
        k5 = derive_manifest_shard_key(
            f"{base}/bucket_5/sharded_manifests/manifest_000042.json", "yt_harvested", shard_key_prefix=prefix
        )
        assert k1 != k5
        assert k1.endswith("bucket_1/sharded_manifests/manifest_000042")
        assert k5.endswith("bucket_5/sharded_manifests/manifest_000042")

    def test_shard_key_prefix_anchor_missing_falls_back_to_basename(self) -> None:
        manifest = "s3://asr/datasets/final/es/youtube/manifest_000042.json"
        key = derive_manifest_shard_key(
            manifest,
            "yt_harvested",
            shard_key_prefix="yt_harvested/es/some_dataset_dir",
        )
        assert key == "yt_harvested/es/some_dataset_dir/manifest_000042"

    def test_corpus_missing_raises_without_prefix(self) -> None:
        manifest = "s3://asr/datasets/final/es/youtube/manifest_000042.json"
        with pytest.raises(ValueError, match="not found"):
            derive_manifest_shard_key(manifest, "yt_harvested")

    def test_duplicate_corpus_raises_without_prefix(self) -> None:
        manifest = "/data/audio-riva-originals/copy/audio-riva-originals/nl/manifest_0.jsonl"
        with pytest.raises(ValueError, match="appears 2 times"):
            derive_manifest_shard_key(manifest, "audio-riva-originals")

    def test_duplicate_corpus_with_prefix_ok(self) -> None:
        manifest = "/data/audio-riva-originals/copy/audio-riva-originals/nl/manifest_0.jsonl"
        key = derive_manifest_shard_key(
            manifest, "audio-riva-originals", shard_key_prefix="audio-riva-originals/nl"
        )
        assert key == "audio-riva-originals/nl/manifest_0"
