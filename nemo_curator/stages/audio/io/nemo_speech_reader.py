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

"""NeMo Speech audio reader using lhotse adapters.

Reads NeMo ``input_cfg`` YAML configs (``nemo_tarred`` and ``nemo``
types) through NeMo's ``LazyNeMoIterator`` and
``LazyNeMoTarredIterator``:

    YAML (input_cfg) -> discovery (shard expansion + checkpointing)
                     -> NeMo lhotse adapter -> CutSet -> cut.load_audio() -> AudioTask

Decomposes into:
1. ``NeMoSpeechDiscoveryStage`` — parses ``input_cfg`` YAML, expands shards, checks .done
2. ``NeMoSpeechReaderStage`` — manifest -> NeMo CutSet -> AudioTask (format-agnostic)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

try:
    from nemo_curator.backends.utils import RayStageSpecKeys
except (ImportError, ModuleNotFoundError):
    try:
        from nemo_curator.backends.experimental.utils import RayStageSpecKeys
    except (ImportError, ModuleNotFoundError):
        RayStageSpecKeys = None

from nemo.collections.common.data.lhotse.nemo_adapters import expand_sharded_filepaths as _expand_nemo_path

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.tasks import AudioTask, FileGroupTask, _EmptyTask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _manifest_to_shard_key(manifest_path: str, corpus: str) -> str:
    """Derive a shard key from a manifest path starting at the corpus directory.

    Matches the logic in ``NemoTarShardDiscoveryStage._manifest_to_rel_path``:
    finds the corpus name (case-insensitive, must appear exactly once) in
    the path components and returns everything from that point onward with
    the file extension stripped.
    """
    parts = manifest_path.replace("\\", "/").split("/")
    parts_lower = [p.lower() for p in parts]
    corpus_lower = corpus.lower()
    matches = [i for i, p in enumerate(parts_lower) if p == corpus_lower]
    if len(matches) == 0:
        msg = (
            f"Corpus name '{corpus}' not found in manifest path: {manifest_path}. "
            f"The YAML 'corpus' field must match a directory component in the manifest path (case-insensitive)."
        )
        raise ValueError(msg)
    if len(matches) > 1:
        msg = (
            f"Corpus name '{corpus}' appears {len(matches)} times in manifest path: {manifest_path}. "
            f"It must appear exactly once for unambiguous path extraction."
        )
        raise ValueError(msg)
    idx = matches[0]
    rel = "/".join(parts[idx:])
    if rel.endswith(".jsonl.gz"):
        rel = rel[: -len(".jsonl.gz")]
    elif rel.endswith(".jsonl"):
        rel = rel[: -len(".jsonl")]
    elif rel.endswith(".json"):
        rel = rel[: -len(".json")]
    return rel




# ---------------------------------------------------------------------------
# YAML parsing (input_cfg format only)
# ---------------------------------------------------------------------------


def _parse_input_cfg(
    yaml_path: str,
    corpus_filter: list[str] | None,
    language_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Parse a NeMo ``input_cfg`` YAML into shard descriptors.

    Each descriptor has ``manifest_path``, optional ``tar_path``,
    ``corpus``, and ``language``.

    Only supports the standard NeMo config format with ``input_cfg``
    entries of type ``nemo_tarred`` or ``nemo``.
    """
    import yaml

    with open(yaml_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, list) or not config:
        msg = f"Expected a YAML list, got {type(config)}"
        raise ValueError(msg)

    shards: list[dict[str, Any]] = []
    for group in config:
        for cfg in group.get("input_cfg", [group]):
            corpus = cfg.get("corpus", "unknown")
            if corpus_filter and corpus not in corpus_filter:
                continue

            language = cfg.get("language", "")
            if language_filter and language not in language_filter:
                continue

            if "tarred_audio_filepaths" in cfg:
                manifest_paths = _expand_nemo_path(cfg["manifest_filepath"])
                tar_paths = _expand_nemo_path(cfg["tarred_audio_filepaths"])
                if len(manifest_paths) != len(tar_paths):
                    msg = f"Manifest/tar count mismatch for {corpus}: {len(manifest_paths)} vs {len(tar_paths)}"
                    raise ValueError(msg)
                for mp, tp in zip(manifest_paths, tar_paths, strict=False):
                    shards.append({"corpus": corpus, "manifest_path": mp, "tar_path": tp, "language": language})
            elif "manifest_filepath" in cfg:
                for mp in _expand_nemo_path(cfg["manifest_filepath"]):
                    shards.append({"corpus": corpus, "manifest_path": mp, "language": language})

    return shards


# ---------------------------------------------------------------------------
# Stage 1: Discovery
# ---------------------------------------------------------------------------


@dataclass
class NeMoSpeechDiscoveryStage(ProcessingStage[_EmptyTask, FileGroupTask]):
    """Parse ``input_cfg`` YAML and emit one ``FileGroupTask`` per shard.

    Supports NeMo ``input_cfg`` format with ``nemo_tarred`` and ``nemo``
    types.  Handles shard expansion and ``.done``-file checkpointing.
    """

    name: str = "nemo_speech_discovery"
    yaml_path: str = ""
    corpus_filter: list[str] | None = None
    language_filter: list[str] | None = None
    output_dir: str | None = None

    def __post_init__(self) -> None:
        if not self.yaml_path:
            msg = "yaml_path is required"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers_per_node": 1}

    def ray_stage_spec(self) -> dict[str, Any]:
        # Fan out per-shard tasks into one block each so the reader runs in parallel.
        if RayStageSpecKeys is not None:
            return {
                RayStageSpecKeys.IS_FANOUT_STAGE: True,
            }
        return {"is_fanout_stage": True}

    def _scan_completed_shards(self) -> set[str]:
        if not self.output_dir or not os.path.isdir(self.output_dir):
            return set()
        completed: set[str] = set()
        for root, _dirs, files in os.walk(self.output_dir):
            for fname in files:
                if fname.endswith(".jsonl.done"):
                    rel = os.path.relpath(os.path.join(root, fname), self.output_dir)
                    completed.add(rel[: -len(".jsonl.done")])
        return completed

    def process(self, _task: _EmptyTask) -> list[FileGroupTask]:
        shard_descs = _parse_input_cfg(self.yaml_path, self.corpus_filter, self.language_filter)

        completed = self._scan_completed_shards()
        if completed:
            logger.info(f"Checkpoint: {len(completed)} shards already completed, first 10: {sorted(completed)[:10]}")

        tasks: list[FileGroupTask] = []
        skipped = 0
        for desc in shard_descs:
            corpus = desc["corpus"]
            shard_key = _manifest_to_shard_key(desc["manifest_path"], corpus)
            if shard_key in completed:
                skipped += 1
                continue
            if self.output_dir:
                partial = os.path.join(self.output_dir, f"{shard_key}.jsonl")
                if os.path.exists(partial):
                    os.remove(partial)
                    logger.info(f"Removed partial output for {shard_key}")

            if "tar_path" in desc:
                tasks.append(FileGroupTask(
                    task_id=shard_key,
                    dataset_name=corpus,
                    data=[desc["manifest_path"], desc["tar_path"]],
                    reader_config={"corpus": corpus, "shard_key": shard_key, "language": desc.get("language", "")},
                ))
            else:
                # Non-tarred: read manifest, emit one task per entry for parallel loading
                import json

                from fsspec.core import url_to_fs

                try:
                    fs, resolved = url_to_fs(desc["manifest_path"])
                    with fs.open(resolved, "r", encoding="utf-8") as f:
                        entries = [json.loads(line) for line in f if line.strip()]
                    for i, entry in enumerate(entries):
                        tasks.append(FileGroupTask(
                            task_id=f"{shard_key}_{i}",
                            dataset_name=corpus,
                            data=[entry.get("audio_filepath", "")],
                            reader_config={
                                "corpus": corpus,
                                "shard_key": shard_key,
                                "language": desc.get("language", ""),
                                "entry": entry,
                                "shard_total": len(entries),
                            },
                        ))
                except Exception:  # noqa: BLE001
                    tasks.append(FileGroupTask(
                        task_id=shard_key,
                        dataset_name=corpus,
                        data=[desc["manifest_path"]],
                        reader_config={"corpus": corpus, "shard_key": shard_key, "language": desc.get("language", "")},
                    ))

        logger.info(
            f"UnifiedDiscovery: {len(tasks)} shards to process, {skipped} skipped "
            f"(corpus_filter={self.corpus_filter}, language_filter={self.language_filter})"
        )
        return tasks

    def process_batch(self, tasks: list[_EmptyTask]) -> list[FileGroupTask]:
        results: list[FileGroupTask] = []
        for task in tasks:
            results.extend(self.process(task))
        return results


# ---------------------------------------------------------------------------
# Stage 2: Reader (format-agnostic, converts CutSet -> AudioTask)
# ---------------------------------------------------------------------------


@dataclass
class NeMoSpeechReaderStage(ProcessingStage[FileGroupTask, AudioTask]):
    """Read a manifest shard and emit AudioTasks via NeMo lhotse adapters.

    Format-agnostic: uses ``LazyNeMoTarredIterator`` when a tar path
    is present, ``LazyNeMoIterator`` otherwise.  Both produce a lhotse
    ``CutSet`` iterated to load audio and emit ``AudioTask`` objects.

    The reader does not parse manifests itself — NeMo's adapters handle
    all I/O (including lazy line-by-line streaming for large files).

    When ``process_batch`` receives multiple single-entry tasks (non-tarred),
    audio files are loaded concurrently using threads to overlap S3/network
    latency. This significantly speeds up reading large files from object
    storage.

    Args:
        max_io_threads: Maximum number of concurrent I/O threads for
            loading audio files in ``process_batch``. Only applies to
            single-entry (non-tarred) tasks. Defaults to 8.
    """

    name: str = "nemo_speech_reader"
    max_io_threads: int = 8
    batch_size: int = 8
    # Max shards read in parallel. Caps in-flight waveforms so the object store
    # doesn't overflow (without it, Ray launches up to one reader task per CPU).
    read_concurrency: int = 2

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["waveform", "sampling_rate", "corpus", "num_channels"]

    def ray_stage_spec(self) -> dict[str, Any]:
        # Fan out AudioTask outputs into 1-row blocks for parallel downstream GPU
        # stages; concurrency caps how many reader tasks run at once (see read_concurrency).
        if RayStageSpecKeys is not None:
            return {
                RayStageSpecKeys.IS_FANOUT_STAGE: True,
                RayStageSpecKeys.RAY_REMOTE_ARGS: {"concurrency": self.read_concurrency},
            }
        return {
            "is_fanout_stage": True,
            "ray_remote_args": {"concurrency": self.read_concurrency},
        }

    @staticmethod
    def _make_cutset(manifest_path: str, tar_path: str | None) -> Any:  # noqa: ANN401
        """Build a lhotse CutSet using NeMo adapters."""
        from lhotse import CutSet
        from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoIterator, LazyNeMoTarredIterator

        if tar_path:
            iterator = LazyNeMoTarredIterator(
                manifest_path=manifest_path,
                tar_paths=tar_path,
                skip_missing_manifest_entries=True,
            )
            return CutSet(iterator)

        return CutSet(LazyNeMoIterator(manifest_path))

    @staticmethod
    def _load_audio(audio_path: str, hint_sr: int | None = None, hint_duration: float = 0.0) -> tuple[np.ndarray, int, float]:
        """Load audio from a file path (local or S3) and return (waveform, sr, duration).

        Tries lhotse first, falls back to smart_open + torchcodec.
        Returns a 1-D float32 numpy array.
        """
        audio: np.ndarray | None = None
        sr: int = 0
        duration: float = 0.0

        try:
            from lhotse import Recording
            from lhotse.audio import AudioSource
            from nemo.utils.data_utils import is_datastore_path

            if hint_sr:
                source_type = "url" if is_datastore_path(audio_path) else "file"
                rec = Recording(
                    id=audio_path,
                    sources=[AudioSource(type=source_type, channels=[0], source=audio_path)],
                    sampling_rate=int(hint_sr),
                    num_samples=int(hint_duration * hint_sr),
                    duration=hint_duration,
                    channel_ids=[0],
                )
            else:
                rec = Recording.from_file(audio_path)
            audio = rec.load_audio().squeeze()
            sr = rec.sampling_rate
            duration = rec.duration
        except Exception:  # noqa: BLE001
            try:
                import smart_open
                from torchcodec.decoders import AudioDecoder

                with smart_open.open(audio_path, "rb") as f:
                    samples = AudioDecoder(f.read()).get_all_samples()
                audio = samples.data.numpy().squeeze()
                sr = samples.sample_rate
            except Exception:  # noqa: BLE001
                logger.warning(f"Skipping unreadable audio: {audio_path}")
                raise

        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        audio = np.asarray(audio, dtype=np.float32)
        if duration <= 0 and sr > 0:
            duration = len(audio) / sr
        return audio, sr, duration

    def _process_single_entry(self, task: FileGroupTask) -> list[AudioTask]:
        """Load a single audio file and return one AudioTask."""
        corpus = task.reader_config.get("corpus", "unknown")
        shard_key = task.reader_config.get("shard_key", task.task_id)
        language = task.reader_config.get("language", "")
        entry = task.reader_config["entry"]

        audio_path = task.data[0]
        hint_sr = entry.get("sampling_rate") or entry.get("sample_rate")

        try:
            audio, sr, duration = self._load_audio(
                audio_path, hint_sr=hint_sr, hint_duration=entry.get("duration", 0.0),
            )
        except Exception:  # noqa: BLE001
            return []

        entry_data = {k: v for k, v in entry.items() if k != "audio_filepath"}
        entry_data.update({
            "waveform": audio,
            "sampling_rate": sr,
            "sample_rate": sr,
            "duration": duration,
            "num_channels": 1,
            "corpus": corpus,
            "audio_filepath": audio_path,
        })
        if language and "source_lang" not in entry_data:
            entry_data["source_lang"] = language

        shard_total = task.reader_config.get("shard_total", 0)
        metadata = {**task._metadata, "_shard_key": shard_key, "_shard_total": shard_total}
        return [AudioTask(task_id=task.task_id, dataset_name=corpus, data=entry_data, _metadata=metadata)]

    def _process_cutset(self, task: FileGroupTask) -> list[AudioTask]:
        """Load all cuts from a manifest/tar shard and return AudioTasks."""
        corpus = task.reader_config.get("corpus", "unknown")
        shard_key = task.reader_config.get("shard_key", task.task_id)
        language = task.reader_config.get("language", "")
        metadata = dict(task._metadata)

        manifest_path = task.data[0]
        tar_path = task.data[1] if len(task.data) >= 2 else None  # noqa: PLR2004
        cutset = self._make_cutset(manifest_path, tar_path)

        mode = "tarred" if tar_path else "non-tarred"
        logger.info(f"Reading shard {shard_key} via NeMo {mode} adapter")

        results: list[AudioTask] = []
        loaded = 0
        for cut in cutset:
            try:
                audio = cut.load_audio().squeeze()
            except Exception:  # noqa: BLE001
                logger.warning(f"Skipping unreadable audio: {cut.id}")
                continue

            if audio.ndim > 1:
                audio = audio.mean(axis=0)

            target_sr = cut.recording.sampling_rate
            if cut.duration > 0:
                actual_sr = round(len(audio) / cut.duration)
                if actual_sr != target_sr and actual_sr > 0:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=actual_sr, target_sr=target_sr)

            loaded += 1
            if loaded % 100 == 0 or loaded == 1:
                logger.info(f"  [{shard_key}] loaded {loaded}")

            entry_data = dict(cut.custom) if cut.custom else {}
            entry_data.update({
                "waveform": np.asarray(audio, dtype=np.float32),
                "sampling_rate": target_sr,
                "sample_rate": target_sr,
                "duration": cut.duration,
                "num_channels": 1,
                "corpus": corpus,
            })
            if "audio_filepath" not in entry_data and cut.recording and cut.recording.sources:
                src = cut.recording.sources[0].source
                entry_data["audio_filepath"] = src if isinstance(src, str) else cut.id
            if language and "source_lang" not in entry_data:
                entry_data["source_lang"] = language

            results.append(AudioTask(
                task_id=f"{shard_key}_{cut.id}",
                dataset_name=corpus,
                data=entry_data,
                _metadata={**metadata, "_shard_key": shard_key},
                _stage_perf=list(task._stage_perf),
            ))

        for r in results:
            r._metadata["_shard_total"] = len(results)

        logger.info(f"Shard {shard_key}: emitted {len(results)} AudioTasks")
        return results

    def process(self, task: FileGroupTask) -> list[AudioTask]:
        if task.reader_config.get("entry") is not None:
            return self._process_single_entry(task)
        return self._process_cutset(task)

    def process_batch(self, tasks: list[FileGroupTask]) -> list[AudioTask]:
        if len(tasks) <= 1:
            return [at for task in tasks for at in self.process(task)]

        single_entry_tasks: list[FileGroupTask] = []
        other_tasks: list[FileGroupTask] = []
        for task in tasks:
            if task.reader_config.get("entry") is not None:
                single_entry_tasks.append(task)
            else:
                other_tasks.append(task)

        results: list[AudioTask] = []

        if single_entry_tasks:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            n_threads = min(self.max_io_threads, len(single_entry_tasks))
            logger.info(
                f"NeMoSpeechReader: loading {len(single_entry_tasks)} audio files "
                f"with {n_threads} I/O threads"
            )
            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                future_to_task = {
                    pool.submit(self._process_single_entry, t): t
                    for t in single_entry_tasks
                }
                for future in as_completed(future_to_task):
                    src_task = future_to_task[future]
                    try:
                        results.extend(future.result())
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            f"Failed to load audio for task {src_task.task_id}: {exc}"
                        )

        for task in other_tasks:
            results.extend(self._process_cutset(task))

        return results


# ---------------------------------------------------------------------------
# Composite stage (user-facing API)
# ---------------------------------------------------------------------------


@dataclass
class NeMoSpeechAudioReader(CompositeStage[_EmptyTask, AudioTask]):
    """Unified reader for NeMo audio datasets.

    Reads NeMo ``input_cfg`` YAML configs and uses NeMo's lhotse
    adapters (``LazyNeMoIterator`` / ``LazyNeMoTarredIterator``)
    for audio loading.

    Args:
        max_io_threads: Maximum concurrent threads for loading audio
            from S3/object storage. Higher values overlap more network
            latency but use more memory. Defaults to 8.
    """

    name: str = "nemo_speech_audio_reader"
    yaml_path: str = ""
    corpus_filter: list[str] | None = None
    language_filter: list[str] | None = None
    output_dir: str | None = None
    max_io_threads: int = 8
    read_concurrency: int = 2

    def __post_init__(self) -> None:
        super().__init__()
        if not self.yaml_path:
            msg = "yaml_path is required for NeMoSpeechAudioReader"
            raise ValueError(msg)
        self._stages: list[ProcessingStage] = [
            NeMoSpeechDiscoveryStage(
                yaml_path=self.yaml_path,
                corpus_filter=self.corpus_filter,
                language_filter=self.language_filter,
                output_dir=self.output_dir,
            ),
            NeMoSpeechReaderStage(max_io_threads=self.max_io_threads, read_concurrency=self.read_concurrency),
        ]

    def inputs(self) -> tuple[list[str], list[str]]:
        return self._stages[0].inputs()

    def outputs(self) -> tuple[list[str], list[str]]:
        return self._stages[-1].outputs()

    def decompose(self) -> list[ProcessingStage]:
        return self._stages
