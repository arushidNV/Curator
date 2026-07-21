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

_MPEG_EXTENSIONS = frozenset({".m4a", ".m4b", ".mp3", ".mp4", ".aac", ".mpeg", ".mpg"})
_TARGET_SR = 16000

# Preferred source format when the same recording appears under multiple extensions
# (lower rank = kept). Anything unlisted ranks last.
_FORMAT_PRIORITY = {".opus": 0, ".wav": 1, ".flac": 2, ".ogg": 3, ".m4a": 4, ".mp3": 5}


def _dedup_entries_by_stem(entries: list[dict], shard_key: str) -> list[dict]:
    """Drop entries that are the same recording in a different container, keeping
    the preferred format. Preserves input order.

    The dedup key is the full path minus its extension (directory included), so only
    genuine same-recording duplicates (e.g. ``d/vid1.opus`` vs ``d/vid1.wav``) collapse.
    Distinct recordings that merely share a basename across directories
    (``set_a/utt_001.wav`` vs ``set_b/utt_001.wav``) are kept — using the basename
    alone would silently drop one of them.

    Prevents identical output filenames from colliding and avoids reprocessing the
    same audio multiple times.
    """
    best: dict[str, tuple[int, int]] = {}  # path-stem -> (format_rank, index into entries)
    order: list[str] = []
    for i, entry in enumerate(entries):
        path = entry.get("audio_filepath", "")
        if not path:
            continue
        key = os.path.splitext(path)[0]  # directory-preserving stem
        # Segment-level input (e.g. Granary ASR reading metadata_extraction output)
        # points many rows at the same source recording, distinguished only by
        # offset/duration. Fold those into the key so distinct segments are kept;
        # true same-recording/different-format duplicates still share offset/duration
        # and collapse as before.
        offset = entry.get("offset")
        duration = entry.get("duration")
        if offset is not None or duration is not None:
            key = f"{key}|{offset}|{duration}"
        rank = _FORMAT_PRIORITY.get(os.path.splitext(path)[1].lower(), 99)
        if key not in best:
            best[key] = (rank, i)
            order.append(key)
        elif rank < best[key][0]:
            best[key] = (rank, i)

    deduped = [entries[best[key][1]] for key in order]
    dropped = len(entries) - len(deduped)
    if dropped:
        logger.warning(
            f"[{shard_key}] deduplicated {dropped} duplicate source(s) (same recording, kept preferred format)"
        )
    return deduped


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


def _as_plain_container(obj: Any) -> Any:  # noqa: ANN401
    """Convert an OmegaConf node (as passed by Hydra ``instantiate``) to plain
    Python containers, resolving interpolations like ``${input_manifest}``.

    Non-OmegaConf objects (e.g. a list already loaded from YAML) pass through
    unchanged, so this is safe to call on either an inline ``input_cfg`` or a
    ``yaml.safe_load`` result.
    """
    try:
        from omegaconf import OmegaConf
    except ImportError:
        return obj
    if OmegaConf.is_config(obj):
        return OmegaConf.to_container(obj, resolve=True)
    return obj


def _load_input_cfg(yaml_path: str | None, input_cfg: Any = None) -> list[dict[str, Any]]:  # noqa: ANN401
    """Return the raw ``input_cfg`` list from either an inline object or a YAML file.

    ``input_cfg`` (when provided) takes precedence over ``yaml_path`` and may be an
    OmegaConf list (interpolations are resolved). This lets callers embed the config
    inline in a pipeline YAML instead of maintaining a separate wrapper file.
    """
    if input_cfg is not None:
        return _as_plain_container(input_cfg)
    if yaml_path:
        import yaml

        with open(yaml_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    msg = "Either input_cfg or yaml_path must be provided"
    raise ValueError(msg)


def _shards_from_cfg_entry(cfg: dict[str, Any], corpus: str, language: str) -> list[dict[str, Any]]:
    """Expand a single ``input_cfg`` entry into one shard descriptor per manifest.

    Tarred entries pair each expanded manifest with its tar path; non-tarred
    entries emit one descriptor per expanded manifest path.
    """
    if "tarred_audio_filepaths" in cfg:
        manifest_paths = _expand_nemo_path(cfg["manifest_filepath"])
        tar_paths = _expand_nemo_path(cfg["tarred_audio_filepaths"])
        if len(manifest_paths) != len(tar_paths):
            msg = f"Manifest/tar count mismatch for {corpus}: {len(manifest_paths)} vs {len(tar_paths)}"
            raise ValueError(msg)
        return [
            {"corpus": corpus, "manifest_path": mp, "tar_path": tp, "language": language}
            for mp, tp in zip(manifest_paths, tar_paths, strict=False)
        ]
    if "manifest_filepath" in cfg:
        return [
            {"corpus": corpus, "manifest_path": mp, "language": language}
            for mp in _expand_nemo_path(cfg["manifest_filepath"])
        ]
    return []


def _parse_input_cfg(
    config: list[dict[str, Any]],
    corpus_filter: list[str] | None,
    language_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Parse a NeMo ``input_cfg`` list into shard descriptors.

    Each descriptor has ``manifest_path``, optional ``tar_path``,
    ``corpus``, and ``language``.

    Only supports the standard NeMo config format with ``input_cfg``
    entries of type ``nemo_tarred`` or ``nemo``.
    """
    if not isinstance(config, list) or not config:
        msg = f"Expected an input_cfg list, got {type(config)}"
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

            shards.extend(_shards_from_cfg_entry(cfg, corpus, language))

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
    input_cfg: Any = None
    corpus_filter: list[str] | None = None
    language_filter: list[str] | None = None
    output_dir: str | None = None

    def __post_init__(self) -> None:
        if not self.yaml_path and self.input_cfg is None:
            msg = "Either input_cfg or yaml_path is required"
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
        config = _load_input_cfg(self.yaml_path, self.input_cfg)
        shard_descs = _parse_input_cfg(config, self.corpus_filter, self.language_filter)

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
                tasks.append(
                    FileGroupTask(
                        task_id=shard_key,
                        dataset_name=corpus,
                        data=[desc["manifest_path"], desc["tar_path"]],
                        reader_config={"corpus": corpus, "shard_key": shard_key, "language": desc.get("language", "")},
                    )
                )
            else:
                # Non-tarred: read manifest, emit one task per entry for parallel loading
                import json

                from fsspec.core import url_to_fs

                try:
                    fs, resolved = url_to_fs(desc["manifest_path"])
                    with fs.open(resolved, "r", encoding="utf-8") as f:
                        entries = [json.loads(line) for line in f if line.strip()]
                    entries = _dedup_entries_by_stem(entries, shard_key)
                    for i, entry in enumerate(entries):
                        tasks.append(
                            FileGroupTask(
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
                            )
                        )
                except Exception:  # noqa: BLE001
                    tasks.append(
                        FileGroupTask(
                            task_id=shard_key,
                            dataset_name=corpus,
                            data=[desc["manifest_path"]],
                            reader_config={
                                "corpus": corpus,
                                "shard_key": shard_key,
                                "language": desc.get("language", ""),
                            },
                        )
                    )

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
        resampled_output_dir: If set, write resampled 16 kHz mono WAV files
            to this directory. The output filename matches the input stem
            with a ``.wav`` extension.
        keep_waveform: Whether to pass the waveform array to the next stage
            in the task data. Defaults to True. Set to False when downstream
            stages only need the resampled file path.
    """

    name: str = "nemo_speech_reader"
    max_io_threads: int = 8
    batch_size: int = 8
    # Max shards read in parallel. Caps in-flight waveforms so the object store
    # doesn't overflow (without it, Ray launches up to one reader task per CPU).
    read_concurrency: int = 2
    resampled_output_dir: str | None = None
    keep_waveform: bool = True

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        cols = ["sampling_rate", "corpus", "num_channels", "resampled_audio_filepath"]
        if self.keep_waveform:
            cols.insert(0, "waveform")
        return ["data"], cols

    def setup_on_node(
        self,
        _node_info: Any = None,  # noqa: ANN401
        _worker_metadata: Any = None,  # noqa: ANN401
    ) -> None:
        """Create the resampled output directory once per node."""
        if self.resampled_output_dir:
            os.makedirs(self.resampled_output_dir, exist_ok=True)

    def _write_resampled_wav(self, audio: np.ndarray, sr: int, source_path: str) -> str:
        """Resample to _TARGET_SR if needed and write a mono WAV file. Returns the output path."""
        import soundfile as sf

        if sr != _TARGET_SR:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=_TARGET_SR)
            sr = _TARGET_SR

        stem = os.path.splitext(os.path.basename(source_path))[0]
        out_path = os.path.join(self.resampled_output_dir, f"{stem}.wav")
        sf.write(out_path, audio, sr, subtype="PCM_16")
        return out_path

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
    def _ffprobe_sample_rate(path: str) -> int | None:
        """Return the source's native audio sample rate via ffprobe, or None if unknown."""
        import shutil
        import subprocess

        ffprobe_bin = shutil.which("ffprobe")
        if ffprobe_bin is None:
            return None
        cmd = [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        proc = subprocess.run(cmd, capture_output=True, check=False)  # noqa: S603
        if proc.returncode != 0:
            return None
        try:
            return int(proc.stdout.decode("utf-8", "replace").strip().splitlines()[0])
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _load_audio_ffmpeg(audio_path: str, target_sr: int = _TARGET_SR) -> tuple[np.ndarray, int]:
        """Decode any format (m4a/mp3/mp4/...) to mono float32 via the ffmpeg CLI.

        Streams bytes through smart_open (S3/AIS aware) to a temp file, probes the
        native sample rate with ffprobe, and decodes at that rate so the original
        sample rate is preserved for manifest provenance (downstream MonoDownsample
        handles the resample to the pipeline target). Only when probing fails do we
        fall back to ``target_sr``. Using a temp file (not a pipe) keeps seekable
        containers like mp4/m4a — whose moov atom may sit at the end — decodable.

        This is the most container-robust path: it needs only the ffmpeg/ffprobe
        binaries, not torchaudio's ffmpeg backend or torchcodec's native libraries
        (both of which are frequently missing/broken in images).
        """
        import shutil
        import subprocess
        import tempfile

        import smart_open

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            msg = "ffmpeg binary not found on PATH"
            raise RuntimeError(msg)

        with smart_open.open(audio_path, "rb") as f:
            raw = f.read()

        suffix = os.path.splitext(audio_path)[1] or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(raw)
            tmp.flush()

            out_sr = NeMoSpeechReaderStage._ffprobe_sample_rate(tmp.name) or target_sr
            cmd = [
                ffmpeg_bin,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                tmp.name,
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ac",
                "1",
                "-ar",
                str(out_sr),
                "pipe:1",
            ]
            proc = subprocess.run(cmd, capture_output=True, check=False)  # noqa: S603

        if proc.returncode != 0 or not proc.stdout:
            stderr_tail = proc.stderr.decode("utf-8", "replace").strip()[-500:]
            msg = f"ffmpeg decode failed (rc={proc.returncode}): {stderr_tail}"
            raise RuntimeError(msg)

        audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
        return audio, out_sr

    @staticmethod
    def _decode_hint_recording(audio_path: str, hint_sr: int, hint_duration: float) -> tuple[np.ndarray, int, float]:
        """Decode using a lhotse ``Recording`` built from manifest SR/duration hints."""
        from lhotse import Recording
        from lhotse.audio import AudioSource
        from nemo.utils.data_utils import is_datastore_path

        source_type = "url" if is_datastore_path(audio_path) else "file"
        rec = Recording(
            id=audio_path,
            sources=[AudioSource(type=source_type, channels=[0], source=audio_path)],
            sampling_rate=int(hint_sr),
            num_samples=int(hint_duration * hint_sr),
            duration=hint_duration,
            channel_ids=[0],
        )
        return rec.load_audio().squeeze(), rec.sampling_rate, rec.duration

    @staticmethod
    def _decode_from_file(audio_path: str) -> tuple[np.ndarray, int, float]:
        """Decode by letting lhotse probe the file header (``Recording.from_file``)."""
        from lhotse import Recording

        rec = Recording.from_file(audio_path)
        return rec.load_audio().squeeze(), rec.sampling_rate, rec.duration

    @staticmethod
    def _decode_torchcodec(audio_path: str) -> tuple[np.ndarray, int, float]:
        """Decode via torchcodec's ``AudioDecoder`` (last-resort fallback)."""
        import smart_open
        from torchcodec.decoders import AudioDecoder

        with smart_open.open(audio_path, "rb") as f:
            samples = AudioDecoder(f.read()).get_all_samples()
        return samples.data.numpy().squeeze(), samples.sample_rate, 0.0

    @staticmethod
    def _load_audio(
        audio_path: str, hint_sr: int | None = None, hint_duration: float = 0.0
    ) -> tuple[np.ndarray, int, float]:
        """Load audio from a file path (local or S3) and return (waveform, sr, duration).

        Tries lhotse first, then an ffmpeg-CLI fallback, then torchcodec.
        Returns a 1-D float32 numpy array.
        """
        ext = os.path.splitext(audio_path)[1].lower()
        # MPEG containers need real header probing; manifest duration/SR hints are often wrong.
        use_hint_recording = bool(hint_sr) and ext not in _MPEG_EXTENSIONS

        # Ordered decoder attempts; the first that succeeds wins. ffmpeg returns
        # (audio, sr) so its duration is padded to 0.0 and recomputed below.
        attempts: list[tuple[str, Any]] = []
        if use_hint_recording:
            attempts.append(
                (
                    "hint-recording",
                    lambda: NeMoSpeechReaderStage._decode_hint_recording(audio_path, hint_sr, hint_duration),
                )
            )
        attempts.append(("from_file", lambda: NeMoSpeechReaderStage._decode_from_file(audio_path)))
        attempts.append(("ffmpeg", lambda: (*NeMoSpeechReaderStage._load_audio_ffmpeg(audio_path), 0.0)))
        attempts.append(("torchcodec", lambda: NeMoSpeechReaderStage._decode_torchcodec(audio_path)))

        audio: np.ndarray | None = None
        sr: int = 0
        duration: float = 0.0
        load_errors: list[str] = []
        for name, decode in attempts:
            try:
                audio, sr, duration = decode()
                break
            except Exception as exc:  # noqa: BLE001
                load_errors.append(f"{name}: {exc}")

        if audio is None:
            joined = "; ".join(load_errors)
            logger.warning(f"Skipping unreadable audio: {audio_path} ({joined})")
            msg = f"All decoders failed for {audio_path}: {joined}"
            raise RuntimeError(msg)

        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        audio = np.asarray(audio, dtype=np.float32)
        if duration <= 0 and sr > 0:
            duration = len(audio) / sr
        return audio, sr, duration

    def _read_error_task(self, task: FileGroupTask) -> AudioTask:
        """Build a read_error placeholder AudioTask for a source that could not be read.

        Emitting a placeholder (rather than dropping the task) is what lets a shard
        complete: the writer records the source as seen and writes an audit row, so
        ``.jsonl.done`` is eventually written even when a file deterministically fails.
        """
        corpus = task.reader_config.get("corpus", "unknown")
        shard_key = task.reader_config.get("shard_key", task.task_id)
        language = task.reader_config.get("language", "")
        entry = task.reader_config.get("entry") or {}
        audio_path = task.data[0] if task.data else entry.get("audio_filepath", "")

        entry_data = {k: v for k, v in entry.items() if k != "audio_filepath"}
        entry_data.update(
            {
                "read_error": True,
                "corpus": corpus,
                "audio_filepath": audio_path,
                "original_file": audio_path,
            }
        )
        if language and "source_lang" not in entry_data:
            entry_data["source_lang"] = language
        shard_total = task.reader_config.get("shard_total", 0)
        metadata = {**task._metadata, "_shard_key": shard_key, "_shard_total": shard_total}
        return AudioTask(task_id=task.task_id, dataset_name=corpus, data=entry_data, _metadata=metadata)

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
                audio_path,
                hint_sr=hint_sr,
                hint_duration=entry.get("duration", 0.0),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Unreadable audio, emitting read-error placeholder: {audio_path} ({exc})")
            return [self._read_error_task(task)]

        # When the manifest entry describes a sub-segment of the source recording
        # (segment-level input, e.g. Granary ASR reading metadata_extraction output),
        # emit only that slice so downstream ASR receives a short clip instead of the
        # full-length recording (which otherwise blows up attention memory).
        seg_offset = float(entry.get("offset") or 0.0)
        seg_duration = float(entry.get("duration") or 0.0)
        wants_segment = seg_offset > 0.0 or (0.0 < seg_duration < duration - 0.05)
        if sr > 0 and wants_segment:
            start = max(0, int(round(seg_offset * sr)))
            end = int(round((seg_offset + seg_duration) * sr)) if seg_duration > 0.0 else len(audio)
            start = min(start, len(audio))
            end = max(start, min(end, len(audio)))
            audio = audio[start:end]
            duration = len(audio) / sr

        entry_data = {k: v for k, v in entry.items() if k != "audio_filepath"}
        entry_data.update(
            {
                "sampling_rate": sr,
                "sample_rate": sr,
                "duration": duration,
                "num_channels": 1,
                "corpus": corpus,
                "audio_filepath": audio_path,
            }
        )

        if self.resampled_output_dir:
            resampled_path = self._write_resampled_wav(audio, sr, audio_path)
            entry_data["resampled_audio_filepath"] = resampled_path
            entry_data["sampling_rate"] = _TARGET_SR
            entry_data["sample_rate"] = _TARGET_SR

        if self.keep_waveform:
            entry_data["waveform"] = audio

        if language and "source_lang" not in entry_data:
            entry_data["source_lang"] = language

        shard_total = task.reader_config.get("shard_total", 0)
        metadata = {**task._metadata, "_shard_key": shard_key, "_shard_total": shard_total}
        return [AudioTask(task_id=task.task_id, dataset_name=corpus, data=entry_data, _metadata=metadata)]

    def _build_cut_entry(self, cut: Any, corpus: str, language: str) -> dict[str, Any]:  # noqa: ANN401
        """Decode a single cut and return an entry_data dict (or raise on failure)."""
        audio = cut.load_audio().squeeze()
        if audio.ndim > 1:
            audio = audio.mean(axis=0)

        target_sr = cut.recording.sampling_rate
        if cut.duration > 0:
            actual_sr = round(len(audio) / cut.duration)
            if actual_sr != target_sr and actual_sr > 0:
                import librosa

                audio = librosa.resample(audio, orig_sr=actual_sr, target_sr=target_sr)

        audio = np.asarray(audio, dtype=np.float32)
        entry_data = dict(cut.custom) if cut.custom else {}
        entry_data.update(
            {
                "sampling_rate": target_sr,
                "sample_rate": target_sr,
                "duration": cut.duration,
                "num_channels": 1,
                "corpus": corpus,
            }
        )

        audio_filepath = ""
        if cut.recording and cut.recording.sources:
            src = cut.recording.sources[0].source
            audio_filepath = src if isinstance(src, str) else cut.id

        if self.resampled_output_dir:
            source_name = audio_filepath or cut.id
            resampled_path = self._write_resampled_wav(audio, target_sr, source_name)
            entry_data["resampled_audio_filepath"] = resampled_path
            entry_data["sampling_rate"] = _TARGET_SR
            entry_data["sample_rate"] = _TARGET_SR

        if self.keep_waveform:
            entry_data["waveform"] = audio
        if "audio_filepath" not in entry_data:
            entry_data["audio_filepath"] = audio_filepath or cut.id
        if language and "source_lang" not in entry_data:
            entry_data["source_lang"] = language

        return entry_data

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
                entry_data = self._build_cut_entry(cut, corpus, language)
            except Exception:  # noqa: BLE001
                logger.warning(f"Skipping unreadable audio: {cut.id}")
                continue

            loaded += 1
            if loaded % 100 == 0 or loaded == 1:
                logger.info(f"  [{shard_key}] loaded {loaded}")

            results.append(
                AudioTask(
                    task_id=f"{shard_key}_{cut.id}",
                    dataset_name=corpus,
                    data=entry_data,
                    _metadata={**metadata, "_shard_key": shard_key},
                    _stage_perf=list(task._stage_perf),
                )
            )

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
                f"NeMoSpeechReader: loading {len(single_entry_tasks)} audio files with {n_threads} I/O threads"
            )
            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                future_to_task = {pool.submit(self._process_single_entry, t): t for t in single_entry_tasks}
                for future in as_completed(future_to_task):
                    src_task = future_to_task[future]
                    try:
                        results.extend(future.result())
                    except Exception as exc:  # noqa: BLE001
                        # Emit a placeholder so the shard can still complete instead of
                        # stalling forever on a deterministically-failing input.
                        logger.warning(
                            f"Failed to load audio for task {src_task.task_id}, emitting read-error placeholder: {exc}"
                        )
                        if src_task.reader_config.get("entry") is not None:
                            results.append(self._read_error_task(src_task))

        for task in other_tasks:
            results.extend(self._process_cutset(task))

        return results


# ---------------------------------------------------------------------------
# Composite stage (user-facing API)
# ---------------------------------------------------------------------------


@dataclass
class NeMoSpeechAudioReader(CompositeStage[_EmptyTask, AudioTask]):
    """Unified reader for NeMo audio datasets.

    Reads NeMo ``input_cfg`` configs (from a YAML file or inline) and uses
    NeMo's lhotse adapters (``LazyNeMoIterator`` / ``LazyNeMoTarredIterator``)
    for audio loading.

    Args:
        yaml_path: Path to a NeMo ``input_cfg`` YAML file.
        input_cfg: Inline ``input_cfg`` (a list of groups, matching the YAML
            file structure). Takes precedence over ``yaml_path`` and may be an
            OmegaConf list — interpolations like ``${input_manifest}`` are
            resolved. Lets a pipeline YAML embed the data config directly
            instead of pointing at a separate wrapper file.
        max_io_threads: Maximum concurrent threads for loading audio
            from S3/object storage. Higher values overlap more network
            latency but use more memory. Defaults to 8.
        resampled_output_dir: If set, write resampled 16 kHz mono WAV files
            to this directory. The output filename matches the input stem
            with a ``.wav`` extension.
        keep_waveform: Whether to pass the waveform array to the next stage
            in the task data. Defaults to True. Set to False when downstream
            stages only need the resampled file path on disk.
    """

    name: str = "nemo_speech_audio_reader"
    yaml_path: str = ""
    input_cfg: Any = None
    corpus_filter: list[str] | None = None
    language_filter: list[str] | None = None
    output_dir: str | None = None
    max_io_threads: int = 8
    read_concurrency: int = 2
    resampled_output_dir: str | None = None
    keep_waveform: bool = True

    def __post_init__(self) -> None:
        super().__init__()
        if not self.yaml_path and self.input_cfg is None:
            msg = "Either input_cfg or yaml_path is required for NeMoSpeechAudioReader"
            raise ValueError(msg)
        self._stages: list[ProcessingStage] = [
            NeMoSpeechDiscoveryStage(
                yaml_path=self.yaml_path,
                input_cfg=self.input_cfg,
                corpus_filter=self.corpus_filter,
                language_filter=self.language_filter,
                output_dir=self.output_dir,
            ),
            NeMoSpeechReaderStage(
                max_io_threads=self.max_io_threads,
                read_concurrency=self.read_concurrency,
                resampled_output_dir=self.resampled_output_dir,
                keep_waveform=self.keep_waveform,
            ),
        ]

    def inputs(self) -> tuple[list[str], list[str]]:
        return self._stages[0].inputs()

    def outputs(self) -> tuple[list[str], list[str]]:
        return self._stages[-1].outputs()

    def decompose(self) -> list[ProcessingStage]:
        return self._stages
