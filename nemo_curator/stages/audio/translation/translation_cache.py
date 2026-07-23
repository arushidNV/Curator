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

"""Lock-free, per-worker translation cache with a driver-side merge.

Speech corpora contain many recurring **short** utterances ("Yes.", "Thank you.", numbers,
boilerplate). Re-translating identical short sources wastes GPU. A previous attempt used a single
shared SQLite (``diskcache``) consulted by every worker; on Lustre its writer lock serialized all
workers and killed parallelism. This module avoids any shared store **during** the run:

* Each worker keeps the canonical cache in an **in-memory** dict (warm start; served lock-free) and
  counts every key access in memory.
* Each worker periodically **flushes a snapshot** of its entries (``{key: src, translation, count}``)
  to its **own** dump file -- no two writers ever touch the same file, and the snapshot is rewritten
  (not appended), so hot keys cost O(distinct keys), not O(accesses).
* The driver, after the pipeline finishes, calls :func:`merge_translation_cache` to fold all
  per-worker snapshots + the previous canonical into one canonical file, written atomically,
  **summing** the access counts.

Why flush-during-run instead of dump-at-teardown: the Ray Data backend never calls a stage's
``teardown()`` (it has no such hook), and the job overlay does not sync ``backends/`` -- so the cache
must persist itself from inside the stage's own ``process_batch`` path, not a lifecycle hook. A
periodic snapshot flush is executor-agnostic (works under ray_data, Xenna, ray_actor_pool alike);
worst case a hard kill loses only the counts since the last flush.

Value / limitation: this gives a **cross-run warm start** for common short phrases, plus intra-run
reuse within a single worker (``record`` updates the in-memory dict). There is no cross-worker
sharing *within* one run (a string seen on two workers is translated by both) -- the accepted
tradeoff for being lock-free.

Cache key: ``sha256(model_id | src_lang | tgt_lang | text)`` -- the target language matters because
the translation differs per target, and the source language captures direction. With a fixed
seed/prompt the translation is deterministic, so model + langs + source text is enough.

Entry format (JSONL), identical for canonical and per-worker dumps:
``{"k": <hash>, "src": <source text>, "tgt": <translation>, "hit_count": <access count>}``

``hit_count`` counts **how many times the key was requested** (every cacheable lookup -- the dominant
term is cache hits, plus the one initial miss that populated it). It is **informational** -- NOT an
eviction key (no LFU).
"""

from __future__ import annotations

import glob
import hashlib
import itertools
import json
import os
import shutil
import socket

from loguru import logger

CANONICAL_FILENAME = "canonical.jsonl"
DUMPS_DIRNAME = "dumps"
_KEY_SEP = "\x1f"  # unit separator -- unlikely to appear in model ids / language names / text
_DEFAULT_FLUSH_EVERY = 50  # flush the snapshot every N process_batch calls


def cache_key(model_id: str, src_lang: str, tgt_lang: str, text: str) -> str:
    """Stable content hash for one (model, direction, source text) translation request."""
    raw = _KEY_SEP.join((model_id or "", src_lang or "", tgt_lang or "", text or ""))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def is_cacheable(text: str, max_chars: int) -> bool:
    """Only short, non-empty sources are cached (long texts always translate)."""
    if not text:
        return False
    return 0 < len(text.strip()) <= max_chars


def canonical_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, CANONICAL_FILENAME)


def dump_path(cache_dir: str, run_id: str, host: str, pid: int) -> str:
    return os.path.join(cache_dir, DUMPS_DIRNAME, run_id, f"worker_{host}_{pid}.jsonl")


class WorkerTranslationCache:
    """Per-worker, in-memory translation cache that snapshots itself periodically.

    Built in ``LLMTranslationStage.setup()``. Loads the canonical cache as a warm start, serves
    lookups from memory (counting each access), and flushes a snapshot of its entries to this
    worker's own dump file every ``flush_every`` batches (no reliance on ``teardown()``). The driver
    merges all snapshots afterwards, summing the access counts.
    """

    def __init__(
        self, cache_dir: str, run_id: str, max_chars: int = 40, flush_every: int = _DEFAULT_FLUSH_EVERY
    ) -> None:
        self.cache_dir = cache_dir
        self.run_id = run_id
        self.max_chars = max_chars
        self.flush_every = max(1, flush_every)
        self._val: dict[str, str] = {}  # key -> translation (warm start + learned this run)
        self._src: dict[str, str] = {}  # key -> source text
        self._count: dict[str, int] = {}  # key -> times requested THIS run (this worker)
        self._since_flush = 0

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        """Read the canonical cache into memory (values only). Fail open on any error.

        Counts are NOT carried into ``_count`` -- ``_count`` holds only this run's accesses, which the
        merge adds to the canonical's running total. Loading counts here would double-count them.
        """
        path = canonical_path(self.cache_dir)
        try:
            if not os.path.exists(path):
                logger.info("TranslationCache: no canonical cache at {} (cold start)", path)
                return
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    self._val[entry["k"]] = entry["tgt"]
                    self._src[entry["k"]] = entry.get("src", "")
            logger.info("TranslationCache: loaded {} entries from {}", len(self._val), path)
        except Exception as exc:  # noqa: BLE001 - cache is best-effort; never block the run
            logger.warning("TranslationCache: failed to load {}: {}; starting empty", path, exc)
            self._val, self._src = {}, {}

    # ---------------------------------------------------------------- lookup
    def is_cacheable(self, text: str) -> bool:
        return is_cacheable(text, self.max_chars)

    def lookup(
        self, model_id: str, src_lang: str, tgt_lang: str, text: str
    ) -> tuple[str | None, str | None]:
        """Return ``(key, value)`` and **count this access**.

        ``(None, None)`` -> not cacheable (caller translates normally, not counted).
        ``(key, value)`` -> cache hit (use ``value``, skip the LLM).
        ``(key, None)`` -> cacheable miss (caller translates, then calls :meth:`record`).
        """
        if not self.is_cacheable(text):
            return None, None
        key = cache_key(model_id, src_lang, tgt_lang, text)
        self._count[key] = self._count.get(key, 0) + 1  # every request (hit or the first miss)
        self._src[key] = text
        return key, self._val.get(key)

    def record(self, key: str, translation: str) -> None:
        """Store a freshly-translated (miss) result so later occurrences hit and it gets persisted."""
        if key and translation:
            self._val[key] = translation

    def maybe_flush(self) -> None:
        """Call once per ``process_batch``; flushes a snapshot every ``flush_every`` batches."""
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Atomically (re)write this worker's full snapshot of counted entries. Best-effort."""
        self._since_flush = 0
        if not self._count:
            return
        path = dump_path(self.cache_dir, self.run_id, socket.gethostname(), os.getpid())
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.tmp"
            written = 0
            with open(tmp, "w", encoding="utf-8") as f:
                for key, count in self._count.items():
                    value = self._val.get(key)
                    if not value:  # accessed but never got a (non-empty) translation -> skip
                        continue
                    f.write(
                        json.dumps(
                            {"k": key, "src": self._src.get(key, ""), "tgt": value, "hit_count": count},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    written += 1
            os.replace(tmp, path)  # atomic: a reader/crash never sees a half-written snapshot
            logger.debug("TranslationCache: flushed {} entries to {}", written, path)
        except Exception as exc:  # noqa: BLE001 - best-effort; keep translating
            logger.warning("TranslationCache: failed to flush snapshot to {}: {}", path, exc)

    def close(self) -> None:
        """Final flush (for backends that call teardown; ray_data relies on maybe_flush)."""
        self.flush()


def merge_translation_cache(cache_dir: str, max_entries: int = 0) -> int:
    """Fold all per-worker snapshots + the previous canonical into one canonical file.

    Single-process, run by the driver after the pipeline finishes. Unions keys (deduped by hash) and
    **sums** the access counter ``n`` across the previous canonical and every worker snapshot, writes
    the result atomically, and deletes the consumed dump dirs. Returns the number of merged entries.
    Never raises -- all errors are logged.

    NOTE: with a shared/global cache_dir, concurrent jobs merging at the same instant can race on the
    read-modify-write of ``canonical.jsonl`` (last writer wins; the atomic rename keeps the file
    valid, a few entries may be dropped and re-translated next run). This is accepted as low-risk —
    lang-groups have disjoint keys and jobs finish at staggered times.
    """
    canonical = canonical_path(cache_dir)
    merged: dict[str, dict] = {}

    # 1. previous canonical first (carries the running count; its src/v win on collision)
    try:
        if os.path.exists(canonical):
            with open(canonical, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    merged[entry["k"]] = {
                        "src": entry.get("src", ""),
                        "tgt": entry["tgt"],
                        "hit_count": int(entry.get("hit_count", 0)),
                    }
    except Exception as exc:  # noqa: BLE001
        logger.warning("TranslationCache.merge: failed to read {}: {}", canonical, exc)

    # 2. fold in every per-worker snapshot (this run + any leftover prior runs); sum the counts
    dump_files = sorted(glob.glob(os.path.join(cache_dir, DUMPS_DIRNAME, "*", "*.jsonl")))
    consumed_dirs: set[str] = set()
    for path in dump_files:
        consumed_dirs.add(os.path.dirname(path))
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    key = entry["k"]
                    count = int(entry.get("hit_count", 0))
                    if key in merged:
                        merged[key]["hit_count"] += count
                    else:
                        merged[key] = {"src": entry.get("src", ""), "tgt": entry["tgt"], "hit_count": count}
        except Exception as exc:  # noqa: BLE001 - skip a corrupt snapshot, keep the rest
            logger.warning("TranslationCache.merge: skipping bad dump {}: {}", path, exc)

    # 3. optional crude safety cap (insertion order; the counter is NOT an eviction key)
    if max_entries and len(merged) > max_entries:
        logger.warning(
            "TranslationCache.merge: {} entries > cap {}; keeping first {}",
            len(merged),
            max_entries,
            max_entries,
        )
        merged = dict(itertools.islice(merged.items(), max_entries))

    # 4. atomic publish: write temp then rename over canonical, ordered by hit_count desc
    #    (presentation only -- most-requested entries first; does not affect eviction).
    try:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = canonical + ".tmp"
        ordered = sorted(merged.items(), key=lambda kv: kv[1]["hit_count"], reverse=True)
        with open(tmp, "w", encoding="utf-8") as f:
            for key, entry in ordered:
                f.write(
                    json.dumps(
                        {"k": key, "src": entry["src"], "tgt": entry["tgt"], "hit_count": entry["hit_count"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        os.replace(tmp, canonical)
    except Exception as exc:  # noqa: BLE001 - never fail the run on a cache write
        logger.warning("TranslationCache.merge: failed to write {}: {}", canonical, exc)
        return len(merged)

    # 5. drop the consumed dump dirs so they don't accumulate across runs
    for directory in consumed_dirs:
        shutil.rmtree(directory, ignore_errors=True)

    logger.info("TranslationCache: merged to {} entries at {}", len(merged), canonical)
    return len(merged)


__all__ = [
    "WorkerTranslationCache",
    "cache_key",
    "canonical_path",
    "dump_path",
    "is_cacheable",
    "merge_translation_cache",
]
