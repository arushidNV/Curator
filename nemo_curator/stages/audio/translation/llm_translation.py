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

import os
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

from nemo_curator.stages.audio.pipeline_utils import set_note
from nemo_curator.stages.audio.translation.translation_cache import WorkerTranslationCache
from nemo_curator.stages.audio.translation.translation_utils import (
    EMPTY_SOURCE_REASON,
    SOURCE_LANG_NAME_KEY,
    TRANSLATE_TO_KEY,
    TRANSLATION_SKIP_KEY,
    TRANSLATIONS_KEY,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

# NOTE: vLLM (and therefore torch) is imported lazily inside _init_model(), NOT at
# module top level. This module is pulled in by the translation package __init__, so a
# top-level `import vllm` would load torch into EVERY actor in the package — including the
# cometoid/PyMarian QE actor, whose Marian CUDA runtime then segfaults colliding with
# torch's. Keeping the import lazy lets the QE actor stay torch-free (pymarian needs no
# torch), so cometoid QE runs on GPU in-pipeline.

_DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "translation_prompt.md"
_DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "system_prompt.md"


def _isolate_compile_caches() -> None:
    """Give this actor process its own torch.compile / inductor / triton caches.

    Multiple vLLM engines co-located on one node otherwise race on the *shared*
    default cache dirs (``~/.cache``, ``/tmp/torchinductor_<user>``,
    ``/tmp/triton``). That race corrupts the inductor cache pickle
    ("pickle data was truncated" / "CompiledFxGraph has no compiled_fn_runner"),
    kills the EngineCore, and its restart then hangs at CUDA-graph capture —
    wedging the whole pipeline (idle GPUs -> killed by the cluster idle reaper).
    Keying every cache dir by PID isolates each engine. Each Ray actor runs one
    engine in its own process, and vLLM's spawned EngineCore subprocess inherits
    these env vars, so all of an actor's (re)starts share one private cache while
    different actors never collide. Must run before the ``LLM(...)`` constructor.
    """
    root = os.environ.get("COMPILE_CACHE_ROOT", os.environ.get("TMPDIR", "/tmp"))
    base = os.path.join(root, f"compile_cache_{os.getpid()}")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(base, "inductor")
    os.environ["TRITON_CACHE_DIR"] = os.path.join(base, "triton")
    os.environ["VLLM_CACHE_ROOT"] = os.path.join(base, "vllm")
    try:
        os.makedirs(base, exist_ok=True)
    except OSError as exc:  # non-fatal — torch/triton/vllm create them lazily too
        logger.warning("_isolate_compile_caches: could not pre-create {}: {}", base, exc)
    logger.info("Per-process compile caches under {} (pid={})", base, os.getpid())


@dataclass
class LLMTranslationStage(ProcessingStage[AudioTask, AudioTask]):
    """Translate source text to a target language via batched vLLM inference.

    Reads source text plus pre-resolved language **display names** from each
    ``AudioTask.data`` dict (``source_lang_key`` and ``target_lang_key``;
    populated upstream by ``TranslationManifestReaderStage``) and writes the
    result into ``data[translations_key]`` as a ``{display_name: translation}``
    mapping, so multiple target languages accumulate without overwriting
    prior entries. The scratch fields ``source_lang_key`` and
    ``target_lang_key`` are removed from each row after the prompts are
    built, so they don't bleed into the output manifest.

    The prompt template uses fixed semantic placeholders ``{target_lang}``,
    ``{source_lang}``, ``{text}`` regardless of the actual manifest key
    names. ``{source_lang}`` is always populated from ``source_lang_key``,
    which is mandatory and assumed pre-resolved by the reader.

    Both the translation prompt and the system prompt have bundled defaults
    in ``prompts/`` (``translation_prompt.md`` and ``system_prompt.md``).
    Override either via the inline-string fields (``translation_prompt``,
    ``system_prompt``) or the ``*_file`` fields (``translation_prompt_file``,
    ``system_prompt_file``); passing both inline and file for the same
    prompt raises.
    """

    name: str = "LLMTranslation"
    model_id: str | None = None
    translation_prompt: str | None = None
    translation_prompt_file: str | None = None
    system_prompt: str | None = None
    system_prompt_file: str | None = None
    text_key: str = "tn_raw"
    source_lang_key: str = SOURCE_LANG_NAME_KEY
    target_lang_key: str = TRANSLATE_TO_KEY
    translations_key: str = TRANSLATIONS_KEY
    skip_me_key: str = TRANSLATION_SKIP_KEY
    notes_key: str = "additional_notes"
    tensor_parallel_size: int | None = None
    max_output_tokens: int = 256
    max_model_len: int = 1024
    max_num_seqs: int = 512
    max_num_batched_tokens: int | None = 16384
    gpu_memory_utilization: float = 0.90
    kv_cache_dtype: str = "fp8"
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0
    seed: int = 1234
    num_workers_override: int | None = None
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    batch_size: int = 512

    # Lock-free per-worker translation cache (warm start from a canonical file, dumped per
    # worker at teardown, merged by the driver after the run). See translation_cache.py.
    cache_enabled: bool = False
    cache_dir: str | None = None
    cache_run_id: str | None = None
    cache_max_chars: int = 40

    _llm: Any = field(default=None, init=False, repr=False)
    _cache: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _sampling_params: Any = field(default=None, init=False, repr=False)
    _translation_prompt: str = field(default="", init=False, repr=False)
    _system_prompt: str | None = field(default=None, init=False, repr=False)
    _prompt_placeholders: frozenset[str] = field(default_factory=frozenset, init=False, repr=False)
    _n_processed: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.model_id:
            msg = "LLMTranslationStage: model_id is required"
            raise ValueError(msg)

        if not self.tensor_parallel_size or self.tensor_parallel_size <= 0:
            from nemo_curator.utils.gpu_utils import get_gpu_count
            self.tensor_parallel_size = get_gpu_count()
        self.resources = Resources(gpus=float(self.tensor_parallel_size))

        self._translation_prompt = self._resolve_prompt(
            inline=self.translation_prompt,
            file_path=self.translation_prompt_file,
            default_path=_DEFAULT_PROMPT_PATH,
            label="translation",
        )
        self._system_prompt = self._resolve_prompt(
            inline=self.system_prompt,
            file_path=self.system_prompt_file,
            default_path=_DEFAULT_SYSTEM_PROMPT_PATH,
            label="system",
        )
        self._prompt_placeholders = frozenset(
            field_name
            for _, field_name, _, _ in string.Formatter().parse(self._translation_prompt)
            if field_name
        )

    def num_workers(self) -> int | None:
        return self.num_workers_override

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if self.num_workers_override is not None:
            spec["num_workers"] = self.num_workers_override
        return spec

    # ------------------------------------------------------------------
    # Prompt resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_prompt(
        inline: str | None,
        file_path: str | None,
        default_path: Path | None,
        label: str,
    ) -> str | None:
        if inline and file_path:
            raise ValueError(
                f"LLMTranslation: pass either {label}_prompt or {label}_prompt_file, not both."
            )
        if inline:
            return inline
        path = Path(file_path) if file_path else default_path
        if path is None:
            return None
        logger.info("LLMTranslation: loading {} prompt from {}", label, path)
        if not path.exists():
            raise FileNotFoundError(f"{label.capitalize()} prompt file not found: {path}")
        return path.read_text(encoding="utf-8").strip()

    # ------------------------------------------------------------------
    # Model initialisation
    # ------------------------------------------------------------------

    def _init_model(self) -> None:
        # Imported here (not at module top level) so loading this module — and thus the
        # translation package __init__ — never pulls torch into non-LLM actors (e.g. the
        # pymarian QE actor, which segfaults if torch's CUDA runtime is also loaded).
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError("vLLM is required for LLMTranslationStage. pip install vllm") from e

        # Isolate this process's compile caches BEFORE building the engine, so
        # co-located engines don't race on the shared inductor/triton cache.
        _isolate_compile_caches()

        max_num_batched_tokens = self.max_num_batched_tokens or max(self.max_model_len, 8192)

        # enforce_eager skips torch.compile + CUDA-graph capture. Set
        # VLLM_ENFORCE_EAGER=1 to avoid the graph-capture cost on every engine
        # (re)spawn and the CUDA-graph-replay hang class that wedges the pipeline
        # when Ray Data / Xenna tears down / respawns an actor mid-stream.
        enforce_eager = os.environ.get("VLLM_ENFORCE_EAGER", "0").lower() in ("1", "true", "yes")

        # Optionally force-disable vLLM V1 async scheduling — the
        # `step_with_batch_queue` path that deadlocks (SM-100%/mem-0% spin) after
        # Ray Data / Xenna tears down/respawns an actor mid-stream. VLLM_ASYNC_SCHEDULING=0
        # forces synchronous stepping; unset -> vLLM default (auto-on in V1).
        _async = os.environ.get("VLLM_ASYNC_SCHEDULING", "").strip().lower()
        async_kwargs: dict[str, Any] = {}
        if _async in ("0", "false", "no"):
            async_kwargs["async_scheduling"] = False
        elif _async in ("1", "true", "yes"):
            async_kwargs["async_scheduling"] = True

        logger.info(
            "LLMTranslation: loading {} (tp={}, max_model_len={}, "
            "max_num_batched_tokens={}, kv_cache_dtype={}, enforce_eager={})",
            self.model_id,
            self.tensor_parallel_size,
            self.max_model_len,
            max_num_batched_tokens,
            self.kv_cache_dtype,
            enforce_eager,
        )

        self._llm = LLM(
            model=self.model_id,
            trust_remote_code=True,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_prefix_caching=True,
            prefix_caching_hash_algo="xxhash",
            kv_cache_dtype=self.kv_cache_dtype,
            enforce_eager=enforce_eager,
            gdn_prefill_backend="triton",
            enable_expert_parallel=False,
            seed=self.seed,
            **async_kwargs,
        )
        self._tokenizer = self._llm.get_tokenizer()
        self._sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            presence_penalty=self.presence_penalty,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_output_tokens,
            seed=self.seed,
        )

        logger.info(
            "LLMTranslation: model ready (prefix_caching=True, prompt={} chars)",
            len(self._translation_prompt),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        # No-op: the vLLM engine is a per-worker GPU resource, so it must be
        # built lazily in setup() (once per actor), not at node-level setup.
        # Loading here would build the engine twice (node + worker).
        pass

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._llm is None:
            self._init_model()
        if self.cache_enabled and self.cache_dir and self._cache is None:
            self._cache = WorkerTranslationCache(
                cache_dir=self.cache_dir,
                run_id=self.cache_run_id or "default",
                max_chars=self.cache_max_chars,
            )
            self._cache.load()

    def teardown(self) -> None:
        if self._cache is not None:
            # Entries are appended to the worker's dump file as they're translated (NOT here),
            # because the Ray Data backend never calls teardown(). This just closes the handle
            # (best-effort) on backends that do call teardown (Xenna / ray_actor_pool).
            self._cache.close()
            self._cache = None
        if self._n_processed:
            logger.info("LLMTranslation: processed {} entries", self._n_processed)
        if self._llm is not None:
            del self._llm
            self._llm = None
            self._tokenizer = None
            self._sampling_params = None

    # ------------------------------------------------------------------
    # I/O contract
    # ------------------------------------------------------------------

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key, self.target_lang_key, self.source_lang_key, self.skip_me_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.translations_key]

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def _build_prompt_values(
        self, data: dict, target_lang: str, source_lang: str
    ) -> dict[str, str]:

        semantic: dict[str, str] = {
            "target_lang": target_lang,
            "source_lang": source_lang,
            # The {text} placeholder always resolves to the configured text_key
            # (e.g. tn_raw), keeping it in sync with the empty-skip check.
            "text": data.get(self.text_key, ""),
        }

        values: dict[str, str] = {}
        missing: list[str] = []
        for placeholder in self._prompt_placeholders:
            if placeholder in semantic:
                value = semantic[placeholder]
            else:
                raw = data.get(placeholder, None)
                value = "" if raw is None else str(raw)
            if not value.strip():
                missing.append(placeholder)
            values[placeholder] = value

        if missing:
            raise ValueError(f"Translation prompt placeholders not filled: {sorted(missing)}")
        return values

    def _format_prompt(
        self, data: dict, target_lang: str, source_lang: str
    ) -> str:
        user_content = self._translation_prompt.format_map(
            self._build_prompt_values(data, target_lang, source_lang)
        )
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": user_content})
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def _emit_empty_translations(self, task: AudioTask, targets: list[str], note: str | None = None) -> None:
        # Do not run the LLM, but keep the row with one empty translation per
        # target so the writer's per-direction counter still reaches .done.
        # ``setdefault`` preserves any translations already produced for the row.
        # ``note`` is None for the flagged-skip path (the skip flag is the reason,
        # so no note is recorded), and an ``applied (...)`` note otherwise.
        translations = task.data.get(self.translations_key) or {}
        for target_lang in targets:
            translations.setdefault(target_lang, "")
        task.data[self.translations_key] = translations
        if note is not None:
            set_note(task.data, self.name, note, self.notes_key)

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        # Ray Data may pass tasks as an ndarray, so use len() not `if not tasks`.
        if len(tasks) == 0:
            return []

        if self._llm is None:
            msg = "Model not initialised — setup() was not called"
            raise RuntimeError(msg)

        prompts: list[str] = []
        prompt_owners: list[tuple[int, str]] = []
        # (task_idx, target_lang) -> cache key, for cacheable misses written back after generate.
        to_learn: dict[tuple[int, str], str] = {}

        for task_idx, task in enumerate(tasks):
            data = task.data

            # Pop scratch fields written by TranslationManifestReaderStage
            # up-front so they never reach the output manifest, regardless of
            # which skip branch (if any) the task hits below.
            raw_targets = data.pop(self.target_lang_key, None) or []
            source_lang = data.pop(self.source_lang_key, "")

            # Normalize target language(s): accept str or list[str].
            if isinstance(raw_targets, str):
                raw_targets = [raw_targets]
            targets = list(dict.fromkeys(raw_targets))  # Remove duplicates

            # Skip tasks with no targets (these rows are not counted by the
            # reader either, so downstream row-count expectations stay matched).
            if not targets:
                continue

            # Honor the skip flag (non-empty string or boolean True): keep the row
            # but emit empty translations instead of running the LLM. The skip flag
            # is the reason, so no note is recorded here.
            if data.get(self.skip_me_key, ""):
                self._emit_empty_translations(task, targets)
                continue

            text = data.get(self.text_key, "")
            if not text or not text.strip():
                # Empty source text: nothing to translate. Mark the working flag with
                # the reserved reason so downstream filters short-circuit it and
                # FinalizeTranslationStage classifies it as the empty-source case
                # (not a quality rejection), regardless of which filters are enabled.
                data[self.skip_me_key] = EMPTY_SOURCE_REASON
                self._emit_empty_translations(task, targets, "applied (empty_text_skipped)")
                continue

            for target_lang in targets:
                # Cache: short, recurring sources are served from the in-memory warm-start cache
                # without an LLM call. lookup() also counts the request + records the source text.
                if self._cache is not None:
                    key, hit = self._cache.lookup(self.model_id, source_lang, target_lang, text)
                    if hit is not None:
                        translations = data.get(self.translations_key) or {}
                        translations[target_lang] = hit
                        data[self.translations_key] = translations
                        set_note(data, self.name, "applied (cache_hit)", self.notes_key)
                        continue
                    if key is not None:  # cacheable miss -> translate, then write back below
                        to_learn[(task_idx, target_lang)] = key

                prompt = self._format_prompt(data, target_lang, source_lang)
                prompts.append(prompt)
                prompt_owners.append((task_idx, target_lang))

        if prompts:
            outputs = self._llm.generate(
                prompts,
                sampling_params=self._sampling_params,
                use_tqdm=False,
            )

            for seq_idx, (task_idx, target_lang) in enumerate(prompt_owners):
                task = tasks[task_idx]
                translation = outputs[seq_idx].outputs[0].text.strip()

                if not translation:
                    logger.warning(
                        "LLMTranslation: empty translation for target={}", target_lang,
                    )

                translations = task.data.get(self.translations_key) or {}
                translations[target_lang] = translation
                task.data[self.translations_key] = translations
                set_note(task.data, self.name, "applied (translated)", self.notes_key)
                self._n_processed += 1

                # Write a fresh (non-empty) translation back to the per-worker cache so it can be
                # merged into the canonical cache for future runs.
                if self._cache is not None and translation:
                    key = to_learn.get((task_idx, target_lang))
                    if key:
                        self._cache.record(key, translation)

        # Persist the cache periodically from inside process_batch (the Ray Data backend never
        # calls teardown()), flushing this worker's snapshot every flush_every batches.
        if self._cache is not None:
            self._cache.maybe_flush()

        logger.debug("LLMTranslation: batch of {} tasks ({} translations)", len(tasks), len(prompts))
        return tasks
