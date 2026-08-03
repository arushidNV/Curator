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

"""AI4Bharat IndicConformer *hybrid* (CTC+RNNT) per-language ``.nemo`` ASR.

This module holds both the model engine and its Curator pipeline stage:

- :class:`IndicConformerHybridASR` — loads the per-language
  ``ai4bharat/indicconformer_stt_<lang>_hybrid_ctc_rnnt_large`` ``.nemo`` checkpoints
  and runs inference (waveforms in → text out).
- :class:`InferenceIndicConformerHybridStage` — the ``ProcessingStage`` that wraps
  it for the audio pipeline (per-sample language routing, Ray scaling, task I/O).

These checkpoints were trained with AI4Bharat's NeMo fork
(https://github.com/AI4Bharat/NeMo, ``nemo-v2`` branch), which adds a *multi-softmax*
head to the standard NeMo ASR models: one shared Conformer encoder + shared RNNT
prediction network, and a **per-language output head** selected at inference time by
``language_id``.

The stock ``nemo-toolkit`` (2.7.x) installed in this container does NOT know those
config keys, so ``ASRModel.restore_from`` fails out of the box:

    * ``RNNTDecoder(multisoftmax=...)``      -> unexpected kwarg
    * ``RNNTJoint(multilingual=..., language_keys=...)`` -> unexpected kwargs +
      a per-language ``ModuleDict`` final layer instead of a single ``Linear``
    * ``ConvASRDecoder(multisoftmax=...)``   -> unexpected kwarg

Rather than installing the fork (which is pinned to NeMo 1.23 and would break the
rest of the pipeline), :func:`_apply_multisoftmax_patches` **monkeypatches just those
three module classes** on top of the installed NeMo so the checkpoint loads, and the
model then runs compact greedy CTC decoding and NeMo's optimized batched label-looping
RNNT decoder while preserving the fork's multilingual semantics (per-language blank
index ``V/num_langs``, per-language joint head, local-id feedback to the prediction
network). Decoding maps the per-language local token ids back to text through the
model's own ``AggregateTokenizer`` (which already ships the per-language tokenizers
and offset tables in 2.7.x).

The patches are idempotent and additive: when ``multisoftmax`` / ``multilingual`` are
absent (a normal NeMo model), every patched path falls back to the original behaviour,
so importing this module does not change ordinary NeMo usage.
"""

# The compatibility shim mirrors dynamically patched NeMo signatures, so its
# concrete types and parameter names cannot follow Curator's normal lint rules.
# ruff: noqa: ANN401, N806, PLR0913, PLW0603

from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from loguru import logger

from nemo_curator.models.base import ModelInterface
from nemo_curator.stages.audio.inference.audio_chunking import (
    engine_chunk_duration,
    has_audio_longer_than,
    merge_chunk_texts,
    split_waveforms,
)
from nemo_curator.stages.audio.pipeline_utils import set_note
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

_TARGET_SR = 16000
_MAX_CHUNK_DURATION_SEC = 40.0
_TENSORRT_INFERENCE_BATCH_SIZE = 64
_TENSORRT_ENCODER_BATCH_SIZE = 8

# Set once ``_apply_multisoftmax_patches`` has run.
_PATCHED = False
# Scratch space used to pass ``multilingual`` / ``language_keys`` from the patched
# ``RNNTJoint.__init__`` into the patched ``_joint_net_modules`` that the original
# ``__init__`` body calls before we get a chance to set the instance attributes.
# Safe because checkpoint restore instantiates modules single-threaded.
_JOINT_CTX: dict[str, Any] = {}

# The 22 languages carried by every IndicConformer hybrid checkpoint's multi-softmax head.
INDIC_CONFORMER_HYBRID_LANGS: frozenset[str] = frozenset({
    "as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai", "ml",
    "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur",
})


class _LanguageRNNTDecoder:
    """Route NeMo's per-language blank to the aggregate predictor's SOS token."""

    def __init__(self, decoder: Any, blank_index: int):
        self._decoder = decoder
        self._blank_index = blank_index

    def __getattr__(self, name: str) -> Any:
        return getattr(self._decoder, name)

    def predict(self, y: Any = None, state: Any = None, **kwargs: Any) -> Any:
        if y is not None:
            y = y.masked_fill(y == self._blank_index, self._decoder.blank_idx)
        return self._decoder.predict(y, state=state, **kwargs)


class _LanguageRNNTJoint:
    """Bind the multilingual joint network to one language head."""

    def __init__(self, joint: Any, language: str, num_classes_with_blank: int):
        self._joint = joint
        self._language = language
        self._num_classes_with_blank = num_classes_with_blank

    def __getattr__(self, name: str) -> Any:
        return getattr(self._joint, name)

    @property
    def num_classes_with_blank(self) -> int:
        return self._num_classes_with_blank

    def project_encoder(self, encoder_output: Any) -> Any:
        project = getattr(self._joint, "project_encoder", self._joint.enc)
        return project(encoder_output)

    def project_prednet(self, prednet_output: Any) -> Any:
        project = getattr(self._joint, "project_prednet", self._joint.pred)
        return project(prednet_output)

    def joint_after_projection(self, f: Any, g: Any) -> Any:
        language_ids = [self._language] * f.shape[0]
        return self._joint.joint_after_projection(f, g, language_ids=language_ids)


def _apply_multisoftmax_patches() -> None:  # noqa: C901, PLR0915
    """Idempotently patch ConvASRDecoder / RNNTJoint / RNNTDecoder for multi-softmax."""
    global _PATCHED
    if _PATCHED:
        return

    import torch
    from nemo.collections.asr.modules import conv_asr, rnnt
    from nemo.collections.asr.parts.mixins.mixins import ASRBPEMixin

    # ------------------------------------------------------------------
    # Tokenizer routing: the fork tags the aggregate tokenizer ``type:
    # multilingual``; stock NeMo only routes ``agg`` to the aggregate path and
    # sends everything else to the monolingual path (which needs a top-level
    # ``dir`` key and fails). Treat ``multilingual`` as an aggregate tokenizer.
    # ------------------------------------------------------------------
    _orig_setup_tokenizer = ASRBPEMixin._setup_tokenizer

    def _setup_tokenizer(self: Any, tokenizer_cfg: Any) -> None:
        ttype = tokenizer_cfg.get("type")
        if ttype is not None and str(ttype).lower() == "multilingual":
            self._setup_aggregate_tokenizer(tokenizer_cfg)
            # Stock NeMo keys its aggregate-tokenizer handling (vocabulary as a
            # list, CTC vocab wiring) off ``tokenizer_type == "agg"``; the fork
            # used "multilingual" for the same thing. Normalise so the model's
            # own __init__ takes the aggregate branch.
            self.tokenizer_type = "agg"
            self._derive_tokenizer_properties()
            return
        _orig_setup_tokenizer(self, tokenizer_cfg)

    ASRBPEMixin._setup_tokenizer = _setup_tokenizer

    # ------------------------------------------------------------------
    # ConvASRDecoder (auxiliary CTC head)
    # ------------------------------------------------------------------
    _ConvASRDecoder = conv_asr.ConvASRDecoder
    _conv_orig_init = _ConvASRDecoder.__init__

    def _conv_init(self: Any, *args: Any, multisoftmax: bool = False,
                   language_masks: Any = None, **kwargs: Any) -> None:
        # Structure is identical to stock NeMo; the extra kwargs only gate the
        # per-language masking applied in forward(). Drop them before delegating.
        _conv_orig_init(self, *args, **kwargs)
        self.multisoftmax = multisoftmax
        self.language_masks = language_masks

    def _conv_forward(self: Any, encoder_output: Any, language_ids: Any = None) -> Any:
        # Mirrors AI4Bharat fork conv_asr.ConvASRDecoder.forward (no @typecheck so
        # language_ids is accepted). decoder_layers -> [B, T, C]; optional mask to
        # the language's contiguous token block + blank, then log_softmax.
        if self.is_adapter_available():
            encoder_output = encoder_output.transpose(1, 2)
            encoder_output = self.forward_enabled_adapters(encoder_output)
            encoder_output = encoder_output.transpose(1, 2)

        if self.temperature != 1.0:
            decoder_output = self.decoder_layers(encoder_output).transpose(1, 2) / self.temperature
        else:
            decoder_output = self.decoder_layers(encoder_output).transpose(1, 2)

        if language_ids is not None:
            sample_mask = torch.tensor(
                [self.language_masks[lang] for lang in language_ids], dtype=torch.bool
            )
            mask = sample_mask.unsqueeze(1).repeat(1, decoder_output.shape[1], 1).to(decoder_output.device)
            decoder_output = torch.masked_select(decoder_output, mask).view(
                decoder_output.shape[0], decoder_output.shape[1], -1
            )
        return torch.nn.functional.log_softmax(decoder_output, dim=-1)

    _ConvASRDecoder.__init__ = _conv_init
    _ConvASRDecoder.forward = _conv_forward

    # ------------------------------------------------------------------
    # RNNTDecoder (shared prediction network) — only absorbs the extra kwargs.
    # ------------------------------------------------------------------
    _RNNTDecoder = rnnt.RNNTDecoder
    _dec_orig_init = _RNNTDecoder.__init__

    def _dec_init(self: Any, *args: Any, multisoftmax: bool = False,
                  language_masks: Any = None, **kwargs: Any) -> None:
        _dec_orig_init(self, *args, **kwargs)
        self.multisoftmax = multisoftmax
        self.language_masks = language_masks

    _RNNTDecoder.__init__ = _dec_init

    # ------------------------------------------------------------------
    # RNNTJoint — per-language ModuleDict final layer + language routing.
    # ------------------------------------------------------------------
    _RNNTJoint = rnnt.RNNTJoint
    _joint_orig_init = _RNNTJoint.__init__
    _joint_orig_jnm = _RNNTJoint._joint_net_modules

    def _joint_init(self: Any, *args: Any, multilingual: bool = False,
                    language_keys: Any = None, language_masks: Any = None,
                    token_id_offsets: Any = None, offset_token_ids_by_token_id: Any = None,
                    **kwargs: Any) -> None:
        # _joint_net_modules runs *inside* the original __init__ before we can set
        # instance attrs, so stash what it needs in module-level scratch.
        _JOINT_CTX["multilingual"] = multilingual
        _JOINT_CTX["language_keys"] = list(language_keys) if language_keys is not None else None
        try:
            _joint_orig_init(self, *args, **kwargs)
        finally:
            _JOINT_CTX.clear()
        self.multilingual = multilingual
        self.language_keys = list(language_keys) if language_keys is not None else None
        self.language_masks = language_masks
        self.token_id_offsets = token_id_offsets
        self.offset_token_ids_by_token_id = offset_token_ids_by_token_id

    def _joint_net_modules(self: Any, num_classes: int, pred_n_hidden: int, enc_n_hidden: int,
                           joint_n_hidden: int, activation: str, dropout: float) -> Any:
        if not _JOINT_CTX.get("multilingual"):
            return _joint_orig_jnm(self, num_classes, pred_n_hidden, enc_n_hidden,
                                   joint_n_hidden, activation, dropout)
        language_keys = _JOINT_CTX["language_keys"]
        pred = torch.nn.Linear(pred_n_hidden, joint_n_hidden)
        enc = torch.nn.Linear(enc_n_hidden, joint_n_hidden)
        act = activation.lower()
        if act == "relu":
            act_mod: Any = torch.nn.ReLU(inplace=True)
        elif act == "sigmoid":
            act_mod = torch.nn.Sigmoid()
        elif act == "tanh":
            act_mod = torch.nn.Tanh()
        else:
            msg = f"Unsupported activation for joint step: {activation}"
            raise ValueError(msg)
        # Per-language head: V/num_langs (+1 for blank). self._vocab_size is the
        # full aggregate vocab; it is set before this method is called.
        per_lang = self._vocab_size // len(language_keys) + 1
        final_layer = torch.nn.ModuleDict(
            {lang: torch.nn.Linear(joint_n_hidden, per_lang) for lang in language_keys}
        )
        logger.info(f"Multilingual RNNT joint: {len(language_keys)} heads x {per_lang} classes")
        layers = [act_mod] + ([torch.nn.Dropout(p=dropout)] if dropout else []) + [final_layer]
        return pred, enc, torch.nn.Sequential(*layers)

    def _joint_after_projection(self: Any, f: Any, g: Any, language_ids: Any = None) -> Any:
        # Mirrors fork RNNTJoint.joint_after_projection with language routing.
        f = f.unsqueeze(dim=2)  # (B, T, 1, H)
        g = g.unsqueeze(dim=1)  # (B, 1, U, H)
        inp = f + g  # (B, T, U, H)
        del f, g
        if self.is_adapter_available():
            inp = self.forward_enabled_adapters(inp)

        if language_ids is not None:
            for module in self.joint_net[:-1]:
                inp = module(inp)
            if len(set(language_ids)) == 1:
                res = self.joint_net[-1][language_ids[0]](inp)
            else:
                res = torch.stack(
                    [self.joint_net[-1][lang](single) for single, lang in zip(inp, language_ids, strict=True)]
                )
        else:
            res = self.joint_net(inp)
        del inp

        if self.preserve_memory:
            torch.cuda.empty_cache()
        if self.log_softmax is None:
            if not res.is_cuda:
                res = (res / self.temperature).log_softmax(dim=-1) if self.temperature != 1.0 else res.log_softmax(dim=-1)
        elif self.log_softmax:
            res = (res / self.temperature).log_softmax(dim=-1) if self.temperature != 1.0 else res.log_softmax(dim=-1)
        return res

    _RNNTJoint.__init__ = _joint_init
    _RNNTJoint._joint_net_modules = _joint_net_modules
    _RNNTJoint.joint_after_projection = _joint_after_projection

    _PATCHED = True
    logger.info("Applied AI4Bharat multi-softmax patches to NeMo ConvASRDecoder/RNNTDecoder/RNNTJoint")


class IndicConformerHybridASR(ModelInterface):
    """AI4Bharat IndicConformer hybrid (CTC+RNNT) per-language NeMo ASR engine.

    Pure inference: ``setup()`` then ``generate(waveforms, sample_rates, lang_codes)``.
    Knows nothing about the Curator pipeline — :class:`InferenceIndicConformerHybridStage`
    (below) adapts it to ``AudioTask`` / Ray.
    """

    def __init__(
        self,
        model_id: str,
        decode_mode: Literal["ctc", "rnnt"] = "rnnt",
        *,
        max_symbols_per_step: int = 10,
        inference_batch_size: int = 128,
        tensorrt_engine_dir: str | None = None,
        rnnt_precision: Literal["fp32", "fp16", "bf16"] = "fp32",
    ):
        if decode_mode not in {"ctc", "rnnt"}:
            msg = f"Unsupported IndicConformer decode mode: {decode_mode!r}"
            raise ValueError(msg)
        if rnnt_precision not in {"fp32", "fp16", "bf16"}:
            msg = f"Unsupported IndicConformer RNNT precision: {rnnt_precision!r}"
            raise ValueError(msg)
        if max_symbols_per_step < 1:
            msg = "max_symbols_per_step must be at least 1"
            raise ValueError(msg)
        if inference_batch_size < 1:
            msg = "inference_batch_size must be at least 1"
            raise ValueError(msg)
        self.model_id = model_id
        self.decode_mode = decode_mode
        self.max_symbols_per_step = max_symbols_per_step
        self.tensorrt_engine_dir = tensorrt_engine_dir
        self.inference_batch_size = int(inference_batch_size)
        self.rnnt_precision = rnnt_precision
        self._model: Any = None
        self._device: Any = None
        self._num_langs: int = 0
        self._per_lang_classes: int = 0  # V / num_langs (blank index within a head)
        self._trt_encoder: Any = None
        self._trt_metadata: dict[str, Any] | None = None
        self._chunk_duration_sec: float | None = None
        self._rnnt_decoders: dict[str, Any] = {}

    @property
    def model_id_names(self) -> list[str]:
        return [self.model_id]

    @staticmethod
    def _offline() -> bool:
        return os.environ.get("HF_HUB_OFFLINE", "0").strip().lower() not in ("0", "", "false", "no")

    @classmethod
    def download_to_cache(cls, model_id: str) -> str:
        """Download the repo's ``.nemo`` into the HF cache **once** (online).

        Meant to be called from :meth:`InferenceIndicConformerHybridStage.setup_on_node`
        so exactly one download happens per node — workers then resolve it from the
        cache in :meth:`setup` without each re-downloading. No-op for a local path or
        when ``HF_HUB_OFFLINE=1`` (the cache is assumed pre-populated). Returns the
        resolved local ``.nemo`` path.
        """
        if model_id.endswith(".nemo") or os.path.exists(model_id):
            return model_id
        if cls._offline():
            # Offline: rely on the pre-populated cache (no network listing/download).
            return cls._resolve_nemo_path(model_id)
        from huggingface_hub import HfApi, hf_hub_download

        files = [f for f in HfApi().list_repo_files(model_id) if f.endswith(".nemo")]
        if not files:
            msg = f"No .nemo file found in HuggingFace repo '{model_id}'"
            raise RuntimeError(msg)
        return hf_hub_download(model_id, files[0])

    @staticmethod
    def _resolve_nemo_path(model_id: str) -> str:
        """Resolve ``model_id`` to a local ``.nemo`` path — **cache-first, no download**.

        Accepts a local ``.nemo`` file, or a HuggingFace repo id like
        ``ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large``.

        Resolution order (so a pre-populated HF cache works offline, i.e. without
        compute-node egress or ``HF_TOKEN``, when ``HF_HUB_OFFLINE=1``):
          1. a literal local ``.nemo`` path,
          2. the ``.nemo`` inside the repo's **cached** snapshot (``local_files_only``),
          3. an online listing + download (only reached when the cache is empty AND
             :meth:`download_to_cache` was not run first; gated repos need ``HF_TOKEN``).
        """
        if model_id.endswith(".nemo") or os.path.exists(model_id):
            return model_id

        # 2. Cache-only lookup: reads $HF_HOME/hub without any network call.
        from huggingface_hub import snapshot_download
        try:
            snap_dir = snapshot_download(model_id, local_files_only=True)
            cached = [f for f in os.listdir(snap_dir) if f.endswith(".nemo")]
            if cached:
                return os.path.join(snap_dir, cached[0])
        except Exception:  # noqa: BLE001, S110
            pass

        # 3. Online fallback (needs egress; gated repos need HF_TOKEN).
        from huggingface_hub import HfApi, hf_hub_download

        files = [f for f in HfApi().list_repo_files(model_id) if f.endswith(".nemo")]
        if not files:
            msg = f"No .nemo file found in HuggingFace repo '{model_id}'"
            raise RuntimeError(msg)
        return hf_hub_download(model_id, files[0])

    def setup(self) -> None:
        import nemo.collections.asr as nemo_asr
        import torch

        _apply_multisoftmax_patches()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.tensorrt_engine_dir is None:
            nemo_path = self._resolve_nemo_path(self.model_id)
        else:
            from nemo_curator.stages.audio.inference.indic_conformer_tensorrt import load_engine_metadata
            from nemo_curator.stages.audio.inference.tensorrt_encoder import ENGINE_FILENAME, MODEL_FILENAME

            if self._device.type != "cuda":
                msg = "IndicConformer TensorRT inference requires CUDA"
                raise RuntimeError(msg)
            engine_dir = Path(self.tensorrt_engine_dir)
            self._trt_metadata = load_engine_metadata(engine_dir)
            nemo_path = str(engine_dir / MODEL_FILENAME)
            engine_path = engine_dir / ENGINE_FILENAME
            if not Path(nemo_path).is_file():
                msg = f"Bundled NeMo model not found: {nemo_path}"
                raise FileNotFoundError(msg)
            if not engine_path.is_file():
                msg = f"TensorRT encoder engine not found: {engine_path}"
                raise FileNotFoundError(msg)
        logger.info(f"Loading IndicConformer hybrid model={nemo_path} device={self._device}")

        self._model = nemo_asr.models.ASRModel.restore_from(nemo_path, map_location=self._device)
        self._model.to(self._device)
        self._model.eval()
        self._chunk_duration_sec = _MAX_CHUNK_DURATION_SEC
        self._configure_rnnt_precision()

        if self._trt_metadata is not None:
            self._enable_tensorrt_encoder(engine_path)

        tok = self._model.tokenizer
        if not hasattr(tok, "langs_by_token_id"):
            msg = "Loaded model does not use an AggregateTokenizer; this wrapper expects the multilingual checkpoint."
            raise RuntimeError(msg)
        self._num_langs = len(tok.tokenizers_dict)
        self._per_lang_classes = self._model.joint._vocab_size // self._num_langs

        # Build the per-language CTC masks (token belongs to lang) + blank, then
        # hand them to the (patched) CTC decoder for masked decoding.
        masks: dict[str, list[bool]] = {}
        for lang in tok.tokenizers_dict:
            m = [tok.langs_by_token_id[i] == lang for i in range(len(tok.langs_by_token_id))]
            m.append(True)  # blank
            masks[lang] = m
        self._model.ctc_decoder.language_masks = masks
        logger.info(
            f"IndicConformer hybrid ready: {self._num_langs} langs, {self._per_lang_classes} tokens/lang"
        )

    def _enable_tensorrt_encoder(self, engine_path: Path) -> None:
        from nemo_curator.stages.audio.inference.tensorrt_encoder import TensorRTEncoder

        metadata = self._trt_metadata
        if metadata is None:
            msg = "TensorRT metadata is not loaded"
            raise RuntimeError(msg)
        encoder = self._model.encoder
        actual_feature_count = int(getattr(encoder, "_feat_in", self._model.cfg.encoder.feat_in))
        actual_subsampling = int(encoder.subsampling_factor)
        actual_sample_rate = int(self._model.cfg.preprocessor.sample_rate)
        actual_encoder_dim = int(self._model.cfg.encoder.d_model)
        expected = (
            ("feature_count", actual_feature_count),
            ("subsampling_factor", actual_subsampling),
            ("sample_rate", actual_sample_rate),
            ("encoder_dim", actual_encoder_dim),
        )
        for key, actual in expected:
            if actual != int(metadata[key]):
                msg = (
                    "Bundled NeMo model does not match the TensorRT engine: "
                    f"{key}={actual}, expected={metadata[key]}"
                )
                raise ValueError(msg)

        self._model.encoder = None
        del encoder
        gc.collect()
        import torch

        torch.cuda.empty_cache()
        self._trt_encoder = TensorRTEncoder(
            engine_path,
            subsampling_factor=int(metadata["subsampling_factor"]),
            max_batch_size=_TENSORRT_ENCODER_BATCH_SIZE,
        )
        max_feature_frames = self._trt_encoder.max_input_shape("audio_signal")[2]
        engine_duration = engine_chunk_duration(self._model, max_feature_frames)
        if engine_duration < _MAX_CHUNK_DURATION_SEC:
            msg = (
                "IndicConformer TensorRT engine does not support 40-second audio: "
                f"max_feature_frames={max_feature_frames}; rebuild with --max-frames 4001"
            )
            raise ValueError(msg)
        self._chunk_duration_sec = _MAX_CHUNK_DURATION_SEC
        self._model.encoder = self._trt_encoder
        logger.info(f"IndicConformer TensorRT encoder loaded: {engine_path}")

    def teardown(self) -> None:
        import torch

        for decoder in self._rnnt_decoders.values():
            if decoder.decoding_computer is not None:
                decoder.decoding_computer.reset_cuda_graphs_state()
        self._rnnt_decoders.clear()
        if self._trt_encoder is not None:
            self._trt_encoder.close()
            self._trt_encoder = None
        del self._model
        self._model = None
        self._device = None
        self._trt_metadata = None
        self._chunk_duration_sec = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _rnnt_dtype(self) -> Any:
        import torch

        return {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.rnnt_precision]

    def _configure_rnnt_precision(self) -> None:
        if self.rnnt_precision == "fp32":
            return
        import torch

        if self._device.type != "cuda":
            msg = f"IndicConformer {self.rnnt_precision.upper()} RNNT inference requires CUDA"
            raise RuntimeError(msg)
        if self.rnnt_precision == "bf16" and not torch.cuda.is_bf16_supported():
            msg = "IndicConformer BF16 RNNT inference is not supported by this GPU"
            raise RuntimeError(msg)
        rnnt_dtype = self._rnnt_dtype()
        self._model.decoder.to(dtype=rnnt_dtype)
        self._model.joint.to(dtype=rnnt_dtype)

    def generate(
        self,
        waveforms: list[np.ndarray],
        sample_rates: list[int],
        lang_codes: list[str],
        decode_mode: str | None = None,
    ) -> tuple[list[str], list[str]]:
        if self._model is None:
            msg = "Model not initialized. Call setup() first."
            raise RuntimeError(msg)
        mode = (decode_mode or self.decode_mode).lower()
        if self._chunk_duration_sec is None:
            msg = "IndicConformer chunk duration was not initialized from the model"
            raise RuntimeError(msg)
        if len(lang_codes) != len(waveforms):
            msg = "waveforms and lang_codes must have the same length"
            raise ValueError(msg)
        requires_merge = has_audio_longer_than(waveforms, sample_rates, self._chunk_duration_sec)

        chunks, chunk_sample_rates, owners = split_waveforms(
            waveforms,
            sample_rates,
            self._chunk_duration_sec,
        )
        if not chunks:
            return [""] * len(waveforms), list(lang_codes)
        chunk_langs = [lang_codes[owner] for owner in owners]
        chunk_texts, _ = self._generate_chunks(
            chunks,
            chunk_sample_rates,
            chunk_langs,
            mode,
        )
        if not requires_merge:
            texts = [""] * len(waveforms)
            for text, owner in zip(chunk_texts, owners, strict=True):
                texts[owner] = text
            return texts, list(lang_codes)
        return merge_chunk_texts(chunk_texts, owners, len(waveforms)), list(lang_codes)

    def _generate_chunks(
        self,
        waveforms: list[np.ndarray],
        sample_rates: list[int],
        lang_codes: list[str],
        mode: str,
    ) -> tuple[list[str], list[str]]:
        if self._trt_encoder is not None:
            return self._generate_tensorrt(waveforms, sample_rates, lang_codes, mode)
        import torch
        import torchaudio.functional as audio_functional

        texts: list[str] = [""] * len(waveforms)
        langs_out: list[str] = [str(lang).strip().lower() for lang in lang_codes]
        prepared: list[Any] = []
        lengths: list[int] = []
        prepared_langs: list[str] = []
        original_indices: list[int] = []
        with torch.inference_mode():
            for idx, (w, sr, lang) in enumerate(zip(waveforms, sample_rates, langs_out, strict=True)):
                if w is None or np.asarray(w).size == 0:
                    continue
                wav = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)).to(self._device)
                if wav.ndim > 1:
                    # Curator readers produce channels-first arrays; also handle
                    # the common channels-last layout without changing 1-D input.
                    wav = wav.mean(dim=0) if wav.shape[0] <= wav.shape[-1] else wav.mean(dim=-1)
                wav = wav.reshape(-1)
                if int(sr) != _TARGET_SR:
                    wav = audio_functional.resample(wav, orig_freq=int(sr), new_freq=_TARGET_SR)
                prepared.append(wav.contiguous())
                lengths.append(int(wav.shape[0]))
                prepared_langs.append(lang)
                original_indices.append(idx)

            duration_order = sorted(range(len(prepared)), key=lengths.__getitem__)
            prepared = [prepared[idx] for idx in duration_order]
            lengths = [lengths[idx] for idx in duration_order]
            prepared_langs = [prepared_langs[idx] for idx in duration_order]
            original_indices = [original_indices[idx] for idx in duration_order]

            for start in range(0, len(prepared), self.inference_batch_size):
                end = start + self.inference_batch_size
                chunk = prepared[start:end]
                chunk_lengths = lengths[start:end]
                chunk_langs = prepared_langs[start:end]
                chunk_indices = original_indices[start:end]
                padded = torch.nn.utils.rnn.pad_sequence(chunk, batch_first=True)
                length_tensor = torch.tensor(chunk_lengths, dtype=torch.long, device=self._device)
                encoded, encoded_len = self._model(input_signal=padded, input_signal_length=length_tensor)
                if mode == "ctc":
                    batch_texts = self._decode_ctc_batch(encoded, encoded_len, chunk_langs)
                else:
                    encoded = encoded.to(dtype=self._rnnt_dtype())
                    batch_texts = self._decode_rnnt_batch(encoded, encoded_len, chunk_langs)
                for original_idx, text in zip(chunk_indices, batch_texts, strict=True):
                    texts[original_idx] = text
        return texts, langs_out

    def _generate_tensorrt(
        self,
        waveforms: list[np.ndarray],
        sample_rates: list[int],
        lang_codes: list[str],
        mode: str,
    ) -> tuple[list[str], list[str]]:
        import torch
        import torchaudio.functional as audio_functional

        texts = [""] * len(waveforms)
        langs_out = list(lang_codes)
        prepared: list[tuple[int, torch.Tensor, str]] = []
        for index, (waveform, sample_rate, lang) in enumerate(
            zip(waveforms, sample_rates, lang_codes, strict=True)
        ):
            if waveform is None or np.asarray(waveform).size == 0:
                continue
            wav = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32)).to(self._device)
            if wav.ndim > 1:
                wav = wav.mean(dim=-1)
            if int(sample_rate) != _TARGET_SR:
                wav = audio_functional.resample(wav, orig_freq=int(sample_rate), new_freq=_TARGET_SR)
            prepared.append((index, wav, lang))

        prepared.sort(key=lambda item: item[1].shape[0])

        max_batch = min(self.inference_batch_size, _TENSORRT_INFERENCE_BATCH_SIZE)
        with torch.inference_mode():
            for start in range(0, len(prepared), max_batch):
                group = prepared[start : start + max_batch]
                lengths = torch.tensor([wav.shape[0] for _, wav, _ in group], device=self._device)
                signals = torch.nn.utils.rnn.pad_sequence(
                    [wav for _, wav, _ in group],
                    batch_first=True,
                )
                features, feature_lengths = self._model.preprocessor(
                    input_signal=signals,
                    length=lengths,
                )
                encoded, encoded_lengths = self._model.encoder(
                    audio_signal=features.to(dtype=torch.float16),
                    length=feature_lengths,
                )
                group_langs = [lang for _, _, lang in group]
                if mode == "ctc":
                    batch_texts = self._decode_ctc_batch(encoded.float(), encoded_lengths, group_langs)
                else:
                    encoded = encoded.to(dtype=self._rnnt_dtype())
                    batch_texts = self._decode_rnnt_batch(encoded, encoded_lengths, group_langs)
                for (output_index, _, _), text in zip(group, batch_texts, strict=True):
                    texts[output_index] = text
        return texts, langs_out

    def _ids_to_text(self, local_ids: list[int], lang: str) -> str:
        """Map per-language local token ids -> aggregate ids -> text."""
        if not local_ids:
            return ""
        offset = self._model.tokenizer.token_id_offset[lang]
        agg_ids = [int(i) + offset for i in local_ids]
        return self._model.tokenizer.ids_to_text(agg_ids).strip()

    def _decode_ctc(self, encoded: Any, encoded_len: Any, lang: str) -> str:
        log_probs = self._model.ctc_decoder(encoder_output=encoded, language_ids=[lang])  # [1, T, per_lang+1]
        elen = int(encoded_len[0].item())
        return self._decode_ctc_row(log_probs[0], elen, lang)

    def _decode_ctc_batch(self, encoded: Any, encoded_len: Any, lang_codes: list[str]) -> list[str]:
        log_probs = self._model.ctc_decoder(encoder_output=encoded, language_ids=lang_codes)
        return [
            self._decode_ctc_row(log_probs[i], int(encoded_len[i].item()), lang)
            for i, lang in enumerate(lang_codes)
        ]

    def _decode_ctc_row(self, log_probs: Any, encoded_len: int, lang: str) -> str:
        preds = log_probs[:encoded_len].argmax(dim=-1).tolist()
        blank = self._per_lang_classes  # per-language blank sits at the last index
        out: list[int] = []
        prev = None
        for p in preds:
            if p != blank and p != prev:  # noqa: PLR1714
                out.append(p)
            prev = p
        return self._ids_to_text(out, lang)

    def _rnnt_decoder(self, lang: str) -> Any:
        decoder = self._rnnt_decoders.get(lang)
        if decoder is not None:
            return decoder

        from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import GreedyBatchedRNNTInfer

        decoder = GreedyBatchedRNNTInfer(
            decoder_model=_LanguageRNNTDecoder(self._model.decoder, self._per_lang_classes),
            joint_model=_LanguageRNNTJoint(
                self._model.joint,
                lang,
                self._per_lang_classes + 1,
            ),
            blank_index=self._per_lang_classes,
            max_symbols_per_step=self.max_symbols_per_step,
            preserve_alignments=False,
            preserve_frame_confidence=False,
            loop_labels=True,
            use_cuda_graph_decoder=self._device.type == "cuda",
        )
        self._rnnt_decoders[lang] = decoder
        return decoder

    def _decode_rnnt_batch(self, encoded: Any, encoded_len: Any, lang_codes: list[str]) -> list[str]:
        import torch

        with torch.inference_mode():
            batch_size = len(lang_codes)
            if batch_size == 0:
                return []
            language_groups: dict[str, list[int]] = {}
            for index, lang in enumerate(lang_codes):
                language_groups.setdefault(lang, []).append(index)

            texts = [""] * batch_size
            for lang, indices in language_groups.items():
                if len(indices) == batch_size:
                    group_encoded = encoded
                    group_lengths = encoded_len
                else:
                    index_tensor = torch.tensor(indices, dtype=torch.long, device=encoded.device)
                    group_encoded = encoded.index_select(0, index_tensor)
                    group_lengths = encoded_len.index_select(0, index_tensor)

                hypotheses = self._rnnt_decoder(lang)(
                    encoder_output=group_encoded,
                    encoded_lengths=group_lengths,
                )[0]
                for index, hypothesis in zip(indices, hypotheses, strict=True):
                    token_ids = hypothesis.y_sequence
                    if torch.is_tensor(token_ids):
                        token_ids = token_ids.tolist()
                    texts[index] = self._ids_to_text(token_ids, lang)
            return texts


@dataclass
class InferenceIndicConformerHybridStage(ProcessingStage[AudioTask, AudioTask]):
    """Audio transcription with an AI4Bharat IndicConformer hybrid (CTC+RNNT) model.

    Pipeline adapter over :class:`IndicConformerHybridASR` (same module): reads
    in-memory waveforms from each ``AudioTask``, routes per-sample by ``source_lang``,
    and writes the predicted transcription.

    Args:
        model_id: Local ``.nemo`` path or HuggingFace repo id (gated; set ``HF_TOKEN``),
            e.g. ``ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large``.
        decode_mode: ``"ctc"`` or ``"rnnt"`` (model card recommends rnnt).
        backend: ``"nemo"`` for the existing implementation or ``"tensorrt"``
            for batched inference through an optimized encoder bundle.
        tensorrt_engine_dir: Directory containing ``encoder.plan``, ``model.nemo``,
            and ``metadata.json``. Required when ``backend="tensorrt"``.
        rnnt_precision: Precision for the RNNT prediction and joint networks.
            Defaults to ``"fp32"``; ``"fp16"`` and ``"bf16"`` require CUDA.
        inference_batch_size: Maximum NeMo inference batch size. When unset, uses
            ``batch_size``. TensorRT remains capped by the engine profile.
        source_lang_key: Task key holding the per-sample ISO language code.
        keep_waveform: When True the waveform is left on the task for a later stage.
    """

    name: str = "IndicConformerHybrid_inference"
    model_id: str = "ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large"
    decode_mode: Literal["ctc", "rnnt"] = "rnnt"
    backend: Literal["nemo", "tensorrt"] = "nemo"
    tensorrt_engine_dir: str | None = None
    rnnt_precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    source_lang_key: str = "source_lang"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sampling_rate"
    pred_text_key: str = "asr_prediction"
    language_key: str = "asr_language"
    notes_key: str = "additional_notes"
    keep_waveform: bool = False
    num_workers_override: int | None = None
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    batch_size: int = 128
    inference_batch_size: int | None = None
    _model: IndicConformerHybridASR | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.decode_mode not in {"ctc", "rnnt"}:
            msg = f"Unsupported IndicConformer decode mode: {self.decode_mode!r}"
            raise ValueError(msg)
        if self.backend not in {"nemo", "tensorrt"}:
            msg = f"Unsupported IndicConformer inference backend: {self.backend!r}"
            raise ValueError(msg)
        if self.rnnt_precision not in {"fp32", "fp16", "bf16"}:
            msg = f"Unsupported IndicConformer RNNT precision: {self.rnnt_precision!r}"
            raise ValueError(msg)
        if self.backend == "tensorrt" and not self.tensorrt_engine_dir:
            msg = "tensorrt_engine_dir is required when backend='tensorrt'"
            raise ValueError(msg)

    def num_workers(self) -> int | None:
        return self.num_workers_override

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if self.num_workers_override is not None:
            spec["num_workers"] = self.num_workers_override
        return spec

    def _create_model(self) -> IndicConformerHybridASR:
        return IndicConformerHybridASR(
            model_id=self.model_id,
            decode_mode=self.decode_mode,
            tensorrt_engine_dir=self.tensorrt_engine_dir if self.backend == "tensorrt" else None,
            rnnt_precision=self.rnnt_precision,
            inference_batch_size=(
                self.batch_size if self.inference_batch_size is None else self.inference_batch_size
            ),
        )

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        if self.backend == "tensorrt":
            from nemo_curator.stages.audio.inference.indic_conformer_tensorrt import load_engine_metadata

            engine_dir = self.tensorrt_engine_dir
            if engine_dir is None:
                msg = "tensorrt_engine_dir is required when backend='tensorrt'"
                raise ValueError(msg)
            load_engine_metadata(engine_dir)
        else:
            # Download the checkpoint into the shared HF cache exactly once per node.
            IndicConformerHybridASR.download_to_cache(self.model_id)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._model is None:
            self._model = self._create_model()
            self._model.setup()
            logger.info(f"Indic Conformer hybrid model ready: {self.model_id}")

    def teardown(self) -> None:
        if self._model is not None:
            self._model.teardown()
            self._model = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.waveform_key, self.sample_rate_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.pred_text_key, self.language_key]

    def process(self, task: AudioTask) -> AudioTask:
        msg = "InferenceIndicConformerHybridStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:  # noqa: C901
        if len(tasks) == 0:
            return []
        if self._model is None:
            msg = "Model not initialized — setup() was not called"
            raise RuntimeError(msg)

        for task in tasks:
            task.data.setdefault(self.pred_text_key, "")
            task.data.setdefault(self.language_key, "")

        eligible_indices: list[int] = []
        for i, task in enumerate(tasks):
            lang = str(task.data.get(self.source_lang_key, "") or "").strip().lower()
            if lang not in INDIC_CONFORMER_HYBRID_LANGS:
                set_note(task.data, self.name, f"skipped (unsupported language: {lang})", self.notes_key)
                set_note(task.data, self.pred_text_key, f"lang_not_supported:{lang}", self.notes_key)
            else:
                eligible_indices.append(i)

        lang_skipped = len(tasks) - len(eligible_indices)
        if not eligible_indices:
            if not self.keep_waveform:
                for task in tasks:
                    task.data.pop(self.waveform_key, None)
            logger.info(f"{self.name}: skipped entire batch of {len(tasks)} (no supported languages)")
            return tasks

        eligible_tasks = [tasks[i] for i in eligible_indices]
        waveforms = [t.data[self.waveform_key] for t in eligible_tasks]
        sample_rates = [t.data[self.sample_rate_key] for t in eligible_tasks]
        lang_codes = [
            str(t.data.get(self.source_lang_key, "") or "").strip().lower() for t in eligible_tasks
        ]

        pred_texts, langs_out = self._model.generate(waveforms, sample_rates, lang_codes)

        for task_idx, pred, lang in zip(eligible_indices, pred_texts, langs_out, strict=True):
            tasks[task_idx].data[self.pred_text_key] = pred
            tasks[task_idx].data[self.language_key] = lang

        if not self.keep_waveform:
            for task in tasks:
                task.data.pop(self.waveform_key, None)

        logger.info(
            f"{self.name}: generated {len(eligible_indices)} predictions, "
            f"skipped {lang_skipped} (unsupported language)"
        )
        return tasks
