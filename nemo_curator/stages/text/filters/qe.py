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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import pandas as pd
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch

QEInput = Any
QEMode = Literal["simple", "always_en_x", "bidi"]

COMET_IMPORT_MSG = (
    "To run QE filtering with COMET, install the translation_qe extra or install `unbabel-comet`. "
    "More information: https://github.com/Unbabel/COMET."
)
PYMARIAN_IMPORT_MSG = (
    "To run QE filtering with Cometoid/PyMarian, install the translation_qe extra or install `pymarian`. "
    "More information: https://github.com/marian-nmt/wmt23-metrics?tab=readme-ov-file#setup."
)


class QEModel(ABC):
    """Abstract model wrapper for bitext quality estimation."""

    @staticmethod
    @abstractmethod
    def wrap_qe_input(src: str, tgt: str, reverse: bool = False) -> QEInput:
        """Wrap one source-target pair into the model-specific input format."""

    @abstractmethod
    def predict(self, inputs: list[QEInput]) -> list[float]:
        """Predict quality scores for wrapped bitext inputs."""


class COMETQEModel(QEModel):
    """Wrapper for COMET quality-estimation models."""

    MODEL_NAME_TO_HF_PATH: ClassVar[dict[str, str]] = {
        "comet-qe": "Unbabel/wmt20-comet-qe-da",
    }

    def __init__(self, model_name: str, model: Any, gpu: bool = False) -> None:  # noqa: ANN401
        self.model_name = model_name
        self.model = model
        self.gpu = gpu

    @classmethod
    def load_model(cls, model_name: str, gpu: bool = False, **_: object) -> "COMETQEModel":
        """Load a COMET QE model by supported short name."""
        try:
            import comet
        except ImportError as e:
            raise ImportError(COMET_IMPORT_MSG) from e

        model_path = comet.download_model(cls.MODEL_NAME_TO_HF_PATH[model_name])
        return cls(model_name, comet.load_from_checkpoint(model_path), gpu=gpu)

    @staticmethod
    def wrap_qe_input(src: str, tgt: str, reverse: bool = False) -> dict[str, str]:
        return {"src": tgt, "mt": src} if reverse else {"src": src, "mt": tgt}

    def predict(self, inputs: list[QEInput]) -> list[float]:
        result = self.model.predict(inputs, gpus=int(self.gpu), num_workers=0)
        return list(result.scores)


class PyMarianQEModel(QEModel):
    """Wrapper for Cometoid/PyMarian quality-estimation models."""

    MODEL_NAME_TO_HF_PATH: ClassVar[dict[str, str]] = {
        "cometoid-wmt23": "marian-nmt/cometoid22-wmt23",
        "cometoid-wmt23-mqm": "marian-nmt/cometoid22-wmt23",
    }
    DEFAULT_GPU_ARGS: ClassVar[str] = "-w 8000 --mini-batch 32 -d 0"
    DEFAULT_CPU_ARGS: ClassVar[str] = "--cpu-threads 1 -w 2000"

    def __init__(self, model_name: str, evaluator: Any, gpu: bool = False, shard_size: int = 5000) -> None:  # noqa: ANN401
        self.model_name = model_name
        self.evaluator = evaluator
        self.gpu = gpu
        self.shard_size = shard_size

    @classmethod
    def load_model(
        cls,
        model_name: str,
        gpu: bool = False,
        shard_size: int = 5000,
        marian_args: str | None = None,
    ) -> "PyMarianQEModel":
        """Load a PyMarian-backed Cometoid QE model by supported short name."""
        try:
            import pymarian
        except ImportError as e:
            raise ImportError(PYMARIAN_IMPORT_MSG) from e

        from huggingface_hub import hf_hub_download

        repo_id = cls.MODEL_NAME_TO_HF_PATH[model_name]
        model_path = hf_hub_download(repo_id, filename="checkpoints/marian.model.bin")
        vocab_path = hf_hub_download(repo_id, filename="vocab.spm")
        args = f"-m {model_path} -v {vocab_path} {vocab_path} --like comet-qe"
        args += (
            f" {marian_args}"
            if marian_args is not None
            else f" {cls.DEFAULT_GPU_ARGS if gpu else cls.DEFAULT_CPU_ARGS}"
        )
        return cls(model_name, pymarian.Evaluator(args), gpu=gpu, shard_size=shard_size)

    @staticmethod
    def wrap_qe_input(src: str, tgt: str, reverse: bool = False) -> list[str]:
        return [tgt, src] if reverse else [src, tgt]

    def predict(self, inputs: list[QEInput]) -> list[float]:
        scores: list[float] = []
        for start_idx in range(0, len(inputs), self.shard_size):
            shard_inputs = inputs[start_idx : start_idx + self.shard_size]
            shard_lines = [self._format_input_line(input_pair) for input_pair in shard_inputs]
            shard_scores = self.evaluator.evaluate(shard_lines)
            scores.extend(self._normalize_score(score) for score in shard_scores)
        return scores

    @staticmethod
    def _format_input_line(input_pair: QEInput) -> str:
        if isinstance(input_pair, list | tuple):
            return "\t".join(str(field) for field in input_pair)
        return str(input_pair)

    @staticmethod
    def _normalize_score(score: Any) -> float:  # noqa: ANN401
        if isinstance(score, list | tuple):
            return float(score[0])
        return float(score)


@dataclass
class QualityEstimationFilter(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Score and filter aligned bitext rows with a QE model.

    Rows already marked with ``skip_field != 0`` are not sent to the model when
    ``mark_only`` is enabled, allowing expensive QE stages to follow cheap CPU
    filters without wasting model calls.
    """

    model_name: str
    cutoff: float
    mode: QEMode = "always_en_x"
    gpu: bool = False
    src_field: str = "src"
    tgt_field: str = "tgt"
    src_lang_field: str = "src_lang"
    tgt_lang_field: str = "tgt_lang"
    score_field: str = "qe_score"
    mark_only: bool = True
    skip_field: str = "_skipme"
    reason_field: str = "reason"
    model: QEModel | None = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    SUPPORTED_MODELS: ClassVar[dict[str, type[QEModel]]] = {
        "comet-qe": COMETQEModel,
        "cometoid-wmt23": PyMarianQEModel,
        "cometoid-wmt23-mqm": PyMarianQEModel,
    }

    def __post_init__(self) -> None:
        if self.mode not in {"simple", "always_en_x", "bidi"}:
            msg = "mode must be one of: simple, always_en_x, bidi"
            raise ValueError(msg)
        if self.model is None and self.model_name not in self.SUPPORTED_MODELS:
            msg = f"Unsupported QE model: {self.model_name}. Supported models: {sorted(self.SUPPORTED_MODELS)}"
            raise ValueError(msg)
        self.name = "quality_estimation_filter"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.src_field, self.tgt_field, self.src_lang_field, self.tgt_lang_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        fields = [self.score_field]
        if self.mark_only:
            fields.extend([self.skip_field, self.reason_field])
        return ["data"], fields

    def setup(self, _: Any = None) -> None:  # noqa: ANN401
        self._get_model()

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas().copy()
        if df.empty:
            logger.info(f"Empty dataset for batch {batch.task_id}")
            return self._make_output_batch(batch, df)

        process_mask = self._get_process_mask(df)
        scores = self._score_rows(df.loc[process_mask])
        if self.score_field not in df.columns:
            df[self.score_field] = pd.NA
        df.loc[scores.index, self.score_field] = scores

        keep_mask = scores.apply(self.keep_bitext)
        if self.mark_only:
            df = self._mark_rejected_rows(df, keep_mask)
        else:
            df = df.loc[keep_mask[keep_mask].index]
            if df.empty:
                logger.info(f"All bitext pairs filtered out for batch {batch.task_id}")

        return self._make_output_batch(batch, df)

    def keep_bitext(self, score: float) -> bool:
        return score >= self.cutoff

    def _get_model(self) -> QEModel:
        if self.model is not None:
            return self.model
        model_cls = self.SUPPORTED_MODELS[self.model_name]
        self.model = model_cls.load_model(self.model_name, gpu=self.gpu, **self.model_kwargs)
        return self.model

    def _get_process_mask(self, df: pd.DataFrame) -> pd.Series:
        if not self.mark_only:
            return pd.Series(True, index=df.index)

        if self.skip_field not in df.columns:
            df[self.skip_field] = 0
        if self.reason_field not in df.columns:
            df[self.reason_field] = None
        return df[self.skip_field].fillna(0).eq(0)

    def _score_rows(self, rows: pd.DataFrame) -> pd.Series:
        if rows.empty:
            return pd.Series(dtype=float)

        model = self._get_model()
        scores = self._predict_scores(model, rows)
        return pd.Series(scores, index=rows.index)

    def _predict_scores(self, model: QEModel, rows: pd.DataFrame) -> list[float]:
        src_values = rows[self.src_field].astype(str).tolist()
        tgt_values = rows[self.tgt_field].astype(str).tolist()

        if self.mode == "simple":
            inputs = [model.wrap_qe_input(src, tgt) for src, tgt in zip(src_values, tgt_values, strict=True)]
            return self._predict_model(model, inputs)

        if self.mode == "always_en_x":
            src_langs = rows[self.src_lang_field].astype(str).str.lower().tolist()
            tgt_langs = rows[self.tgt_lang_field].astype(str).str.lower().tolist()
            inputs = [
                model.wrap_qe_input(src, tgt, reverse=(src_lang != "en" and tgt_lang == "en"))
                for src, tgt, src_lang, tgt_lang in zip(src_values, tgt_values, src_langs, tgt_langs, strict=True)
            ]
            return self._predict_model(model, inputs)

        forward_inputs = [model.wrap_qe_input(src, tgt) for src, tgt in zip(src_values, tgt_values, strict=True)]
        reverse_inputs = [
            model.wrap_qe_input(src, tgt, reverse=True) for src, tgt in zip(src_values, tgt_values, strict=True)
        ]
        scores = self._predict_model(model, forward_inputs + reverse_inputs)
        midpoint = len(forward_inputs)
        return [(forward + reverse) / 2 for forward, reverse in zip(scores[:midpoint], scores[midpoint:], strict=True)]

    @staticmethod
    def _predict_model(model: QEModel, inputs: list[QEInput]) -> list[float]:
        scores = [float(score) for score in model.predict(inputs)]
        if len(scores) != len(inputs):
            msg = f"QE model returned {len(scores)} scores for {len(inputs)} inputs"
            raise RuntimeError(msg)
        return scores

    def _mark_rejected_rows(self, df: pd.DataFrame, keep_mask: pd.Series) -> pd.DataFrame:
        rejected_indices = keep_mask[~keep_mask].index
        processed_indices = keep_mask.index
        df.loc[processed_indices, self.skip_field] = 0
        df.loc[rejected_indices, self.skip_field] = 1
        df.loc[rejected_indices, self.reason_field] = self.__class__.__name__
        return df

    def _make_output_batch(self, batch: DocumentBatch, df: pd.DataFrame) -> DocumentBatch:
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


__all__ = ["COMETQEModel", "PyMarianQEModel", "QEModel", "QualityEstimationFilter"]
