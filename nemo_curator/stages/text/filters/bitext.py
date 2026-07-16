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

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.utils.text_utils import get_word_splitter
from nemo_curator.tasks import DocumentBatch


class BitextFilter(ABC):
    """Base class for filters that score an aligned source/target text pair."""

    def __init__(self) -> None:
        super().__init__()
        self._name = self.__class__.__name__

    @abstractmethod
    def score_bitext(self, src: str, tgt: str) -> Any:  # noqa: ANN401
        """Calculate a score from one aligned source/target pair."""

    @abstractmethod
    def keep_bitext(self, score: Any) -> bool:  # noqa: ANN401
        """Return True if the aligned pair should be kept for the score."""

    @property
    def name(self) -> str:
        return self._name


class LengthRatioFilter(BitextFilter):
    """Filter bitext rows whose source/target token counts differ too much."""

    def __init__(self, max_ratio: float = 3.0, src_lang: str = "en", tgt_lang: str = "en") -> None:
        super().__init__()
        self._max_ratio = float(max_ratio)
        self._src_word_splitter = get_word_splitter(src_lang)
        self._tgt_word_splitter = get_word_splitter(tgt_lang)
        self._name = "length_ratio"

    def score_bitext(self, src: str, tgt: str) -> float:
        src_len = len(self._src_word_splitter(src.strip()))
        tgt_len = len(self._tgt_word_splitter(tgt.strip()))
        if src_len == 0 or tgt_len == 0:
            return math.inf
        return max(src_len / tgt_len, tgt_len / src_len)

    def keep_bitext(self, score: float) -> bool:
        return score < self._max_ratio


@dataclass
class BitextScoreFilter(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Score and filter aligned source/target text rows.

    When ``mark_only`` is false, rejected rows are dropped. When ``mark_only`` is
    true, rejected rows are kept but marked with ``skip_field=1`` and the first
    rejection reason. Rows that are already marked are not rescored, which lets
    pipelines put cheap filters ahead of expensive model-based filters.
    """

    filter_obj: BitextFilter
    src_field: str = "src"
    tgt_field: str = "tgt"
    score_field: str | None = None
    mark_only: bool = False
    skip_field: str = "_skipme"
    reason_field: str = "reason"
    invert: bool = False
    name: str = "bitext_score_filter"

    def __post_init__(self) -> None:
        self.name = self.filter_obj.name

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.src_field, self.tgt_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        fields: list[str] = []
        if self.score_field is not None:
            fields.append(self.score_field)
        if self.mark_only:
            fields.extend([self.skip_field, self.reason_field])
        return ["data"], fields

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas().copy()
        if df.empty:
            logger.info(f"Empty dataset for batch {batch.task_id}")
            return self._make_output_batch(batch, df)

        process_mask = self._get_process_mask(df)
        processed_scores = self._score_rows(df, process_mask)
        if self.score_field is not None:
            if self.score_field not in df.columns:
                df[self.score_field] = pd.NA
            df.loc[processed_scores.index, self.score_field] = processed_scores

        processed_keep_mask = processed_scores.apply(self.filter_obj.keep_bitext)
        if self.invert:
            processed_keep_mask = ~processed_keep_mask

        if self.mark_only:
            df = self._mark_rejected_rows(df, processed_keep_mask)
        else:
            df = df.loc[processed_keep_mask[processed_keep_mask].index]
            if len(df) == 0:
                logger.info(f"All bitext pairs filtered out for batch {batch.task_id}")

        return self._make_output_batch(batch, df)

    def _get_process_mask(self, df: pd.DataFrame) -> pd.Series:
        if not self.mark_only:
            return pd.Series(True, index=df.index)

        if self.skip_field not in df.columns:
            df[self.skip_field] = 0
        if self.reason_field not in df.columns:
            df[self.reason_field] = None
        return df[self.skip_field].fillna(0).eq(0)

    def _score_rows(self, df: pd.DataFrame, process_mask: pd.Series) -> pd.Series:
        rows_to_score = df.loc[process_mask, [self.src_field, self.tgt_field]]
        scores = [
            self.filter_obj.score_bitext(str(row[self.src_field]), str(row[self.tgt_field]))
            for _, row in rows_to_score.iterrows()
        ]
        return pd.Series(scores, index=rows_to_score.index)

    def _mark_rejected_rows(self, df: pd.DataFrame, processed_keep_mask: pd.Series) -> pd.DataFrame:
        rejected_indices = processed_keep_mask[~processed_keep_mask].index
        processed_indices = processed_keep_mask.index

        df.loc[processed_indices, self.skip_field] = 0
        df.loc[rejected_indices, self.skip_field] = 1
        df.loc[rejected_indices, self.reason_field] = self.filter_obj.__class__.__name__
        return df

    def _make_output_batch(self, batch: DocumentBatch, df: pd.DataFrame) -> DocumentBatch:
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


__all__ = ["BitextFilter", "BitextScoreFilter", "LengthRatioFilter"]
