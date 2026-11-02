from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..dataset import SupervisedDataset
from ..types import NumericArray
from .comparison import ModelComparison
from .data_split import DataSplit
from .fold import FoldEvaluation
from .metric_summary import MetricSummary
from .model_evaluation import ModelEvaluation
from .protocols import EvaluableParametersLike, EvaluableProcessorLike, MetricLike
from .ranked_model import RankedModel
from .report import EvaluationReport
from .splitter import DatasetSplitter


@dataclass(frozen=True)
class SupervisedEvaluator[TargetT: np.generic, MetricsT](ABC):
    """
    Estimates generalization performance by training on splits of a dataset.

    Each fold builds a fresh processor from the parameters, so nothing carries
    over between folds and all preprocessing is fitted on training samples only.

    Attributes
    ----------
    splitter : DatasetSplitter
        The strategy that partitions the dataset.
    baseline : EvaluableParametersLike[TargetT, MetricsT] | None
        The reference model evaluated on the same splits; None to skip it.
    """

    splitter: DatasetSplitter
    baseline: EvaluableParametersLike[TargetT, MetricsT] | None

    def evaluate(
        self,
        parameters: EvaluableParametersLike[TargetT, MetricsT],
        X: NumericArray,
        y: NDArray[TargetT],
    ) -> EvaluationReport[MetricsT]:
        """
        Evaluate one model configuration.

        Parameters
        ----------
        parameters : EvaluableParametersLike[TargetT, MetricsT]
            The model configuration.
        X : NumericArray
            The feature matrix.
        y : NDArray[TargetT]
            The targets.

        Returns
        -------
        EvaluationReport[MetricsT]
            The per-fold results and the baseline.

        Raises
        ------
        ValueError
            If the data is invalid or a split cannot be trained and tested on.
        """
        dataset: SupervisedDataset[TargetT] = self._build_dataset(X, y)
        splits: tuple[DataSplit, ...] = self.splitter.split(dataset.X, dataset.y)
        return EvaluationReport(
            model=self._evaluate_on_splits(parameters, dataset, splits),
            baseline=self._evaluate_baseline(dataset, splits),
            metrics=self._list_metrics(),
        )

    def compare(
        self,
        candidates: Mapping[str, EvaluableParametersLike[TargetT, MetricsT]],
        X: NumericArray,
        y: NDArray[TargetT],
        metric: MetricLike[MetricsT],
    ) -> ModelComparison[MetricsT]:
        """
        Rank model configurations evaluated on identical splits.

        Parameters
        ----------
        candidates : Mapping[str, EvaluableParametersLike[TargetT, MetricsT]]
            The model configurations by name.
        X : NumericArray
            The feature matrix.
        y : NDArray[TargetT]
            The targets.
        metric : MetricLike[MetricsT]
            The metric to rank by, using its mean over the test partitions.

        Returns
        -------
        ModelComparison[MetricsT]
            The candidates, best first.

        Raises
        ------
        ValueError
            If there are no candidates, the data is invalid, or the metric is
            undefined in any test fold of a candidate, which would make its mean
            incomparable to the others.
        """
        if not candidates:
            raise ValueError("candidates must not be empty")
        dataset: SupervisedDataset[TargetT] = self._build_dataset(X, y)
        splits: tuple[DataSplit, ...] = self.splitter.split(dataset.X, dataset.y)
        evaluated: list[RankedModel[MetricsT]] = [
            self._rank_candidate(name, parameters, dataset, splits, metric)
            for name, parameters in candidates.items()
        ]
        ranking: list[RankedModel[MetricsT]] = sorted(
            evaluated,
            key=lambda entry: entry.score.mean,
            reverse=metric.is_higher_better,
        )
        return ModelComparison(
            metric=metric,
            ranking=tuple(ranking),
            baseline=self._evaluate_baseline(dataset, splits),
        )

    @abstractmethod
    def _build_dataset(
        self, X: NumericArray, y: NDArray[TargetT]
    ) -> SupervisedDataset[TargetT]: ...

    @abstractmethod
    def _list_metrics(self) -> tuple[MetricLike[MetricsT], ...]: ...

    def _rank_candidate(
        self,
        name: str,
        parameters: EvaluableParametersLike[TargetT, MetricsT],
        dataset: SupervisedDataset[TargetT],
        splits: tuple[DataSplit, ...],
        metric: MetricLike[MetricsT],
    ) -> RankedModel[MetricsT]:
        evaluation: ModelEvaluation[MetricsT] = self._evaluate_on_splits(
            parameters, dataset, splits
        )
        score: MetricSummary = evaluation.summarize(metric)
        if score.is_partial:
            raise ValueError(
                f"cannot rank by {metric.value}: candidate {name!r} has it in only {score.num_folds} of {score.num_total_folds} test folds"
            )
        return RankedModel(name=name, evaluation=evaluation, score=score)

    def _evaluate_baseline(
        self,
        dataset: SupervisedDataset[TargetT],
        splits: tuple[DataSplit, ...],
    ) -> ModelEvaluation[MetricsT] | None:
        if self.baseline is None:
            return None
        return self._evaluate_on_splits(self.baseline, dataset, splits)

    def _evaluate_on_splits(
        self,
        parameters: EvaluableParametersLike[TargetT, MetricsT],
        dataset: SupervisedDataset[TargetT],
        splits: tuple[DataSplit, ...],
    ) -> ModelEvaluation[MetricsT]:
        folds: tuple[FoldEvaluation[MetricsT], ...] = tuple(
            self._evaluate_fold(parameters, dataset, split, fold_index)
            for fold_index, split in enumerate(splits)
        )
        return ModelEvaluation(method=parameters.build_processor().method, folds=folds)

    @staticmethod
    def _evaluate_fold(
        parameters: EvaluableParametersLike[TargetT, MetricsT],
        dataset: SupervisedDataset[TargetT],
        split: DataSplit,
        fold_index: int,
    ) -> FoldEvaluation[MetricsT]:
        train: SupervisedDataset[TargetT] = dataset.subset(split.train_indices)
        test: SupervisedDataset[TargetT] = dataset.subset(split.test_indices)
        processor: EvaluableProcessorLike[TargetT, MetricsT] = (
            parameters.build_processor()
        )
        processor.fit(train.X, train.y)
        return FoldEvaluation(
            fold_index=fold_index,
            train_metrics=processor.evaluate(train.X, train.y),
            test_metrics=processor.evaluate(test.X, test.y),
        )
