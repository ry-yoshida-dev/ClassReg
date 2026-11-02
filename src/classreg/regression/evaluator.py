from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import override

from ..evaluation import (
    DatasetSplitter,
    EvaluableParametersLike,
    KFoldSplitter,
    MetricLike,
    ModelComparison,
    SupervisedEvaluator,
)
from .dataset import RegressionDataset
from .methods import BaselineParameters
from .metrics import MetricValues, RegressionMetric
from .types import NumericArray, TargetArray, TargetScalar


@dataclass(frozen=True)
class RegressionEvaluator(SupervisedEvaluator[TargetScalar, MetricValues]):
    """
    Evaluator with shuffled k-fold splits and a mean baseline by default.

    Attributes
    ----------
    splitter : DatasetSplitter
        The strategy that partitions the dataset.
    baseline : EvaluableParametersLike[TargetScalar, MetricValues] | None
        The reference model evaluated on the same splits; None to skip it.
    """

    splitter: DatasetSplitter = field(default_factory=KFoldSplitter)
    baseline: EvaluableParametersLike[TargetScalar, MetricValues] | None = field(
        default_factory=BaselineParameters
    )

    @override
    def compare(
        self,
        candidates: Mapping[str, EvaluableParametersLike[TargetScalar, MetricValues]],
        X: NumericArray,
        y: TargetArray,
        metric: MetricLike[MetricValues] = RegressionMetric.RMSE,
    ) -> ModelComparison[MetricValues]:
        """
        Rank model configurations evaluated on identical splits.

        Parameters
        ----------
        candidates : Mapping[str, EvaluableParametersLike[TargetScalar, MetricValues]]
            The model configurations by name.
        X : NumericArray
            The feature matrix.
        y : TargetArray
            The target values.
        metric : MetricLike[MetricValues]
            The metric to rank by, using its mean over the test partitions.

        Returns
        -------
        ModelComparison[MetricValues]
            The candidates, best first.
        """
        return super().compare(candidates, X, y, metric)

    @override
    def _build_dataset(self, X: NumericArray, y: TargetArray) -> RegressionDataset:
        return RegressionDataset(X=X, y=y)

    @override
    def _list_metrics(self) -> tuple[MetricLike[MetricValues], ...]:
        return tuple(RegressionMetric)
