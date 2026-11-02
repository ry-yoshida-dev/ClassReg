from __future__ import annotations

from dataclasses import dataclass

from .fold import FoldEvaluation
from .metric_summary import MetricSummary
from .partition import EvaluationPartition
from .protocols import MethodLike, MetricLike


@dataclass(frozen=True)
class ModelEvaluation[MetricsT]:
    """
    Performance of one model configuration across all folds.

    Attributes
    ----------
    method : MethodLike
        The evaluated method.
    folds : tuple[FoldEvaluation[MetricsT], ...]
        The non-empty result of each fold.

    Raises
    ------
    ValueError
        If there are no folds.
    """

    method: MethodLike
    folds: tuple[FoldEvaluation[MetricsT], ...]

    def __post_init__(self) -> None:
        if not self.folds:
            raise ValueError("folds must not be empty")

    def is_defined(
        self,
        metric: MetricLike[MetricsT],
        partition: EvaluationPartition = EvaluationPartition.TEST,
    ) -> bool:
        """
        Check whether a metric is defined in at least one fold.

        Parameters
        ----------
        metric : MetricLike[MetricsT]
            The metric.
        partition : EvaluationPartition
            The partition to look at.

        Returns
        -------
        bool
            True if summarize would succeed for this metric and partition.
        """
        return bool(self._collect_values(metric, partition))

    def get_defined_fold_indices(
        self,
        metric: MetricLike[MetricsT],
        partition: EvaluationPartition = EvaluationPartition.TEST,
    ) -> tuple[int, ...]:
        """
        Get the folds in which a metric is defined.

        Means of a metric are only comparable between evaluations that are
        defined on the same folds.

        Parameters
        ----------
        metric : MetricLike[MetricsT]
            The metric.
        partition : EvaluationPartition
            The partition to look at.

        Returns
        -------
        tuple[int, ...]
            The indices of the folds in which the metric has a value.
        """
        return tuple(
            fold.fold_index
            for fold in self.folds
            if metric.read_from(fold.get_metrics(partition)) is not None
        )

    def summarize(
        self,
        metric: MetricLike[MetricsT],
        partition: EvaluationPartition = EvaluationPartition.TEST,
    ) -> MetricSummary:
        """
        Summarize a metric over the folds in which it is defined.

        The summary states how many folds it covers, so that a mean over a
        subset of the folds is recognizable as such.

        Parameters
        ----------
        metric : MetricLike[MetricsT]
            The metric.
        partition : EvaluationPartition
            The partition to look at.

        Returns
        -------
        MetricSummary
            The summary.

        Raises
        ------
        ValueError
            If the metric is undefined in every fold.
        """
        values: list[float] = self._collect_values(metric, partition)
        if not values:
            raise ValueError(
                f"{metric.value} is undefined for {self.method.value} on the {partition.value} partition"
            )
        return MetricSummary.from_values(values, num_total_folds=len(self.folds))

    def _collect_values(
        self,
        metric: MetricLike[MetricsT],
        partition: EvaluationPartition,
    ) -> list[float]:
        values: list[float | None] = [
            metric.read_from(fold.get_metrics(partition)) for fold in self.folds
        ]
        return [value for value in values if value is not None]
