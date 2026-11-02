from __future__ import annotations

from dataclasses import dataclass

from .model_evaluation import ModelEvaluation
from .partition import EvaluationPartition
from .protocols import MetricLike
from .text_table import TextTable


@dataclass(frozen=True)
class EvaluationReport[MetricsT]:
    """
    Evaluation of a model, next to a baseline measured on the same splits.

    Attributes
    ----------
    model : ModelEvaluation[MetricsT]
        The evaluated model.
    baseline : ModelEvaluation[MetricsT] | None
        The reference model; None when the evaluator has no baseline.
    metrics : tuple[MetricLike[MetricsT], ...]
        The metrics listed by format_table.
    """

    model: ModelEvaluation[MetricsT]
    baseline: ModelEvaluation[MetricsT] | None
    metrics: tuple[MetricLike[MetricsT], ...]

    def get_baseline_improvement(self, metric: MetricLike[MetricsT]) -> float:
        """
        Get how much the model's mean test value improves on the baseline.

        Parameters
        ----------
        metric : MetricLike[MetricsT]
            The metric.

        Returns
        -------
        float
            The signed difference, positive when the model is better,
            regardless of the direction of the metric.

        Raises
        ------
        ValueError
            If there is no baseline, the metric is undefined for a model, or
            the metric is defined on different folds for the two models.
        """
        if self.baseline is None:
            raise ValueError("this report has no baseline")
        if self.model.get_defined_fold_indices(
            metric
        ) != self.baseline.get_defined_fold_indices(metric):
            raise ValueError(
                f"{metric.value} is defined on different folds for the model and the baseline"
            )
        difference: float = (
            self.model.summarize(metric).mean - self.baseline.summarize(metric).mean
        )
        return difference if metric.is_higher_better else -difference

    def format_table(self) -> str:
        """
        Format all metrics as a plain text table.

        Returns
        -------
        str
            The table, with train, test, and baseline test columns; undefined
            values are shown as a dash.
        """
        rows: tuple[tuple[str, str, str, str], ...] = tuple(
            (
                metric.value,
                self._format_cell(self.model, metric, EvaluationPartition.TRAIN),
                self._format_cell(self.model, metric, EvaluationPartition.TEST),
                self._format_cell(self.baseline, metric, EvaluationPartition.TEST),
            )
            for metric in self.metrics
        )
        return TextTable(
            header=("metric", "train", "test", "baseline test"),
            rows=rows,
        ).render()

    @staticmethod
    def _format_cell(
        evaluation: ModelEvaluation[MetricsT] | None,
        metric: MetricLike[MetricsT],
        partition: EvaluationPartition,
    ) -> str:
        if evaluation is None or not evaluation.is_defined(metric, partition):
            return "-"
        return evaluation.summarize(metric, partition).format()
