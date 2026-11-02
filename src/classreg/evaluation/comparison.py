from __future__ import annotations

from dataclasses import dataclass

from .model_evaluation import ModelEvaluation
from .protocols import MetricLike
from .ranked_model import RankedModel
from .text_table import TextTable


@dataclass(frozen=True)
class ModelComparison[MetricsT]:
    """
    Candidate models ranked by one metric, all evaluated on identical splits.

    Attributes
    ----------
    metric : MetricLike[MetricsT]
        The ranking metric.
    ranking : tuple[RankedModel[MetricsT], ...]
        The candidates, best first.
    baseline : ModelEvaluation[MetricsT] | None
        The reference model; None when the evaluator has no baseline.
    """

    metric: MetricLike[MetricsT]
    ranking: tuple[RankedModel[MetricsT], ...]
    baseline: ModelEvaluation[MetricsT] | None

    @property
    def best(self) -> RankedModel[MetricsT]:
        """
        Get the candidate with the best mean score.

        Returns
        -------
        RankedModel[MetricsT]
            The best candidate.
        """
        return self.ranking[0]

    def format_table(self) -> str:
        """
        Format the ranking as a plain text table.

        Returns
        -------
        str
            The table, best candidate first, followed by the baseline if the
            ranking metric is defined on the same folds for it.
        """
        rows: list[tuple[str, str]] = [
            (entry.name, entry.score.format()) for entry in self.ranking
        ]
        if self.baseline is not None and self.baseline.get_defined_fold_indices(
            self.metric
        ) == self.best.evaluation.get_defined_fold_indices(self.metric):
            rows.append(("(baseline)", self.baseline.summarize(self.metric).format()))
        return TextTable(header=("model", self.metric.value), rows=tuple(rows)).render()
