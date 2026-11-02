from dataclasses import dataclass

from .metric_summary import MetricSummary
from .model_evaluation import ModelEvaluation


@dataclass(frozen=True)
class RankedModel[MetricsT]:
    """
    Candidate model in a comparison.

    Attributes
    ----------
    name : str
        The name given to the candidate.
    evaluation : ModelEvaluation[MetricsT]
        The evaluation of the candidate.
    score : MetricSummary
        The summary of the ranking metric on the test partitions.
    """

    name: str
    evaluation: ModelEvaluation[MetricsT]
    score: MetricSummary
