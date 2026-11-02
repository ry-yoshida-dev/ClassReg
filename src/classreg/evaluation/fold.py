from dataclasses import dataclass

from .partition import EvaluationPartition


@dataclass(frozen=True)
class FoldEvaluation[MetricsT]:
    """
    Performance of a model trained and tested on one split.

    Attributes
    ----------
    fold_index : int
        Position of the split.
    train_metrics : MetricsT
        Performance on the training samples.
    test_metrics : MetricsT
        Performance on the held-out samples.
    """

    fold_index: int
    train_metrics: MetricsT
    test_metrics: MetricsT

    def get_metrics(self, partition: EvaluationPartition) -> MetricsT:
        """
        Get the metrics of a partition.

        Parameters
        ----------
        partition : EvaluationPartition
            The partition.

        Returns
        -------
        MetricsT
            The metrics measured on that partition.
        """
        match partition:
            case EvaluationPartition.TRAIN:
                return self.train_metrics
            case EvaluationPartition.TEST:
                return self.test_metrics
