from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import reduce
from typing import override

from ..evaluation import (
    DatasetSplitter,
    EvaluableParametersLike,
    EvaluableProcessorLike,
    MetricLike,
    ModelComparison,
    ModelEvaluation,
    SupervisedEvaluator,
)
from .dataset import ClassificationDataset
from .methods import BaselineParameters
from .metrics import (
    ClassificationMetric,
    ClassScoreKind,
    ConfusionMatrix,
    MetricValues,
)
from .processor import ClassificationProcessor
from .splitters import StratifiedKFoldSplitter
from .types import LabelArray, LabelScalar, NumericArray


@dataclass(frozen=True)
class ClassificationEvaluator(SupervisedEvaluator[LabelScalar, MetricValues]):
    """
    Evaluator with stratified folds and a class prior baseline by default.

    Evaluation raises ValueError if a split trains on fewer than two classes or
    tests on a class missing from its training samples.

    Attributes
    ----------
    splitter : DatasetSplitter
        The strategy that partitions the dataset.
    baseline : EvaluableParametersLike[LabelScalar, MetricValues] | None
        The reference model evaluated on the same splits; None to skip it.
    """

    splitter: DatasetSplitter = field(default_factory=StratifiedKFoldSplitter)
    baseline: EvaluableParametersLike[LabelScalar, MetricValues] | None = field(
        default_factory=BaselineParameters
    )

    @staticmethod
    def combine_confusion_matrices(
        evaluation: ModelEvaluation[MetricValues],
    ) -> ConfusionMatrix:
        """
        Sum the confusion matrices of all test partitions of an evaluation.

        Parameters
        ----------
        evaluation : ModelEvaluation[MetricValues]
            The cross-validated evaluation.

        Returns
        -------
        ConfusionMatrix
            The combined confusion matrix.
        """
        return reduce(
            ConfusionMatrix.combine,
            [fold.test_metrics.confusion_matrix for fold in evaluation.folds],
        )

    @override
    def compare(
        self,
        candidates: Mapping[str, EvaluableParametersLike[LabelScalar, MetricValues]],
        X: NumericArray,
        y: LabelArray,
        metric: MetricLike[MetricValues] = ClassificationMetric.MACRO_F1,
    ) -> ModelComparison[MetricValues]:
        """
        Rank model configurations evaluated on identical splits.

        Parameters
        ----------
        candidates : Mapping[str, EvaluableParametersLike[LabelScalar, MetricValues]]
            The model configurations by name.
        X : NumericArray
            The feature matrix.
        y : LabelArray
            The class labels.
        metric : MetricLike[MetricValues]
            The metric to rank by, using its mean over the test partitions.

        Returns
        -------
        ModelComparison[MetricValues]
            The candidates, best first.

        Raises
        ------
        ValueError
            If a candidate's kind of scores cannot define the metric, which is
            detected before any training, or for the reasons of the base method.
        """
        self._validate_candidates(candidates, metric)
        return super().compare(candidates, X, y, metric)

    @staticmethod
    def _validate_candidates(
        candidates: Mapping[str, EvaluableParametersLike[LabelScalar, MetricValues]],
        metric: MetricLike[MetricValues],
    ) -> None:
        if not isinstance(metric, ClassificationMetric):
            return
        for name, parameters in candidates.items():
            processor: EvaluableProcessorLike[LabelScalar, MetricValues] = (
                parameters.build_processor()
            )
            if not isinstance(processor, ClassificationProcessor):
                continue
            kind: ClassScoreKind = processor.score_kind
            if not metric.is_defined_for(kind):
                raise ValueError(
                    f"{metric.value} is undefined for candidate {name!r}, which produces {kind.value} scores"
                )

    @override
    def _build_dataset(self, X: NumericArray, y: LabelArray) -> ClassificationDataset:
        return ClassificationDataset(X=X, y=y)

    @override
    def _list_metrics(self) -> tuple[MetricLike[MetricValues], ...]:
        return tuple(ClassificationMetric)
