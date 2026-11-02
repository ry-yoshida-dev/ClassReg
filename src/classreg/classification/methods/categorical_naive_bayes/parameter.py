from dataclasses import dataclass
from typing import override

from sklearn.naive_bayes import CategoricalNB

from ....feature_scaling import FeatureScaling
from ....parameter_validator import ParameterValidator
from ...method import ClassificationMethod
from ...parameter import ClassificationParameters
from ...processors import (
    ProbabilisticClassificationProcessor,
)


@dataclass(frozen=True)
class CategoricalNaiveBayesParameters(ClassificationParameters):
    """
    Parameters for scikit-learn's CategoricalNB.

    Features must be category codes: negative values raise ValueError, and
    fractional values are truncated silently. A code above those seen in training
    raises IndexError during prediction unless ``min_categories`` covers it.

    Attributes
    ----------
    alpha : float
        Additive smoothing.
    min_categories : int | None
        Minimum number of categories per feature; inferred from training data when None.
    is_prior_fitted : bool
        Whether class priors are learned from the data rather than uniform.
    """

    alpha: float = 1.0
    min_categories: int | None = None
    is_prior_fitted: bool = True

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_integers(
            min_categories=self.min_categories,
        )
        ParameterValidator.require_not_nan(
            alpha=self.alpha,
        )
        if self.alpha <= 0:
            raise ValueError("alpha must be greater than 0")
        if self.min_categories is not None and self.min_categories <= 0:
            raise ValueError("min_categories must be greater than 0 when set")

    @override
    def build_processor(self) -> ProbabilisticClassificationProcessor:
        """
        Build an unfitted categorical naive Bayes processor.

        Returns
        -------
        ProbabilisticClassificationProcessor
            The processor.
        """
        estimator: CategoricalNB = CategoricalNB(
            alpha=self.alpha,
            min_categories=self.min_categories,
            fit_prior=self.is_prior_fitted,
        )
        return ProbabilisticClassificationProcessor(
            method=ClassificationMethod.CATEGORICAL_NAIVE_BAYES,
            estimator=FeatureScaling.NONE.build_pipeline(estimator),
        )
