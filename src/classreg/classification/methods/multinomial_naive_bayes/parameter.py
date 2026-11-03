from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.naive_bayes import MultinomialNB

from ....parameter_validator import ParameterValidator
from ...method import ClassificationMethod
from ...parameter import ClassificationParameters
from ...processors import (
    ProbabilisticClassificationProcessor,
)


@dataclass(frozen=True)
class MultinomialNaiveBayesParameters(ClassificationParameters):
    """
    Parameters for scikit-learn's MultinomialNB.

    Features must be non-negative, such as counts; fitting raises ValueError
    otherwise. Scaling that produces negative features is therefore rejected.

    Attributes
    ----------
    alpha : float
        Additive smoothing.
    is_prior_fitted : bool
        Whether class priors are learned from the data rather than uniform.
    scaling : FeatureScaling
        Feature scaling; only NONE and MIN_MAX keep features non-negative.
    """

    alpha: float = 1.0
    is_prior_fitted: bool = True
    scaling: FeatureScaling = FeatureScaling.NONE

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_not_nan(
            alpha=self.alpha,
        )
        if self.alpha <= 0:
            raise ValueError("alpha must be greater than 0")
        if self.scaling in (FeatureScaling.STANDARD, FeatureScaling.ROBUST):
            raise ValueError("scaling must not produce negative features")

    @override
    def build_processor(self) -> ProbabilisticClassificationProcessor:
        """
        Build an unfitted multinomial naive Bayes processor.

        Returns
        -------
        ProbabilisticClassificationProcessor
            The processor.
        """
        estimator: MultinomialNB = MultinomialNB(
            alpha=self.alpha,
            fit_prior=self.is_prior_fitted,
        )
        return ProbabilisticClassificationProcessor(
            method=ClassificationMethod.MULTINOMIAL_NAIVE_BAYES,
            estimator=self.scaling.build_pipeline(estimator),
        )
