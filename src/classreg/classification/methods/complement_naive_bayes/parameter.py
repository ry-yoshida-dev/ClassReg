from dataclasses import dataclass
from typing import override

from sklearn.naive_bayes import ComplementNB

from ....feature_scaling import FeatureScaling
from ....parameter_validator import ParameterValidator
from ...method import ClassificationMethod
from ...parameter import ClassificationParameters
from ...processors import (
    ProbabilisticClassificationProcessor,
)


@dataclass(frozen=True)
class ComplementNaiveBayesParameters(ClassificationParameters):
    """
    Parameters for scikit-learn's ComplementNB.

    Features must be non-negative, such as counts; fitting raises ValueError
    otherwise. Scaling that produces negative features is therefore rejected.

    Attributes
    ----------
    alpha : float
        Additive smoothing.
    is_normalized : bool
        Whether feature weights receive a second normalization.
    is_prior_fitted : bool
        Whether class priors are learned from the data rather than uniform.
    scaling : FeatureScaling
        Feature scaling; only NONE and MIN_MAX keep features non-negative.
    """

    alpha: float = 1.0
    is_normalized: bool = False
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
        Build an unfitted complement naive Bayes processor.

        Returns
        -------
        ProbabilisticClassificationProcessor
            The processor.
        """
        estimator: ComplementNB = ComplementNB(
            alpha=self.alpha,
            norm=self.is_normalized,
            fit_prior=self.is_prior_fitted,
        )
        return ProbabilisticClassificationProcessor(
            method=ClassificationMethod.COMPLEMENT_NAIVE_BAYES,
            estimator=self.scaling.build_pipeline(estimator),
        )
