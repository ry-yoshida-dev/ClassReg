from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.naive_bayes import BernoulliNB

from ....parameter_validator import ParameterValidator
from ...method import ClassificationMethod
from ...parameter import ClassificationParameters
from ...processors import (
    ProbabilisticClassificationProcessor,
)


@dataclass(frozen=True)
class BernoulliNaiveBayesParameters(ClassificationParameters):
    """
    Parameters for scikit-learn's BernoulliNB.

    The default pipeline standardizes features, so the default threshold splits
    each feature at its training mean. Without scaling, a threshold of 0 turns all
    positive features into the same value.

    Attributes
    ----------
    alpha : float
        Additive smoothing.
    binarize_threshold : float | None
        Threshold that binarizes features; features are taken as binary when None.
    is_prior_fitted : bool
        Whether class priors are learned from the data rather than uniform.
    scaling : FeatureScaling
        Feature scaling, applied before binarization.
    """

    alpha: float = 1.0
    binarize_threshold: float | None = 0.0
    is_prior_fitted: bool = True
    scaling: FeatureScaling = FeatureScaling.STANDARD

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_not_nan(
            alpha=self.alpha,
        )
        if self.alpha <= 0:
            raise ValueError("alpha must be greater than 0")
        if self.binarize_threshold is not None:
            ParameterValidator.require_not_nan(
                binarize_threshold=self.binarize_threshold
            )

    @override
    def build_processor(self) -> ProbabilisticClassificationProcessor:
        """
        Build an unfitted Bernoulli naive Bayes processor.

        Returns
        -------
        ProbabilisticClassificationProcessor
            The processor.
        """
        estimator: BernoulliNB = BernoulliNB(
            alpha=self.alpha,
            binarize=self.binarize_threshold,
            fit_prior=self.is_prior_fitted,
        )
        return ProbabilisticClassificationProcessor(
            method=ClassificationMethod.BERNOULLI_NAIVE_BAYES,
            estimator=self.scaling.build_pipeline(estimator),
        )
