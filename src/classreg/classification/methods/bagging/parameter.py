from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.ensemble import BaggingClassifier

from ....parameter_validator import ParameterValidator
from ....random_state import DEFAULT_RANDOM_STATE
from ...method import ClassificationMethod
from ...parameter import ClassificationParameters
from ...processors import ProbabilisticClassificationProcessor
from ..decision_tree import DecisionTreeParameters


@dataclass(frozen=True)
class BaggingParameters(ClassificationParameters):
    """
    Parameters for scikit-learn's BaggingClassifier.

    Bagged methods without class probabilities contribute the fractions of their
    votes, so scores are always probabilities.

    Attributes
    ----------
    estimator : ClassificationParameters
        The method fitted on each random subset, with its own feature scaling.
    n_estimators : int
        Number of fitted copies.
    max_samples : float
        Fraction of the samples drawn for each copy.
    max_features : float
        Fraction of the features drawn for each copy.
    is_bootstrapped : bool
        Whether samples are drawn with replacement.
    is_feature_bootstrapped : bool
        Whether features are drawn with replacement.
    n_jobs : int | None
        Number of parallel jobs.
    random_state : int | None
        Seed for drawing subsets and for the copies.
    """

    estimator: ClassificationParameters = DecisionTreeParameters()
    n_estimators: int = 10
    max_samples: float = 1.0
    max_features: float = 1.0
    is_bootstrapped: bool = True
    is_feature_bootstrapped: bool = False
    n_jobs: int | None = None
    random_state: int | None = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_integers(
            n_estimators=self.n_estimators,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0")
        if not 0 < self.max_samples <= 1:
            raise ValueError("max_samples must be in the range (0, 1]")
        if not 0 < self.max_features <= 1:
            raise ValueError("max_features must be in the range (0, 1]")

    @override
    def build_processor(self) -> ProbabilisticClassificationProcessor:
        """
        Build an unfitted bagging processor.

        Returns
        -------
        ProbabilisticClassificationProcessor
            The processor.
        """
        estimator: BaggingClassifier = BaggingClassifier(
            estimator=self.estimator.build_processor().underlying_estimator,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=self.is_bootstrapped,
            bootstrap_features=self.is_feature_bootstrapped,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        return ProbabilisticClassificationProcessor(
            method=ClassificationMethod.BAGGING,
            estimator=FeatureScaling.NONE.build_pipeline(estimator),
        )
