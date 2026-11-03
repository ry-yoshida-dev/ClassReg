from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from ....parameter_validator import ParameterValidator
from ....random_state import DEFAULT_RANDOM_STATE
from ...method import ClassificationMethod
from ...parameter import ClassificationParameters
from ...processors import ProbabilisticClassificationProcessor


@dataclass(frozen=True)
class AdaBoostParameters(ClassificationParameters):
    """
    Parameters for scikit-learn's AdaBoostClassifier over decision trees.

    Attributes
    ----------
    n_estimators : int
        Maximum number of boosting stages.
    learning_rate : float
        Weight applied to each tree.
    max_depth : int | None
        Maximum depth of each tree; no limit when None.
    random_state : int | None
        Seed for the trees.
    """

    n_estimators: int = 50
    learning_rate: float = 1.0
    max_depth: int | None = 1
    random_state: int | None = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_integers(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        ParameterValidator.require_not_nan(
            learning_rate=self.learning_rate,
        )
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")
        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError("max_depth must be greater than 0 when set")

    @override
    def build_processor(self) -> ProbabilisticClassificationProcessor:
        """
        Build an unfitted AdaBoost processor.

        Returns
        -------
        ProbabilisticClassificationProcessor
            The processor.
        """
        estimator: AdaBoostClassifier = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=self.max_depth),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
        return ProbabilisticClassificationProcessor(
            method=ClassificationMethod.ADA_BOOST,
            estimator=FeatureScaling.NONE.build_pipeline(estimator),
        )
