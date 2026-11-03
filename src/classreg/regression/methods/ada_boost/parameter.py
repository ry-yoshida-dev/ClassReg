from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

from ....parameter_validator import ParameterValidator
from ....random_state import DEFAULT_RANDOM_STATE
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor
from .loss import AdaBoostLoss


@dataclass(frozen=True)
class AdaBoostParameters(RegressionParameters):
    """
    Parameters for scikit-learn's AdaBoostRegressor over decision trees.

    Attributes
    ----------
    n_estimators : int
        Maximum number of boosting stages.
    learning_rate : float
        Weight applied to each tree.
    loss : AdaBoostLoss
        The loss used to update sample weights.
    max_depth : int | None
        Maximum depth of each tree; no limit when None.
    random_state : int | None
        Seed for the trees and the weighted resampling.
    """

    n_estimators: int = 50
    learning_rate: float = 1.0
    loss: AdaBoostLoss = AdaBoostLoss.LINEAR
    max_depth: int | None = 3
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
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted AdaBoost processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator: AdaBoostRegressor = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=self.max_depth),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            loss=self.loss.value,
            random_state=self.random_state,
        )
        return RegressionProcessor(
            method=RegressionMethod.ADA_BOOST,
            estimator=FeatureScaling.NONE.build_pipeline(estimator),
        )
