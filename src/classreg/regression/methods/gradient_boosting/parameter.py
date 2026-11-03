from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.ensemble import GradientBoostingRegressor

from ....options import RandomForestMaxFeatures
from ....parameter_validator import ParameterValidator
from ....random_state import DEFAULT_RANDOM_STATE
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor
from .loss import GradientBoostingLoss


@dataclass(frozen=True)
class GradientBoostingParameters(RegressionParameters):
    """
    Parameters for scikit-learn's GradientBoostingRegressor.

    Attributes
    ----------
    loss : GradientBoostingLoss
        The loss function to optimize.
    learning_rate : float
        Shrinkage applied to each tree.
    n_estimators : int
        Number of boosting stages.
    subsample : float
        Fraction of the samples used to fit each tree.
    min_samples_split : int
        Minimum number of samples required to split a node.
    min_samples_leaf : int
        Minimum number of samples required at a leaf.
    max_depth : int | None
        Maximum tree depth; no limit when None.
    max_features : RandomForestMaxFeatures | None
        Features considered per split; all features when None.
    alpha : float
        Quantile of the huber and quantile losses; ignored by the other losses.
    is_early_stopping_enabled : bool
        Whether to stop when an internal validation split stops improving.
    validation_fraction : float
        Fraction of the training data held out for early stopping.
    n_iter_no_change : int
        Iterations without improvement before early stopping.
    tol : float
        Minimum improvement counted by early stopping.
    random_state : int | None
        Seed for subsampling, feature sampling, and the early stopping split.
    """

    loss: GradientBoostingLoss = GradientBoostingLoss.SQUARED_ERROR
    learning_rate: float = 0.1
    n_estimators: int = 100
    subsample: float = 1.0
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_depth: int | None = 3
    max_features: RandomForestMaxFeatures | None = None
    alpha: float = 0.9
    is_early_stopping_enabled: bool = False
    validation_fraction: float = 0.1
    n_iter_no_change: int = 10
    tol: float = 1e-4
    random_state: int | None = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_integers(
            n_estimators=self.n_estimators,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            n_iter_no_change=self.n_iter_no_change,
            random_state=self.random_state,
        )
        ParameterValidator.require_not_nan(
            learning_rate=self.learning_rate,
            tol=self.tol,
        )
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0")
        if not 0 < self.subsample <= 1:
            raise ValueError("subsample must be in the range (0, 1]")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be greater than 0")
        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError("max_depth must be greater than 0 when set")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in the range (0, 1)")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in the range (0, 1)")
        if self.n_iter_no_change <= 0:
            raise ValueError("n_iter_no_change must be greater than 0")
        if self.tol < 0:
            raise ValueError("tol must be non-negative")

    @override
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted gradient boosting processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator: GradientBoostingRegressor = GradientBoostingRegressor(
            loss=self.loss.value,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            max_features=None if self.max_features is None else self.max_features.value,
            alpha=self.alpha,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change
            if self.is_early_stopping_enabled
            else None,
            tol=self.tol,
            random_state=self.random_state,
        )
        return RegressionProcessor(
            method=RegressionMethod.GRADIENT_BOOSTING,
            estimator=FeatureScaling.NONE.build_pipeline(estimator),
        )
