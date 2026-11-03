from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.linear_model import SGDRegressor

from ....parameter_validator import ParameterValidator
from ....random_state import DEFAULT_RANDOM_STATE
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor
from .loss import StochasticGradientDescentLoss


@dataclass(frozen=True)
class StochasticGradientDescentParameters(RegressionParameters):
    """
    Parameters for scikit-learn's SGDRegressor.

    Targets are not scaled, so ``epsilon`` must suit the target's magnitude: the
    default is far too small for targets in the hundreds under the Huber loss.

    Attributes
    ----------
    loss : StochasticGradientDescentLoss
        The loss function to optimize.
    alpha : float
        Regularization strength.
    l1_ratio : float
        Elastic-net mixing: 0 is pure L2 and 1 is pure L1.
    epsilon : float
        Huber threshold or half-width of the epsilon tube, in target units.
        Ignored by the squared error loss.
    max_iter : int
        Maximum number of passes over the training data.
    tol : float
        Stopping tolerance.
    is_intercept_fitted : bool
        Whether to fit an intercept.
    is_shuffled : bool
        Whether samples are shuffled after each pass.
    scaling : FeatureScaling
        Feature scaling; regularization and step sizes depend on feature scale.
    random_state : int | None
        Seed for shuffling.
    """

    loss: StochasticGradientDescentLoss = StochasticGradientDescentLoss.SQUARED_ERROR
    alpha: float = 1e-4
    l1_ratio: float = 0.0
    epsilon: float = 0.1
    max_iter: int = 1000
    tol: float = 1e-3
    is_intercept_fitted: bool = True
    is_shuffled: bool = True
    scaling: FeatureScaling = FeatureScaling.STANDARD
    random_state: int | None = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_integers(
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        ParameterValidator.require_not_nan(
            alpha=self.alpha,
            epsilon=self.epsilon,
            tol=self.tol,
        )
        if self.alpha <= 0:
            raise ValueError("alpha must be greater than 0")
        if not 0 <= self.l1_ratio <= 1:
            raise ValueError("l1_ratio must be in the range [0, 1]")
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")
        if self.tol < 0:
            raise ValueError("tol must be non-negative")

    @override
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted stochastic gradient descent processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator: SGDRegressor = SGDRegressor(
            loss=self.loss.value,
            penalty="elasticnet",
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.is_intercept_fitted,
            max_iter=self.max_iter,
            tol=self.tol,
            shuffle=self.is_shuffled,
            epsilon=self.epsilon,
            random_state=self.random_state,
        )
        return RegressionProcessor(
            method=RegressionMethod.STOCHASTIC_GRADIENT_DESCENT,
            estimator=self.scaling.build_pipeline(estimator),
        )
