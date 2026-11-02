from dataclasses import dataclass
from typing import override

from sklearn.linear_model import HuberRegressor

from ....feature_scaling import FeatureScaling
from ....parameter_validator import ParameterValidator
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor


@dataclass(frozen=True)
class HuberParameters(RegressionParameters):
    """
    Parameters for scikit-learn's HuberRegressor.

    Attributes
    ----------
    epsilon : float
        Residual size, at least 1, beyond which samples count as outliers.
    alpha : float
        L2 regularization strength.
    max_iter : int
        Maximum number of solver iterations.
    tol : float
        Stopping tolerance.
    is_intercept_fitted : bool
        Whether to fit an intercept.
    scaling : FeatureScaling
        Feature scaling; the penalty depends on feature scale.
    """

    epsilon: float = 1.35
    alpha: float = 1e-4
    max_iter: int = 100
    tol: float = 1e-5
    is_intercept_fitted: bool = True
    scaling: FeatureScaling = FeatureScaling.STANDARD

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_integers(
            max_iter=self.max_iter,
        )
        ParameterValidator.require_not_nan(
            epsilon=self.epsilon,
            alpha=self.alpha,
            tol=self.tol,
        )
        if self.epsilon < 1:
            raise ValueError("epsilon must be at least 1")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")
        if self.tol < 0:
            raise ValueError("tol must be non-negative")

    @override
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted Huber regression processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator: HuberRegressor = HuberRegressor(
            epsilon=self.epsilon,
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.is_intercept_fitted,
        )
        return RegressionProcessor(
            method=RegressionMethod.HUBER,
            estimator=self.scaling.build_pipeline(estimator),
        )
