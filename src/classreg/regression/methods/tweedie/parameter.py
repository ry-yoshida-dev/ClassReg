from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.linear_model import TweedieRegressor

from ....parameter_validator import ParameterValidator
from ...generalized_linear_solver import GeneralizedLinearSolver
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor
from .link import TweedieLink


@dataclass(frozen=True)
class TweedieParameters(RegressionParameters):
    """
    Parameters for scikit-learn's TweedieRegressor.

    The power selects the distribution and the valid targets: 0 is normal with
    any target, 1 is Poisson and values between 1 and 2 are compound Poisson-Gamma
    with non-negative targets, and 2 and above need strictly positive targets.
    Fitting raises ValueError for invalid targets.

    Attributes
    ----------
    power : float
        The Tweedie power; values in the open range (0, 1) do not exist.
    link : TweedieLink
        The link function.
    alpha : float
        L2 regularization strength.
    solver : GeneralizedLinearSolver
        The optimization algorithm.
    max_iter : int
        Maximum number of solver iterations.
    tol : float
        Stopping tolerance.
    is_intercept_fitted : bool
        Whether to fit an intercept.
    scaling : FeatureScaling
        Feature scaling; the penalty depends on feature scale.
    """

    power: float = 0.0
    link: TweedieLink = TweedieLink.AUTO
    alpha: float = 1.0
    solver: GeneralizedLinearSolver = GeneralizedLinearSolver.LBFGS
    max_iter: int = 100
    tol: float = 1e-4
    is_intercept_fitted: bool = True
    scaling: FeatureScaling = FeatureScaling.STANDARD

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_integers(
            max_iter=self.max_iter,
        )
        ParameterValidator.require_not_nan(
            power=self.power,
            alpha=self.alpha,
            tol=self.tol,
        )
        if 0 < self.power < 1:
            raise ValueError("power must not be in the open range (0, 1)")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")
        if self.tol <= 0:
            raise ValueError("tol must be greater than 0")

    @override
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted Tweedie regression processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator: TweedieRegressor = TweedieRegressor(
            power=self.power,
            link=self.link.value,
            alpha=self.alpha,
            solver=self.solver.value,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.is_intercept_fitted,
        )
        return RegressionProcessor(
            method=RegressionMethod.TWEEDIE,
            estimator=self.scaling.build_pipeline(estimator),
        )
