from dataclasses import dataclass
from typing import override

from sklearn.linear_model import GammaRegressor

from ....feature_scaling import FeatureScaling
from ....parameter_validator import ParameterValidator
from ...generalized_linear_solver import GeneralizedLinearSolver
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor


@dataclass(frozen=True)
class GammaParameters(RegressionParameters):
    """
    Parameters for scikit-learn's GammaRegressor.

    Targets must be strictly positive; fitting raises ValueError otherwise.

    Attributes
    ----------
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
            alpha=self.alpha,
            tol=self.tol,
        )
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")
        if self.tol <= 0:
            raise ValueError("tol must be greater than 0")

    @override
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted Gamma regression processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator: GammaRegressor = GammaRegressor(
            alpha=self.alpha,
            solver=self.solver.value,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.is_intercept_fitted,
        )
        return RegressionProcessor(
            method=RegressionMethod.GAMMA,
            estimator=self.scaling.build_pipeline(estimator),
        )
