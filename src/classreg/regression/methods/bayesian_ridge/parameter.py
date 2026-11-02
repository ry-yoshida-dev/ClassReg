from dataclasses import dataclass
from typing import override

from sklearn.linear_model import BayesianRidge

from ....feature_scaling import FeatureScaling
from ....parameter_validator import ParameterValidator
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor


@dataclass(frozen=True)
class BayesianRidgeParameters(RegressionParameters):
    """
    Parameters for scikit-learn's BayesianRidge.

    Attributes
    ----------
    max_iter : int
        Maximum number of iterations.
    tol : float
        Stopping tolerance.
    alpha_1 : float
        Shape parameter of the Gamma prior over the noise precision.
    alpha_2 : float
        Rate parameter of the Gamma prior over the noise precision.
    lambda_1 : float
        Shape parameter of the Gamma prior over the weight precision.
    lambda_2 : float
        Rate parameter of the Gamma prior over the weight precision.
    is_intercept_fitted : bool
        Whether to fit an intercept.
    scaling : FeatureScaling
        Feature scaling; the penalty depends on feature scale.
    """

    max_iter: int = 300
    tol: float = 1e-3
    alpha_1: float = 1e-6
    alpha_2: float = 1e-6
    lambda_1: float = 1e-6
    lambda_2: float = 1e-6
    is_intercept_fitted: bool = True
    scaling: FeatureScaling = FeatureScaling.STANDARD

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_integers(
            max_iter=self.max_iter,
        )
        ParameterValidator.require_not_nan(
            tol=self.tol,
            alpha_1=self.alpha_1,
            alpha_2=self.alpha_2,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
        )
        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")
        if self.tol <= 0:
            raise ValueError("tol must be greater than 0")
        if self.alpha_1 < 0:
            raise ValueError("alpha_1 must be non-negative")
        if self.alpha_2 < 0:
            raise ValueError("alpha_2 must be non-negative")
        if self.lambda_1 < 0:
            raise ValueError("lambda_1 must be non-negative")
        if self.lambda_2 < 0:
            raise ValueError("lambda_2 must be non-negative")

    @override
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted Bayesian ridge processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator: BayesianRidge = BayesianRidge(
            max_iter=self.max_iter,
            tol=self.tol,
            alpha_1=self.alpha_1,
            alpha_2=self.alpha_2,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            fit_intercept=self.is_intercept_fitted,
        )
        return RegressionProcessor(
            method=RegressionMethod.BAYESIAN_RIDGE,
            estimator=self.scaling.build_pipeline(estimator),
        )
