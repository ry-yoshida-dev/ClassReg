from dataclasses import dataclass
from typing import override

from sklearn.linear_model import ARDRegression

from ....feature_scaling import FeatureScaling
from ....parameter_validator import ParameterValidator
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor


@dataclass(frozen=True)
class AutomaticRelevanceDeterminationParameters(RegressionParameters):
    """
    Parameters for scikit-learn's ARDRegression.

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
    threshold_lambda : float
        Weight precision above which a feature is pruned.
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
    threshold_lambda: float = 1e4
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
            threshold_lambda=self.threshold_lambda,
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
        if self.threshold_lambda <= 0:
            raise ValueError("threshold_lambda must be greater than 0")

    @override
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted automatic relevance determination processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator: ARDRegression = ARDRegression(
            max_iter=self.max_iter,
            tol=self.tol,
            alpha_1=self.alpha_1,
            alpha_2=self.alpha_2,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            threshold_lambda=self.threshold_lambda,
            fit_intercept=self.is_intercept_fitted,
        )
        return RegressionProcessor(
            method=RegressionMethod.AUTOMATIC_RELEVANCE_DETERMINATION,
            estimator=self.scaling.build_pipeline(estimator),
        )
