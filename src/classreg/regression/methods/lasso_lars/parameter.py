from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.linear_model import LassoLars

from ....parameter_validator import ParameterValidator
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor


@dataclass(frozen=True)
class LassoLarsParameters(RegressionParameters):
    """
    Parameters for scikit-learn's LassoLars.

    Attributes
    ----------
    alpha : float
        L1 regularization strength.
    max_iter : int
        Maximum number of iterations.
    is_positive : bool
        Whether coefficients are forced to be non-negative.
    is_intercept_fitted : bool
        Whether to fit an intercept.
    scaling : FeatureScaling
        Feature scaling; the penalty depends on feature scale.
    """

    alpha: float = 1.0
    max_iter: int = 500
    is_positive: bool = False
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
        )
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")

    @override
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted lasso least angle regression processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator: LassoLars = LassoLars(
            alpha=self.alpha,
            max_iter=self.max_iter,
            positive=self.is_positive,
            fit_intercept=self.is_intercept_fitted,
        )
        return RegressionProcessor(
            method=RegressionMethod.LASSO_LARS,
            estimator=self.scaling.build_pipeline(estimator),
        )
