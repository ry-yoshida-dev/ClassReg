from dataclasses import dataclass
from typing import override

from sklearn.linear_model import QuantileRegressor

from ....feature_scaling import FeatureScaling
from ....parameter_validator import ParameterValidator
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor


@dataclass(frozen=True)
class QuantileParameters(RegressionParameters):
    """
    Parameters for scikit-learn's QuantileRegressor.

    Attributes
    ----------
    quantile : float
        The conditional quantile to predict, in the range (0, 1).
    alpha : float
        L1 regularization strength; the default, taken from scikit-learn, can
        shrink all coefficients to zero on standardized features.
    is_intercept_fitted : bool
        Whether to fit an intercept.
    scaling : FeatureScaling
        Feature scaling; the penalty depends on feature scale.
    """

    quantile: float = 0.5
    alpha: float = 1.0
    is_intercept_fitted: bool = True
    scaling: FeatureScaling = FeatureScaling.STANDARD

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_not_nan(
            alpha=self.alpha,
        )
        if not 0 < self.quantile < 1:
            raise ValueError("quantile must be in the range (0, 1)")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")

    @override
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted quantile regression processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator: QuantileRegressor = QuantileRegressor(
            quantile=self.quantile,
            alpha=self.alpha,
            fit_intercept=self.is_intercept_fitted,
        )
        return RegressionProcessor(
            method=RegressionMethod.QUANTILE,
            estimator=self.scaling.build_pipeline(estimator),
        )
