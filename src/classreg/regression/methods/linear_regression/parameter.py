from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.linear_model import LinearRegression

from ....parameter_validator import ParameterValidator
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor


@dataclass(frozen=True)
class LinearRegressionParameters(RegressionParameters):
    """
    Parameters for scikit-learn's LinearRegression.

    Attributes
    ----------
    is_intercept_fitted : bool
        Whether to fit an intercept.
    is_coefficient_positive : bool
        Whether coefficients are constrained to be non-negative.
    n_jobs : int | None
        Number of parallel jobs.
    scaling : FeatureScaling
        Feature scaling; predictions of unregularized least squares do not
        depend on it, so it is off by default.
    """

    is_intercept_fitted: bool = True
    is_coefficient_positive: bool = False
    n_jobs: int | None = None
    scaling: FeatureScaling = FeatureScaling.NONE

    def __post_init__(self) -> None:
        ParameterValidator.require_integers(n_jobs=self.n_jobs)

    @override
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted linear regression processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator: LinearRegression = LinearRegression(
            fit_intercept=self.is_intercept_fitted,
            positive=self.is_coefficient_positive,
            n_jobs=self.n_jobs,
        )
        return RegressionProcessor(
            method=RegressionMethod.LINEAR_REGRESSION,
            estimator=self.scaling.build_pipeline(estimator),
        )
