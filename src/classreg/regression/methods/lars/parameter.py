from dataclasses import dataclass
from typing import override

from sklearn.linear_model import Lars

from ....feature_scaling import FeatureScaling
from ....parameter_validator import ParameterValidator
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor


@dataclass(frozen=True)
class LarsParameters(RegressionParameters):
    """
    Parameters for scikit-learn's Lars.

    Attributes
    ----------
    n_nonzero_coefs : int
        Target number of non-zero coefficients.
    is_intercept_fitted : bool
        Whether to fit an intercept.
    scaling : FeatureScaling
        Feature scaling; the selection path depends on feature scale.
    """

    n_nonzero_coefs: int = 500
    is_intercept_fitted: bool = True
    scaling: FeatureScaling = FeatureScaling.STANDARD

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_integers(
            n_nonzero_coefs=self.n_nonzero_coefs,
        )
        if self.n_nonzero_coefs <= 0:
            raise ValueError("n_nonzero_coefs must be greater than 0")

    @override
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted least angle regression processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator: Lars = Lars(
            n_nonzero_coefs=self.n_nonzero_coefs,
            fit_intercept=self.is_intercept_fitted,
        )
        return RegressionProcessor(
            method=RegressionMethod.LARS,
            estimator=self.scaling.build_pipeline(estimator),
        )
