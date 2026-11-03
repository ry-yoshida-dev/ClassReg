from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.linear_model import OrthogonalMatchingPursuit

from ....parameter_validator import ParameterValidator
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor


@dataclass(frozen=True)
class OrthogonalMatchingPursuitParameters(RegressionParameters):
    """
    Parameters for scikit-learn's OrthogonalMatchingPursuit.

    Without ``n_nonzero_coefs`` and ``tol``, scikit-learn keeps 10% of the features,
    at least one. The two stopping rules are mutually exclusive.

    Attributes
    ----------
    n_nonzero_coefs : int | None
        Target number of non-zero coefficients.
    tol : float | None
        Target squared norm of the residual.
    is_intercept_fitted : bool
        Whether to fit an intercept.
    scaling : FeatureScaling
        Feature scaling; the selection path depends on feature scale.
    """

    n_nonzero_coefs: int | None = None
    tol: float | None = None
    is_intercept_fitted: bool = True
    scaling: FeatureScaling = FeatureScaling.STANDARD

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_integers(
            n_nonzero_coefs=self.n_nonzero_coefs,
        )
        if self.n_nonzero_coefs is not None and self.n_nonzero_coefs <= 0:
            raise ValueError("n_nonzero_coefs must be greater than 0 when set")
        if self.tol is not None:
            ParameterValidator.require_not_nan(tol=self.tol)
            if self.tol <= 0:
                raise ValueError("tol must be greater than 0 when set")
        if self.n_nonzero_coefs is not None and self.tol is not None:
            raise ValueError("n_nonzero_coefs and tol are mutually exclusive")

    @override
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted orthogonal matching pursuit processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator: OrthogonalMatchingPursuit = OrthogonalMatchingPursuit(
            n_nonzero_coefs=self.n_nonzero_coefs,
            tol=self.tol,
            fit_intercept=self.is_intercept_fitted,
        )
        return RegressionProcessor(
            method=RegressionMethod.ORTHOGONAL_MATCHING_PURSUIT,
            estimator=self.scaling.build_pipeline(estimator),
        )
