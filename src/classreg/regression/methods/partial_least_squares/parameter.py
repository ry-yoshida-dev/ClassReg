from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.cross_decomposition import PLSRegression

from ....parameter_validator import ParameterValidator
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor


@dataclass(frozen=True)
class PartialLeastSquaresParameters(RegressionParameters):
    """
    Parameters for scikit-learn's PLSRegression.

    Fitting raises ValueError when ``n_components`` exceeds the number of features
    or samples.

    Attributes
    ----------
    n_components : int
        Number of latent components.
    is_scaled : bool
        Whether the estimator standardizes features and targets internally.
    max_iter : int
        Maximum number of iterations per component.
    tol : float
        Stopping tolerance.
    scaling : FeatureScaling
        Feature scaling applied before the estimator.
    """

    n_components: int = 2
    is_scaled: bool = True
    max_iter: int = 500
    tol: float = 1e-6
    scaling: FeatureScaling = FeatureScaling.NONE

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_integers(
            n_components=self.n_components,
            max_iter=self.max_iter,
        )
        ParameterValidator.require_not_nan(
            tol=self.tol,
        )
        if self.n_components <= 0:
            raise ValueError("n_components must be greater than 0")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")
        if self.tol < 0:
            raise ValueError("tol must be non-negative")

    @override
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted partial least squares processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator: PLSRegression = PLSRegression(
            n_components=self.n_components,
            scale=self.is_scaled,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        return RegressionProcessor(
            method=RegressionMethod.PARTIAL_LEAST_SQUARES,
            estimator=self.scaling.build_pipeline(estimator),
        )
