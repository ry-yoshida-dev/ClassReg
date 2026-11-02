from dataclasses import dataclass
from typing import override

from sklearn.linear_model import TheilSenRegressor

from ....feature_scaling import FeatureScaling
from ....parameter_validator import ParameterValidator
from ....random_state import DEFAULT_RANDOM_STATE
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor


@dataclass(frozen=True)
class TheilSenParameters(RegressionParameters):
    """
    Parameters for scikit-learn's TheilSenRegressor.

    Fitting raises ValueError when ``n_subsamples`` is smaller than the number of
    coefficients or larger than the number of samples.

    Attributes
    ----------
    max_subpopulation : int
        Maximum number of subsets considered.
    n_subsamples : int | None
        Samples per subset; the smallest valid number when None.
    max_iter : int
        Maximum number of iterations of the spatial median.
    tol : float
        Stopping tolerance.
    is_intercept_fitted : bool
        Whether to fit an intercept.
    n_jobs : int | None
        Number of parallel jobs.
    scaling : FeatureScaling
        Feature scaling; the spatial median depends on feature scale.
    random_state : int | None
        Seed for subset sampling.
    """

    max_subpopulation: int = 10000
    n_subsamples: int | None = None
    max_iter: int = 300
    tol: float = 1e-3
    is_intercept_fitted: bool = True
    n_jobs: int | None = None
    scaling: FeatureScaling = FeatureScaling.STANDARD
    random_state: int | None = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_integers(
            max_subpopulation=self.max_subpopulation,
            n_subsamples=self.n_subsamples,
            max_iter=self.max_iter,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        ParameterValidator.require_not_nan(
            tol=self.tol,
        )
        if self.max_subpopulation <= 0:
            raise ValueError("max_subpopulation must be greater than 0")
        if self.n_subsamples is not None and self.n_subsamples <= 0:
            raise ValueError("n_subsamples must be greater than 0 when set")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")
        if self.tol < 0:
            raise ValueError("tol must be non-negative")

    @override
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted Theil-Sen processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator: TheilSenRegressor = TheilSenRegressor(
            max_subpopulation=self.max_subpopulation,
            n_subsamples=self.n_subsamples,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.is_intercept_fitted,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        return RegressionProcessor(
            method=RegressionMethod.THEIL_SEN,
            estimator=self.scaling.build_pipeline(estimator),
        )
