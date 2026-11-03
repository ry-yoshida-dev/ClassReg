from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.gaussian_process import GaussianProcessRegressor

from ....options import GaussianProcessKernel
from ....parameter_validator import ParameterValidator
from ....random_state import DEFAULT_RANDOM_STATE
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor


@dataclass(frozen=True)
class GaussianProcessParameters(RegressionParameters):
    """
    Parameters for scikit-learn's GaussianProcessRegressor.

    Training time grows with the cube of the number of samples, so this method
    suits small datasets. Targets are normalized by default, unlike scikit-learn,
    because the zero-mean prior otherwise pulls predictions towards zero.

    Attributes
    ----------
    kernel : GaussianProcessKernel
        The covariance shape.
    alpha : float
        Value added to the kernel diagonal for numerical stability.
    n_restarts_optimizer : int
        Additional restarts of the kernel hyperparameter optimizer.
    is_target_normalized : bool
        Whether targets are scaled to zero mean and unit variance while fitting.
    scaling : FeatureScaling
        Feature scaling; kernel length scales depend on feature scale.
    random_state : int | None
        Seed for the optimizer restarts.
    """

    kernel: GaussianProcessKernel = GaussianProcessKernel.RBF
    alpha: float = 1e-10
    n_restarts_optimizer: int = 0
    is_target_normalized: bool = True
    scaling: FeatureScaling = FeatureScaling.STANDARD
    random_state: int | None = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_integers(
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state,
        )
        ParameterValidator.require_not_nan(
            alpha=self.alpha,
        )
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.n_restarts_optimizer < 0:
            raise ValueError("n_restarts_optimizer must be non-negative")

    @override
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted Gaussian process processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator: GaussianProcessRegressor = GaussianProcessRegressor(
            kernel=self.kernel.build(),
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=self.is_target_normalized,
            random_state=self.random_state,
        )
        return RegressionProcessor(
            method=RegressionMethod.GAUSSIAN_PROCESS,
            estimator=self.scaling.build_pipeline(estimator),
        )
