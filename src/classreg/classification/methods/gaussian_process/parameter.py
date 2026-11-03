from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.gaussian_process import GaussianProcessClassifier

from ....options import (
    GaussianProcessKernel,
)
from ....parameter_validator import ParameterValidator
from ....random_state import DEFAULT_RANDOM_STATE
from ...method import ClassificationMethod
from ...parameter import ClassificationParameters
from ...processors import (
    ProbabilisticClassificationProcessor,
)


@dataclass(frozen=True)
class GaussianProcessParameters(ClassificationParameters):
    """
    Parameters for scikit-learn's GaussianProcessClassifier.

    Training time grows with the cube of the number of samples, so this method
    suits small datasets. Only the kernel shape is used, because a fitted signal
    variance makes the Laplace approximation produce NaN probabilities.

    Attributes
    ----------
    kernel : GaussianProcessKernel
        The covariance shape.
    n_restarts_optimizer : int
        Additional restarts of the kernel hyperparameter optimizer.
    max_iter_predict : int
        Maximum Newton iterations of the Laplace approximation.
    n_jobs : int | None
        Number of parallel jobs for multiclass problems.
    scaling : FeatureScaling
        Feature scaling; kernels depend on feature scale.
    random_state : int | None
        Seed for the optimizer restarts.
    """

    kernel: GaussianProcessKernel = GaussianProcessKernel.RBF
    n_restarts_optimizer: int = 0
    max_iter_predict: int = 100
    n_jobs: int | None = None
    scaling: FeatureScaling = FeatureScaling.STANDARD
    random_state: int | None = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_integers(
            n_restarts_optimizer=self.n_restarts_optimizer,
            max_iter_predict=self.max_iter_predict,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        if self.n_restarts_optimizer < 0:
            raise ValueError("n_restarts_optimizer must be non-negative")
        if self.max_iter_predict <= 0:
            raise ValueError("max_iter_predict must be greater than 0")

    @override
    def build_processor(self) -> ProbabilisticClassificationProcessor:
        """
        Build an unfitted Gaussian process processor.

        Returns
        -------
        ProbabilisticClassificationProcessor
            The processor.
        """
        estimator: GaussianProcessClassifier = GaussianProcessClassifier(
            kernel=self.kernel.build_shape(),
            n_restarts_optimizer=self.n_restarts_optimizer,
            max_iter_predict=self.max_iter_predict,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        return ProbabilisticClassificationProcessor(
            method=ClassificationMethod.GAUSSIAN_PROCESS,
            estimator=self.scaling.build_pipeline(estimator),
        )
