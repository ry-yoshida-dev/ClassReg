from dataclasses import dataclass
from typing import override

from sklearn.compose import TransformedTargetRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

from ....feature_scaling import FeatureScaling
from ....options import SVMKernel
from ....parameter_validator import ParameterValidator
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor


@dataclass(frozen=True)
class KernelRidgeParameters(RegressionParameters):
    """
    Parameters for scikit-learn's KernelRidge.

    Targets are centered by default, unlike scikit-learn, because this estimator
    has no intercept and otherwise pulls predictions towards zero. Training time
    grows with the cube of the number of samples.

    Attributes
    ----------
    alpha : float
        L2 regularization strength.
    kernel : SVMKernel
        The kernel function.
    gamma : float | None
        Kernel coefficient; scikit-learn's default of one over the number of
        features when None. Unused by the linear kernel.
    degree : int
        Degree of the polynomial kernel.
    coef0 : float
        Independent term of the polynomial and sigmoid kernels.
    is_target_centered : bool
        Whether the training mean of the targets is removed while fitting.
    scaling : FeatureScaling
        Feature scaling; kernels depend on feature scale.
    """

    alpha: float = 1.0
    kernel: SVMKernel = SVMKernel.LINEAR
    gamma: float | None = None
    degree: int = 3
    coef0: float = 1.0
    is_target_centered: bool = True
    scaling: FeatureScaling = FeatureScaling.STANDARD

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_integers(
            degree=self.degree,
        )
        ParameterValidator.require_not_nan(
            alpha=self.alpha,
            coef0=self.coef0,
        )
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.gamma is not None:
            ParameterValidator.require_not_nan(gamma=self.gamma)
            if self.gamma <= 0:
                raise ValueError("gamma must be greater than 0 when set")
        if self.degree < 0:
            raise ValueError("degree must be non-negative")

    @override
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted kernel ridge processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator: KernelRidge = KernelRidge(
            alpha=self.alpha,
            kernel=self.kernel.value,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
        )
        if not self.is_target_centered:
            return RegressionProcessor(
                method=RegressionMethod.KERNEL_RIDGE,
                estimator=self.scaling.build_pipeline(estimator),
            )
        centered: TransformedTargetRegressor = TransformedTargetRegressor(
            regressor=estimator,
            transformer=StandardScaler(with_std=False),
        )
        return RegressionProcessor(
            method=RegressionMethod.KERNEL_RIDGE,
            estimator=self.scaling.build_pipeline(centered),
        )
