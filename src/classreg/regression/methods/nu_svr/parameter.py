from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.svm import NuSVR

from ....options import (
    SVMGamma,
    SVMKernel,
)
from ....parameter_validator import ParameterValidator
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor


@dataclass(frozen=True)
class NuSVRParameters(RegressionParameters):
    """
    Parameters for scikit-learn's NuSVR.

    Targets are not scaled, so targets of large magnitude may need a larger ``C``
    to fit well.

    Attributes
    ----------
    nu : float
        Upper bound on the fraction of training errors and lower bound on the fraction of support vectors.
    C : float
        Inverse regularization strength.
    kernel : SVMKernel
        The kernel function.
    degree : int
        Degree of the polynomial kernel.
    gamma : SVMGamma | float
        Kernel coefficient, either a rule or a positive value.
    coef0 : float
        Independent term of the polynomial and sigmoid kernels.
    tol : float
        Stopping tolerance.
    max_iter : int | None
        Maximum number of solver iterations; no limit when None.
    scaling : FeatureScaling
        Feature scaling; kernels depend on feature scale.
    """

    nu: float = 0.5
    C: float = 1.0
    kernel: SVMKernel = SVMKernel.RBF
    degree: int = 3
    gamma: SVMGamma | float = SVMGamma.SCALE
    coef0: float = 0.0
    tol: float = 1e-3
    max_iter: int | None = None
    scaling: FeatureScaling = FeatureScaling.STANDARD

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_integers(
            degree=self.degree,
            max_iter=self.max_iter,
        )
        ParameterValidator.require_not_nan(
            nu=self.nu,
            C=self.C,
            coef0=self.coef0,
            tol=self.tol,
        )
        if self.C <= 0:
            raise ValueError("C must be greater than 0")
        if self.degree < 0:
            raise ValueError("degree must be non-negative")
        if not isinstance(self.gamma, SVMGamma):
            ParameterValidator.require_not_nan(gamma=self.gamma)
            if self.gamma <= 0:
                raise ValueError("gamma must be greater than 0 when numeric")
        if self.tol <= 0:
            raise ValueError("tol must be greater than 0")
        if self.max_iter is not None and self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0 when set")
        if not 0 < self.nu <= 1:
            raise ValueError("nu must be in the range (0, 1]")

    @override
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted nu support vector regression processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator: NuSVR = NuSVR(
            C=self.C,
            nu=self.nu,
            kernel=self.kernel.value,
            degree=self.degree,
            gamma=self.gamma.value if isinstance(self.gamma, SVMGamma) else self.gamma,
            coef0=self.coef0,
            tol=self.tol,
            max_iter=-1 if self.max_iter is None else self.max_iter,
        )
        return RegressionProcessor(
            method=RegressionMethod.NU_SVR,
            estimator=self.scaling.build_pipeline(estimator),
        )
