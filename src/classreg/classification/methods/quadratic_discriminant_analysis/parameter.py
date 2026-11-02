from dataclasses import dataclass
from typing import override

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from ....feature_scaling import FeatureScaling
from ....parameter_validator import ParameterValidator
from ...method import ClassificationMethod
from ...parameter import ClassificationParameters
from ...processors import ProbabilisticClassificationProcessor


@dataclass(frozen=True)
class QuadraticDiscriminantAnalysisParameters(ClassificationParameters):
    """
    Parameters for scikit-learn's QuadraticDiscriminantAnalysis.

    Collinear features can make fitting fail with numpy.linalg.LinAlgError, a
    ValueError; increase reg_param in that case.

    Attributes
    ----------
    reg_param : float
        Regularization of the per-class covariance, in the range [0, 1].
    tol : float
        Rank threshold of the per-class covariance.
    """

    reg_param: float = 0.0
    tol: float = 1e-4

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_not_nan(
            tol=self.tol,
        )
        if not 0 <= self.reg_param <= 1:
            raise ValueError("reg_param must be in the range [0, 1]")
        if self.tol < 0:
            raise ValueError("tol must be non-negative")

    @override
    def build_processor(self) -> ProbabilisticClassificationProcessor:
        """
        Build an unfitted quadratic discriminant analysis processor.

        Returns
        -------
        ProbabilisticClassificationProcessor
            The processor.
        """
        estimator: QuadraticDiscriminantAnalysis = QuadraticDiscriminantAnalysis(
            reg_param=self.reg_param,
            tol=self.tol,
        )
        return ProbabilisticClassificationProcessor(
            method=ClassificationMethod.QUADRATIC_DISCRIMINANT_ANALYSIS,
            estimator=FeatureScaling.NONE.build_pipeline(estimator),
        )
