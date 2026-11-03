from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ....parameter_validator import ParameterValidator
from ...method import ClassificationMethod
from ...parameter import ClassificationParameters
from ...processors import ProbabilisticClassificationProcessor
from .solver import LinearDiscriminantAnalysisSolver


@dataclass(frozen=True)
class LinearDiscriminantAnalysisParameters(ClassificationParameters):
    """
    Parameters for scikit-learn's LinearDiscriminantAnalysis.

    Attributes
    ----------
    solver : LinearDiscriminantAnalysisSolver
        The solver.
    shrinkage : float | None
        Covariance shrinkage in the range [0, 1]; no shrinkage when None.
    tol : float
        Rank threshold of the SVD solver.
    """

    solver: LinearDiscriminantAnalysisSolver = LinearDiscriminantAnalysisSolver.SVD
    shrinkage: float | None = None
    tol: float = 1e-4

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_not_nan(
            tol=self.tol,
        )
        if self.tol < 0:
            raise ValueError("tol must be non-negative")
        if self.shrinkage is None:
            return
        if not 0 <= self.shrinkage <= 1:
            raise ValueError("shrinkage must be in the range [0, 1] when set")
        if self.solver is LinearDiscriminantAnalysisSolver.SVD:
            raise ValueError("solver svd does not support shrinkage")

    @override
    def build_processor(self) -> ProbabilisticClassificationProcessor:
        """
        Build an unfitted linear discriminant analysis processor.

        Returns
        -------
        ProbabilisticClassificationProcessor
            The processor.
        """
        estimator: LinearDiscriminantAnalysis = LinearDiscriminantAnalysis(
            solver=self.solver.value,
            shrinkage=self.shrinkage,
            tol=self.tol,
        )
        return ProbabilisticClassificationProcessor(
            method=ClassificationMethod.LINEAR_DISCRIMINANT_ANALYSIS,
            estimator=FeatureScaling.NONE.build_pipeline(estimator),
        )
