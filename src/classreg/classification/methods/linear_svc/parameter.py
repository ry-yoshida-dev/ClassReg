from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.svm import LinearSVC

from ....parameter_validator import ParameterValidator
from ....random_state import DEFAULT_RANDOM_STATE
from ...class_weighting import ClassWeighting
from ...method import ClassificationMethod
from ...parameter import ClassificationParameters
from ...processors import MarginClassificationProcessor
from .loss import LinearSVCLoss
from .penalty import LinearSVCPenalty


@dataclass(frozen=True)
class LinearSVCParameters(ClassificationParameters):
    """
    Parameters for scikit-learn's LinearSVC.

    Multiclass problems are trained one-vs-rest, and the primal or dual
    formulation is chosen automatically. Scores are decision margins, so
    ``log_loss`` is undefined for this method.

    Attributes
    ----------
    C : float
        Inverse regularization strength.
    penalty : LinearSVCPenalty
        The regularization penalty. L1 cannot be combined with the hinge loss.
    loss : LinearSVCLoss
        The loss function to optimize.
    tol : float
        Stopping tolerance.
    max_iter : int
        Maximum number of solver iterations.
    is_intercept_fitted : bool
        Whether to fit an intercept.
    class_weighting : ClassWeighting
        How samples are weighted according to their class.
    scaling : FeatureScaling
        Feature scaling; regularization depends on feature scale.
    random_state : int | None
        Seed for the dual coordinate descent.
    """

    C: float = 1.0
    penalty: LinearSVCPenalty = LinearSVCPenalty.L2
    loss: LinearSVCLoss = LinearSVCLoss.SQUARED_HINGE
    tol: float = 1e-4
    max_iter: int = 1000
    is_intercept_fitted: bool = True
    class_weighting: ClassWeighting = ClassWeighting.UNIFORM
    scaling: FeatureScaling = FeatureScaling.STANDARD
    random_state: int | None = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_integers(
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        ParameterValidator.require_not_nan(
            C=self.C,
            tol=self.tol,
        )
        if self.C <= 0:
            raise ValueError("C must be greater than 0")
        if self.penalty is LinearSVCPenalty.L1 and self.loss is LinearSVCLoss.HINGE:
            raise ValueError("The L1 penalty cannot be combined with the hinge loss")
        if self.tol <= 0:
            raise ValueError("tol must be greater than 0")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")

    @override
    def build_processor(self) -> MarginClassificationProcessor:
        """
        Build an unfitted linear support vector classification processor.

        Returns
        -------
        MarginClassificationProcessor
            The processor.
        """
        estimator: LinearSVC = LinearSVC(
            penalty=self.penalty.value,
            loss=self.loss.value,
            dual="auto",
            tol=self.tol,
            C=self.C,
            fit_intercept=self.is_intercept_fitted,
            class_weight=self.class_weighting.to_estimator_argument(),
            random_state=self.random_state,
            max_iter=self.max_iter,
        )
        return MarginClassificationProcessor(
            method=ClassificationMethod.LINEAR_SVC,
            estimator=self.scaling.build_pipeline(estimator),
        )
