from dataclasses import dataclass
from typing import override

from sklearn.svm import LinearSVR

from ....feature_scaling import FeatureScaling
from ....parameter_validator import ParameterValidator
from ....random_state import DEFAULT_RANDOM_STATE
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor
from .loss import LinearSVRLoss


@dataclass(frozen=True)
class LinearSVRParameters(RegressionParameters):
    """
    Parameters for scikit-learn's LinearSVR.

    The primal or dual formulation is chosen automatically. Targets are not
    scaled, so targets of large magnitude may need a larger ``C`` to fit well.

    Attributes
    ----------
    C : float
        Inverse regularization strength.
    epsilon : float
        Half-width of the tube within which errors are not penalized, in target
        units.
    loss : LinearSVRLoss
        The loss function to optimize.
    tol : float
        Stopping tolerance.
    max_iter : int
        Maximum number of solver iterations.
    is_intercept_fitted : bool
        Whether to fit an intercept.
    scaling : FeatureScaling
        Feature scaling; regularization depends on feature scale.
    random_state : int | None
        Seed for the dual coordinate descent.
    """

    C: float = 1.0
    epsilon: float = 0.0
    loss: LinearSVRLoss = LinearSVRLoss.EPSILON_INSENSITIVE
    tol: float = 1e-4
    max_iter: int = 1000
    is_intercept_fitted: bool = True
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
            epsilon=self.epsilon,
            tol=self.tol,
        )
        if self.C <= 0:
            raise ValueError("C must be greater than 0")
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        if self.tol <= 0:
            raise ValueError("tol must be greater than 0")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")

    @override
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted linear support vector regression processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator: LinearSVR = LinearSVR(
            epsilon=self.epsilon,
            tol=self.tol,
            C=self.C,
            loss=self.loss.value,
            fit_intercept=self.is_intercept_fitted,
            dual="auto",
            random_state=self.random_state,
            max_iter=self.max_iter,
        )
        return RegressionProcessor(
            method=RegressionMethod.LINEAR_SVR,
            estimator=self.scaling.build_pipeline(estimator),
        )
