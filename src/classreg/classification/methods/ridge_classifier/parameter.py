from dataclasses import dataclass
from typing import override

from sklearn.linear_model import RidgeClassifier

from ....feature_scaling import FeatureScaling
from ....parameter_validator import ParameterValidator
from ....random_state import DEFAULT_RANDOM_STATE
from ...class_weighting import ClassWeighting
from ...method import ClassificationMethod
from ...parameter import ClassificationParameters
from ...processors import (
    MarginClassificationProcessor,
)


@dataclass(frozen=True)
class RidgeClassifierParameters(ClassificationParameters):
    """
    Parameters for scikit-learn's RidgeClassifier.

    Scores are decision margins, so ``log_loss`` is undefined for this method.

    Attributes
    ----------
    alpha : float
        L2 regularization strength.
    is_intercept_fitted : bool
        Whether to fit an intercept.
    max_iter : int | None
        Maximum number of solver iterations; solver default when None.
    tol : float
        Stopping tolerance.
    class_weighting : ClassWeighting
        How samples are weighted according to their class.
    scaling : FeatureScaling
        Feature scaling; the penalty depends on feature scale.
    random_state : int | None
        Seed for randomized solvers.
    """

    alpha: float = 1.0
    is_intercept_fitted: bool = True
    max_iter: int | None = None
    tol: float = 1e-4
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
            alpha=self.alpha,
            tol=self.tol,
        )
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.max_iter is not None and self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0 when set")
        if self.tol < 0:
            raise ValueError("tol must be non-negative")

    @override
    def build_processor(self) -> MarginClassificationProcessor:
        """
        Build an unfitted ridge classifier processor.

        Returns
        -------
        MarginClassificationProcessor
            The processor.
        """
        estimator: RidgeClassifier = RidgeClassifier(
            alpha=self.alpha,
            fit_intercept=self.is_intercept_fitted,
            max_iter=self.max_iter,
            tol=self.tol,
            class_weight=self.class_weighting.to_estimator_argument(),
            random_state=self.random_state,
        )
        return MarginClassificationProcessor(
            method=ClassificationMethod.RIDGE_CLASSIFIER,
            estimator=self.scaling.build_pipeline(estimator),
        )
