from dataclasses import dataclass
from typing import override

from sklearn.linear_model import Perceptron

from ....feature_scaling import FeatureScaling
from ....parameter_validator import ParameterValidator
from ....random_state import DEFAULT_RANDOM_STATE
from ...class_weighting import ClassWeighting
from ...method import ClassificationMethod
from ...parameter import ClassificationParameters
from ...processors import (
    MarginClassificationProcessor,
)
from .penalty import PerceptronPenalty


@dataclass(frozen=True)
class PerceptronParameters(ClassificationParameters):
    """
    Parameters for scikit-learn's Perceptron.

    Multiclass problems are trained one-vs-rest. Scores are decision margins, so
    ``log_loss`` is undefined for this method.

    Attributes
    ----------
    penalty : PerceptronPenalty | None
        The regularization penalty; no regularization when None.
    alpha : float
        Regularization strength.
    l1_ratio : float
        Mixing of the elastic-net penalty: 0 is pure L2 and 1 is pure L1.
    learning_rate : float
        Constant step size.
    max_iter : int
        Maximum number of passes over the training data.
    tol : float
        Stopping tolerance.
    is_intercept_fitted : bool
        Whether to fit an intercept.
    is_shuffled : bool
        Whether samples are shuffled after each pass.
    class_weighting : ClassWeighting
        How samples are weighted according to their class.
    n_jobs : int | None
        Number of parallel jobs for multiclass problems.
    scaling : FeatureScaling
        Feature scaling; step sizes depend on feature scale.
    random_state : int | None
        Seed for shuffling.
    """

    penalty: PerceptronPenalty | None = None
    alpha: float = 1e-4
    l1_ratio: float = 0.15
    learning_rate: float = 1.0
    max_iter: int = 1000
    tol: float = 1e-3
    is_intercept_fitted: bool = True
    is_shuffled: bool = True
    class_weighting: ClassWeighting = ClassWeighting.UNIFORM
    n_jobs: int | None = None
    scaling: FeatureScaling = FeatureScaling.STANDARD
    random_state: int | None = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_integers(
            max_iter=self.max_iter,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        ParameterValidator.require_not_nan(
            alpha=self.alpha,
            learning_rate=self.learning_rate,
            tol=self.tol,
        )
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if not 0 <= self.l1_ratio <= 1:
            raise ValueError("l1_ratio must be in the range [0, 1]")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")
        if self.tol < 0:
            raise ValueError("tol must be non-negative")

    @override
    def build_processor(self) -> MarginClassificationProcessor:
        """
        Build an unfitted perceptron processor.

        Returns
        -------
        MarginClassificationProcessor
            The processor.
        """
        estimator: Perceptron = Perceptron(
            penalty=None if self.penalty is None else self.penalty.value,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            eta0=self.learning_rate,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.is_intercept_fitted,
            shuffle=self.is_shuffled,
            class_weight=self.class_weighting.to_estimator_argument(),
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        return MarginClassificationProcessor(
            method=ClassificationMethod.PERCEPTRON,
            estimator=self.scaling.build_pipeline(estimator),
        )
