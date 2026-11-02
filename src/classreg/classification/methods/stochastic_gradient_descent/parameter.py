from dataclasses import dataclass
from typing import override

from sklearn.linear_model import SGDClassifier

from ....feature_scaling import FeatureScaling
from ....parameter_validator import ParameterValidator
from ....random_state import DEFAULT_RANDOM_STATE
from ...class_weighting import ClassWeighting
from ...method import ClassificationMethod
from ...parameter import ClassificationParameters
from ...processors import MarginClassificationProcessor
from .loss import StochasticGradientDescentLoss


@dataclass(frozen=True)
class StochasticGradientDescentParameters(ClassificationParameters):
    """
    Parameters for scikit-learn's SGDClassifier.

    Scores are decision margins for every loss, so ``log_loss`` is undefined for
    this method even when training with StochasticGradientDescentLoss.LOG_LOSS.

    Attributes
    ----------
    loss : StochasticGradientDescentLoss
        The loss function to optimize.
    alpha : float
        Regularization strength.
    l1_ratio : float
        Elastic-net mixing: 0 is pure L2 and 1 is pure L1.
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
        Feature scaling; regularization and step sizes depend on feature scale.
    random_state : int | None
        Seed for shuffling.
    """

    loss: StochasticGradientDescentLoss = StochasticGradientDescentLoss.HINGE
    alpha: float = 1e-4
    l1_ratio: float = 0.0
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
            tol=self.tol,
        )
        if self.alpha <= 0:
            raise ValueError("alpha must be greater than 0")
        if not 0 <= self.l1_ratio <= 1:
            raise ValueError("l1_ratio must be in the range [0, 1]")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")
        if self.tol < 0:
            raise ValueError("tol must be non-negative")

    @override
    def build_processor(self) -> MarginClassificationProcessor:
        """
        Build an unfitted stochastic gradient descent processor.

        Returns
        -------
        MarginClassificationProcessor
            The processor.
        """
        estimator: SGDClassifier = SGDClassifier(
            loss=self.loss.value,
            penalty="elasticnet",
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.is_intercept_fitted,
            shuffle=self.is_shuffled,
            class_weight=self.class_weighting.to_estimator_argument(),
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        return MarginClassificationProcessor(
            method=ClassificationMethod.STOCHASTIC_GRADIENT_DESCENT,
            estimator=self.scaling.build_pipeline(estimator),
        )
