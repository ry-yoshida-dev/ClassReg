from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.neural_network import MLPClassifier

from ....options import MultilayerPerceptronActivation, MultilayerPerceptronSolver
from ....parameter_validator import ParameterValidator
from ....random_state import DEFAULT_RANDOM_STATE
from ...method import ClassificationMethod
from ...parameter import ClassificationParameters
from ...processors import ProbabilisticClassificationProcessor


@dataclass(frozen=True)
class MultilayerPerceptronParameters(ClassificationParameters):
    """
    Parameters for scikit-learn's MLPClassifier.

    Attributes
    ----------
    hidden_layer_sizes : tuple[int, ...]
        Number of units in each hidden layer.
    activation : MultilayerPerceptronActivation
        Activation function of the hidden layers.
    solver : MultilayerPerceptronSolver
        The weight optimization algorithm.
    alpha : float
        L2 regularization strength.
    learning_rate_init : float
        Initial learning rate of the stochastic solvers.
    max_iter : int
        Maximum number of solver iterations.
    tol : float
        Minimum improvement counted as progress.
    is_early_stopping_enabled : bool
        Whether to stop when an internal validation split stops improving.
    validation_fraction : float
        Fraction of the training data held out for early stopping.
    n_iter_no_change : int
        Iterations without improvement before stopping.
    scaling : FeatureScaling
        Feature scaling; gradient-based training depends on feature scale.
    random_state : int | None
        Seed for weight initialization, batching, and the early stopping split.
    """

    hidden_layer_sizes: tuple[int, ...] = (100,)
    activation: MultilayerPerceptronActivation = MultilayerPerceptronActivation.RELU
    solver: MultilayerPerceptronSolver = MultilayerPerceptronSolver.ADAM
    alpha: float = 1e-4
    learning_rate_init: float = 1e-3
    max_iter: int = 200
    tol: float = 1e-4
    is_early_stopping_enabled: bool = False
    validation_fraction: float = 0.1
    n_iter_no_change: int = 10
    scaling: FeatureScaling = FeatureScaling.STANDARD
    random_state: int | None = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_integers(
            max_iter=self.max_iter,
            n_iter_no_change=self.n_iter_no_change,
            random_state=self.random_state,
            **{
                f"hidden_layer_sizes[{index}]": size
                for index, size in enumerate(self.hidden_layer_sizes)
            },
        )
        ParameterValidator.require_not_nan(
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            tol=self.tol,
        )
        if not self.hidden_layer_sizes:
            raise ValueError("hidden_layer_sizes must not be empty")
        if any(size <= 0 for size in self.hidden_layer_sizes):
            raise ValueError("hidden_layer_sizes must contain only positive sizes")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.learning_rate_init <= 0:
            raise ValueError("learning_rate_init must be greater than 0")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")
        if self.tol < 0:
            raise ValueError("tol must be non-negative")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in the range (0, 1)")
        if self.n_iter_no_change <= 0:
            raise ValueError("n_iter_no_change must be greater than 0")

    @override
    def build_processor(self) -> ProbabilisticClassificationProcessor:
        """
        Build an unfitted multilayer perceptron processor.

        Returns
        -------
        ProbabilisticClassificationProcessor
            The processor.
        """
        estimator: MLPClassifier = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation.value,
            solver=self.solver.value,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            tol=self.tol,
            early_stopping=self.is_early_stopping_enabled,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            random_state=self.random_state,
        )
        return ProbabilisticClassificationProcessor(
            method=ClassificationMethod.MULTILAYER_PERCEPTRON,
            estimator=self.scaling.build_pipeline(estimator),
        )
