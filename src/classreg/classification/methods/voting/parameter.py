from dataclasses import dataclass
from typing import override

from sklearn.ensemble import VotingClassifier

from ....feature_scaling import FeatureScaling
from ....parameter_validator import ParameterValidator
from ...method import ClassificationMethod
from ...metrics import ClassScoreKind
from ...parameter import ClassificationParameters
from ...processor import ClassificationProcessor
from ...processors import ProbabilisticClassificationProcessor


@dataclass(frozen=True)
class VotingParameters(ClassificationParameters):
    """
    Parameters for scikit-learn's VotingClassifier.

    Votes are always soft, averaging class probabilities, because hard votes
    produce no per-class scores. Methods without class probabilities are rejected.

    Attributes
    ----------
    estimators : tuple[ClassificationParameters, ...]
        The methods that vote, each with its own feature scaling.
    weights : tuple[float, ...] | None
        Weight of each method; equal weights when None.
    n_jobs : int | None
        Number of parallel jobs.
    """

    estimators: tuple[ClassificationParameters, ...]
    weights: tuple[float, ...] | None = None
    n_jobs: int | None = None

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_integers(
            n_jobs=self.n_jobs,
        )
        if not self.estimators:
            raise ValueError("estimators must not be empty")
        if self.weights is not None:
            if len(self.weights) != len(self.estimators):
                raise ValueError("weights must have one value per estimator")
            for weight in self.weights:
                ParameterValidator.require_not_nan(weight=weight)
                if weight < 0:
                    raise ValueError("weights must be non-negative")
            if not any(self.weights):
                raise ValueError("weights must not all be 0")
        for parameters in self.estimators:
            if (
                parameters.build_processor().score_kind
                is not ClassScoreKind.PROBABILITY
            ):
                raise ValueError(
                    f"{type(parameters).__name__} produces no class probabilities"
                )

    @override
    def build_processor(self) -> ProbabilisticClassificationProcessor:
        """
        Build an unfitted voting processor.

        Returns
        -------
        ProbabilisticClassificationProcessor
            The processor.
        """
        estimator: VotingClassifier = VotingClassifier(
            estimators=self._build_named_estimators(),
            voting="soft",
            weights=None if self.weights is None else list(self.weights),
            n_jobs=self.n_jobs,
        )
        return ProbabilisticClassificationProcessor(
            method=ClassificationMethod.VOTING,
            estimator=FeatureScaling.NONE.build_pipeline(estimator),
        )

    def _build_named_estimators(self) -> list[tuple[str, object]]:
        named_estimators: list[tuple[str, object]] = []
        for index, parameters in enumerate(self.estimators):
            processor: ClassificationProcessor = parameters.build_processor()
            named_estimators.append(
                (f"{processor.method.value}_{index}", processor.underlying_estimator)
            )
        return named_estimators
