from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.ensemble import VotingRegressor

from ....parameter_validator import ParameterValidator
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor


@dataclass(frozen=True)
class VotingParameters(RegressionParameters):
    """
    Parameters for scikit-learn's VotingRegressor.

    Predictions are the weighted average of the predictions of all methods.

    Attributes
    ----------
    estimators : tuple[RegressionParameters, ...]
        The methods that vote, each with its own feature scaling.
    weights : tuple[float, ...] | None
        Weight of each method; equal weights when None.
    n_jobs : int | None
        Number of parallel jobs.
    """

    estimators: tuple[RegressionParameters, ...]
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

    @override
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted voting processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator: VotingRegressor = VotingRegressor(
            estimators=self._build_named_estimators(),
            weights=None if self.weights is None else list(self.weights),
            n_jobs=self.n_jobs,
        )
        return RegressionProcessor(
            method=RegressionMethod.VOTING,
            estimator=FeatureScaling.NONE.build_pipeline(estimator),
        )

    def _build_named_estimators(self) -> list[tuple[str, object]]:
        named_estimators: list[tuple[str, object]] = []
        for index, parameters in enumerate(self.estimators):
            processor: RegressionProcessor = parameters.build_processor()
            named_estimators.append(
                (f"{processor.method.value}_{index}", processor.estimator)
            )
        return named_estimators
