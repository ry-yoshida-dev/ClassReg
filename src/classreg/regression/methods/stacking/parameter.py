from dataclasses import dataclass
from typing import override

from sklearn.ensemble import StackingRegressor

from ....feature_scaling import FeatureScaling
from ....parameter_validator import ParameterValidator
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor
from ..ridge import RidgeParameters


@dataclass(frozen=True)
class StackingParameters(RegressionParameters):
    """
    Parameters for scikit-learn's StackingRegressor.

    The final method is Ridge by default, not scikit-learn's RidgeCV, because this
    package does not tune hyperparameters.

    Attributes
    ----------
    estimators : tuple[RegressionParameters, ...]
        The base methods, each with its own feature scaling.
    final_estimator : RegressionParameters
        The method trained on out-of-fold predictions of the base methods.
    n_splits : int
        Folds used to produce the out-of-fold predictions.
    is_feature_passthrough : bool
        Whether the final method also receives the original features.
    n_jobs : int | None
        Number of parallel jobs.
    """

    estimators: tuple[RegressionParameters, ...]
    final_estimator: RegressionParameters = RidgeParameters()
    n_splits: int = 5
    is_feature_passthrough: bool = False
    n_jobs: int | None = None

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_integers(
            n_splits=self.n_splits,
            n_jobs=self.n_jobs,
        )
        if not self.estimators:
            raise ValueError("estimators must not be empty")
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")

    @override
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted stacking processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator: StackingRegressor = StackingRegressor(
            estimators=self._build_named_estimators(),
            final_estimator=self.final_estimator.build_processor().estimator,
            cv=self.n_splits,
            passthrough=self.is_feature_passthrough,
            n_jobs=self.n_jobs,
        )
        return RegressionProcessor(
            method=RegressionMethod.STACKING,
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
