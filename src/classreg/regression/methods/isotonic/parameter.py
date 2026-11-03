from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.isotonic import IsotonicRegression

from ....parameter_validator import ParameterValidator
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor
from .direction import IsotonicDirection
from .out_of_bounds import IsotonicOutOfBounds


@dataclass(frozen=True)
class IsotonicParameters(RegressionParameters):
    """
    Parameters for scikit-learn's IsotonicRegression.

    Data must have exactly one feature; fitting raises ValueError otherwise.
    Predictions outside the training range are clipped by default, unlike
    scikit-learn, whose default of NaN would stop metric computation.

    Attributes
    ----------
    direction : IsotonicDirection
        The monotonic direction of the fit.
    out_of_bounds : IsotonicOutOfBounds
        What happens to samples outside the training range.
    y_min : float | None
        Lower bound on predictions; no bound when None.
    y_max : float | None
        Upper bound on predictions; no bound when None.
    """

    direction: IsotonicDirection = IsotonicDirection.INCREASING
    out_of_bounds: IsotonicOutOfBounds = IsotonicOutOfBounds.CLIP
    y_min: float | None = None
    y_max: float | None = None

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self.y_min is not None:
            ParameterValidator.require_not_nan(y_min=self.y_min)
        if self.y_max is not None:
            ParameterValidator.require_not_nan(y_max=self.y_max)
        if (
            self.y_min is not None
            and self.y_max is not None
            and self.y_min > self.y_max
        ):
            raise ValueError("y_min must not exceed y_max")

    @override
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted isotonic regression processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator: IsotonicRegression = IsotonicRegression(
            y_min=self.y_min,
            y_max=self.y_max,
            increasing=self.direction.value,
            out_of_bounds=self.out_of_bounds.value,
        )
        return RegressionProcessor(
            method=RegressionMethod.ISOTONIC,
            estimator=FeatureScaling.NONE.build_pipeline(estimator),
        )
