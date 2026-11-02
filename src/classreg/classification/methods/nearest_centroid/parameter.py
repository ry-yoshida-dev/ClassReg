from dataclasses import dataclass
from typing import override

from sklearn.neighbors import NearestCentroid

from ....feature_scaling import FeatureScaling
from ....parameter_validator import ParameterValidator
from ...method import ClassificationMethod
from ...parameter import ClassificationParameters
from ...processors import (
    ProbabilisticClassificationProcessor,
)


@dataclass(frozen=True)
class NearestCentroidParameters(ClassificationParameters):
    """
    Parameters for scikit-learn's NearestCentroid.

    Only the Euclidean metric is offered, because scikit-learn does not provide
    class probabilities for the Manhattan metric.

    Attributes
    ----------
    shrink_threshold : float | None
        Threshold for shrinking centroids; no shrinking when None.
    is_prior_fitted : bool
        Whether class priors are learned from the data rather than uniform.
    scaling : FeatureScaling
        Feature scaling; distances depend on feature scale.
    """

    shrink_threshold: float | None = None
    is_prior_fitted: bool = False
    scaling: FeatureScaling = FeatureScaling.STANDARD

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self.shrink_threshold is not None:
            ParameterValidator.require_not_nan(shrink_threshold=self.shrink_threshold)
            if self.shrink_threshold <= 0:
                raise ValueError("shrink_threshold must be greater than 0 when set")

    @override
    def build_processor(self) -> ProbabilisticClassificationProcessor:
        """
        Build an unfitted nearest centroid processor.

        Returns
        -------
        ProbabilisticClassificationProcessor
            The processor.
        """
        estimator: NearestCentroid = NearestCentroid(
            metric="euclidean",
            shrink_threshold=self.shrink_threshold,
            priors="empirical" if self.is_prior_fitted else "uniform",
        )
        return ProbabilisticClassificationProcessor(
            method=ClassificationMethod.NEAREST_CENTROID,
            estimator=self.scaling.build_pipeline(estimator),
        )
