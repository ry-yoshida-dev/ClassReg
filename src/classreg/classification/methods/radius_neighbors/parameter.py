from dataclasses import dataclass
from typing import override

from sklearn.neighbors import RadiusNeighborsClassifier

from ....feature_scaling import FeatureScaling
from ....options import (
    KNearestNeighborsAlgorithm,
    KNearestNeighborsMetric,
    KNearestNeighborsWeighting,
)
from ....parameter_validator import ParameterValidator
from ...method import ClassificationMethod
from ...parameter import ClassificationParameters
from ...processors import (
    ProbabilisticClassificationProcessor,
)
from .outlier_handling import RadiusNeighborsOutlierHandling


@dataclass(frozen=True)
class RadiusNeighborsParameters(ClassificationParameters):
    """
    Parameters for scikit-learn's RadiusNeighborsClassifier.

    A radius that suits the data is essential: the default raises ValueError
    during prediction whenever a sample has no neighbor within the radius, which
    is common with many features.

    Attributes
    ----------
    radius : float
        Neighborhood radius, in scaled feature units.
    weighting : KNearestNeighborsWeighting
        How neighbors contribute to a prediction.
    algorithm : KNearestNeighborsAlgorithm
        Neighbor search backend.
    leaf_size : int
        Leaf size for tree-based backends.
    metric : KNearestNeighborsMetric
        Distance metric.
    outlier_handling : RadiusNeighborsOutlierHandling
        What happens to samples without neighbors.
    n_jobs : int | None
        Number of parallel jobs.
    scaling : FeatureScaling
        Feature scaling; distances depend on feature scale.
    """

    radius: float = 1.0
    weighting: KNearestNeighborsWeighting = KNearestNeighborsWeighting.UNIFORM
    algorithm: KNearestNeighborsAlgorithm = KNearestNeighborsAlgorithm.AUTO
    leaf_size: int = 30
    metric: KNearestNeighborsMetric = KNearestNeighborsMetric.EUCLIDEAN
    outlier_handling: RadiusNeighborsOutlierHandling = (
        RadiusNeighborsOutlierHandling.RAISE
    )
    n_jobs: int | None = None
    scaling: FeatureScaling = FeatureScaling.STANDARD

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_integers(
            leaf_size=self.leaf_size,
            n_jobs=self.n_jobs,
        )
        ParameterValidator.require_not_nan(
            radius=self.radius,
        )
        if self.radius <= 0:
            raise ValueError("radius must be greater than 0")
        if self.leaf_size <= 0:
            raise ValueError("leaf_size must be greater than 0")
        if self.metric == KNearestNeighborsMetric.COSINE and self.algorithm in (
            KNearestNeighborsAlgorithm.BALL_TREE,
            KNearestNeighborsAlgorithm.KD_TREE,
        ):
            raise ValueError("cosine metric requires the auto or brute algorithm")

    @override
    def build_processor(self) -> ProbabilisticClassificationProcessor:
        """
        Build an unfitted radius neighbors processor.

        Returns
        -------
        ProbabilisticClassificationProcessor
            The processor.
        """
        estimator: RadiusNeighborsClassifier = RadiusNeighborsClassifier(
            radius=self.radius,
            weights=self.weighting.value,
            algorithm=self.algorithm.value,
            leaf_size=self.leaf_size,
            metric=self.metric.value,
            outlier_label=self.outlier_handling.to_estimator_argument(),
            n_jobs=self.n_jobs,
        )
        return ProbabilisticClassificationProcessor(
            method=ClassificationMethod.RADIUS_NEIGHBORS,
            estimator=self.scaling.build_pipeline(estimator),
        )
