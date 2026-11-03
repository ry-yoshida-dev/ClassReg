from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.neighbors import RadiusNeighborsRegressor

from ....options import (
    KNearestNeighborsAlgorithm,
    KNearestNeighborsMetric,
    KNearestNeighborsWeighting,
)
from ....parameter_validator import ParameterValidator
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor


@dataclass(frozen=True)
class RadiusNeighborsParameters(RegressionParameters):
    """
    Parameters for scikit-learn's RadiusNeighborsRegressor.

    A radius that suits the data is essential: samples without a neighbor within
    the radius are predicted as NaN, with a warning, and ``evaluate`` then raises
    ValueError.

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
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted radius neighbors processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator: RadiusNeighborsRegressor = RadiusNeighborsRegressor(
            radius=self.radius,
            weights=self.weighting.value,
            algorithm=self.algorithm.value,
            leaf_size=self.leaf_size,
            metric=self.metric.value,
            n_jobs=self.n_jobs,
        )
        return RegressionProcessor(
            method=RegressionMethod.RADIUS_NEIGHBORS,
            estimator=self.scaling.build_pipeline(estimator),
        )
