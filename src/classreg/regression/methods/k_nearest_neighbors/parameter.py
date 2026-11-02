from dataclasses import dataclass
from typing import override

from sklearn.neighbors import KNeighborsRegressor

from ....feature_scaling import FeatureScaling
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
class KNearestNeighborsParameters(RegressionParameters):
    """
    Parameters for scikit-learn's KNeighborsRegressor.

    Attributes
    ----------
    n_neighbors : int
        Number of neighbors.
    weighting : KNearestNeighborsWeighting
        How neighbors contribute to a prediction.
    algorithm : KNearestNeighborsAlgorithm
        Neighbor search backend.
    leaf_size : int
        Leaf size for tree-based backends.
    metric : KNearestNeighborsMetric
        Distance metric.
    n_jobs : int | None
        Number of parallel jobs for the neighbor search.
    scaling : FeatureScaling
        Feature scaling; distances depend on feature scale.
    """

    n_neighbors: int = 5
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
            n_neighbors=self.n_neighbors,
            leaf_size=self.leaf_size,
            n_jobs=self.n_jobs,
        )
        if self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be greater than 0")
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
        Build an unfitted k-nearest neighbors processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator: KNeighborsRegressor = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weighting.value,
            algorithm=self.algorithm.value,
            leaf_size=self.leaf_size,
            metric=self.metric.value,
            n_jobs=self.n_jobs,
        )
        return RegressionProcessor(
            method=RegressionMethod.K_NEAREST_NEIGHBORS,
            estimator=self.scaling.build_pipeline(estimator),
        )
