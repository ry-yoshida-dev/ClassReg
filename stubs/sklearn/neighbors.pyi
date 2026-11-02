from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

class KNeighborsClassifier:
    n_neighbors: int
    weights: str
    metric: str

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        weights: str = "uniform",
        algorithm: str = "auto",
        leaf_size: int = 30,
        metric: str = "minkowski",
        n_jobs: int | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...

class KNeighborsRegressor:
    n_neighbors: int
    weights: str
    metric: str

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        weights: str = "uniform",
        algorithm: str = "auto",
        leaf_size: int = 30,
        metric: str = "minkowski",
        n_jobs: int | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...

class NearestCentroid:
    shrink_threshold: float | None
    priors: str

    def __init__(
        self,
        metric: str = "euclidean",
        *,
        shrink_threshold: float | None = None,
        priors: str = "uniform",
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...

class RadiusNeighborsClassifier:
    radius: float
    outlier_label: str | None

    def __init__(
        self,
        radius: float = 1.0,
        *,
        weights: str = "uniform",
        algorithm: str = "auto",
        leaf_size: int = 30,
        metric: str = "minkowski",
        outlier_label: str | None = None,
        n_jobs: int | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...

class RadiusNeighborsRegressor:
    radius: float
    weights: str

    def __init__(
        self,
        radius: float = 1.0,
        *,
        weights: str = "uniform",
        algorithm: str = "auto",
        leaf_size: int = 30,
        metric: str = "minkowski",
        n_jobs: int | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...
