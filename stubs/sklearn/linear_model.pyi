from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

class LogisticRegression:
    C: float
    l1_ratio: float
    solver: str
    max_iter: int
    class_weight: Literal["balanced"] | None

    def __init__(
        self,
        *,
        C: float = 1.0,
        l1_ratio: float = 0.0,
        tol: float = 1e-4,
        fit_intercept: bool = True,
        class_weight: Literal["balanced"] | None = None,
        random_state: int | None = None,
        solver: str = "lbfgs",
        max_iter: int = 100,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...

class LinearRegression:
    fit_intercept: bool
    positive: bool

    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        n_jobs: int | None = None,
        positive: bool = False,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...

class Ridge:
    alpha: float

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        fit_intercept: bool = True,
        max_iter: int | None = None,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...

class Lasso:
    alpha: float

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        positive: bool = False,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...

class ElasticNet:
    alpha: float
    l1_ratio: float

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        positive: bool = False,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...
