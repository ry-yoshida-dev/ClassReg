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

class SGDClassifier:
    loss: str
    penalty: str
    l1_ratio: float
    class_weight: Literal["balanced"] | None

    def __init__(
        self,
        loss: str = "hinge",
        *,
        penalty: str = "l2",
        alpha: float = 1e-4,
        l1_ratio: float = 0.15,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-3,
        shuffle: bool = True,
        n_jobs: int | None = None,
        random_state: int | None = None,
        class_weight: Literal["balanced"] | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...

class SGDRegressor:
    loss: str
    penalty: str
    l1_ratio: float
    epsilon: float

    def __init__(
        self,
        loss: str = "squared_error",
        *,
        penalty: str = "l2",
        alpha: float = 1e-4,
        l1_ratio: float = 0.15,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-3,
        shuffle: bool = True,
        epsilon: float = 0.1,
        random_state: int | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...

class HuberRegressor:
    epsilon: float

    def __init__(
        self,
        *,
        epsilon: float = 1.35,
        max_iter: int = 100,
        alpha: float = 1e-4,
        fit_intercept: bool = True,
        tol: float = 1e-5,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...

class QuantileRegressor:
    quantile: float

    def __init__(
        self,
        *,
        quantile: float = 0.5,
        alpha: float = 1.0,
        fit_intercept: bool = True,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...

class PoissonRegressor:
    solver: str

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        solver: str = "lbfgs",
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...

class Perceptron:
    penalty: str | None
    eta0: float

    def __init__(
        self,
        *,
        penalty: str | None = None,
        alpha: float = 1e-4,
        l1_ratio: float = 0.15,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-3,
        shuffle: bool = True,
        eta0: float = 1.0,
        n_jobs: int | None = None,
        random_state: int | None = 0,
        class_weight: Literal["balanced"] | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...

class RidgeClassifier:
    alpha: float
    class_weight: Literal["balanced"] | None

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        fit_intercept: bool = True,
        max_iter: int | None = None,
        tol: float = 1e-4,
        class_weight: Literal["balanced"] | None = None,
        random_state: int | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...

class ARDRegression:
    alpha_1: float
    threshold_lambda: float

    def __init__(
        self,
        *,
        max_iter: int = 300,
        tol: float = 1e-3,
        alpha_1: float = 1e-6,
        alpha_2: float = 1e-6,
        lambda_1: float = 1e-6,
        lambda_2: float = 1e-6,
        threshold_lambda: float = 1e4,
        fit_intercept: bool = True,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...

class BayesianRidge:
    alpha_1: float
    lambda_2: float

    def __init__(
        self,
        *,
        max_iter: int = 300,
        tol: float = 1e-3,
        alpha_1: float = 1e-6,
        alpha_2: float = 1e-6,
        lambda_1: float = 1e-6,
        lambda_2: float = 1e-6,
        fit_intercept: bool = True,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...

class GammaRegressor:
    alpha: float
    solver: str

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        solver: str = "lbfgs",
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...

class TweedieRegressor:
    power: float
    link: str
    solver: str

    def __init__(
        self,
        *,
        power: float = 0.0,
        link: str = "auto",
        alpha: float = 1.0,
        fit_intercept: bool = True,
        solver: str = "lbfgs",
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...

class Lars:
    n_nonzero_coefs: int

    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        n_nonzero_coefs: int = 500,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...

class LassoLars:
    alpha: float
    positive: bool

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        fit_intercept: bool = True,
        max_iter: int = 500,
        positive: bool = False,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...

class LassoLarsIC:
    criterion: str
    noise_variance: float | None

    def __init__(
        self,
        criterion: str = "aic",
        *,
        fit_intercept: bool = True,
        max_iter: int = 500,
        positive: bool = False,
        noise_variance: float | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...

class OrthogonalMatchingPursuit:
    n_nonzero_coefs: int | None
    tol: float | None

    def __init__(
        self,
        *,
        n_nonzero_coefs: int | None = None,
        tol: float | None = None,
        fit_intercept: bool = True,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...

class TheilSenRegressor:
    max_subpopulation: int
    n_subsamples: int | None

    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        max_subpopulation: int = 10000,
        n_subsamples: int | None = None,
        max_iter: int = 300,
        tol: float = 1e-3,
        random_state: int | None = None,
        n_jobs: int | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...
