from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

class SVC:
    C: float
    kernel: str
    gamma: str | float
    max_iter: int

    def __init__(
        self,
        *,
        C: float = 1.0,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: str | float = "scale",
        coef0: float = 0.0,
        tol: float = 1e-3,
        class_weight: Literal["balanced"] | None = None,
        max_iter: int = -1,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...

class SVR:
    C: float
    epsilon: float
    kernel: str
    gamma: str | float
    max_iter: int

    def __init__(
        self,
        *,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: str | float = "scale",
        coef0: float = 0.0,
        tol: float = 1e-3,
        C: float = 1.0,
        epsilon: float = 0.1,
        max_iter: int = -1,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...

class LinearSVC:
    C: float
    penalty: str
    loss: str
    dual: str
    class_weight: Literal["balanced"] | None

    def __init__(
        self,
        penalty: str = "l2",
        loss: str = "squared_hinge",
        *,
        dual: str = "auto",
        tol: float = 1e-4,
        C: float = 1.0,
        fit_intercept: bool = True,
        class_weight: Literal["balanced"] | None = None,
        random_state: int | None = None,
        max_iter: int = 1000,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...

class LinearSVR:
    C: float
    epsilon: float
    loss: str
    dual: str

    def __init__(
        self,
        *,
        epsilon: float = 0.0,
        tol: float = 1e-4,
        C: float = 1.0,
        loss: str = "epsilon_insensitive",
        fit_intercept: bool = True,
        dual: str = "auto",
        random_state: int | None = None,
        max_iter: int = 1000,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...

class NuSVC:
    nu: float
    kernel: str
    class_weight: Literal["balanced"] | None

    def __init__(
        self,
        *,
        nu: float = 0.5,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: str | float = "scale",
        coef0: float = 0.0,
        tol: float = 1e-3,
        max_iter: int = -1,
        class_weight: Literal["balanced"] | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...

class NuSVR:
    nu: float
    C: float
    kernel: str

    def __init__(
        self,
        *,
        nu: float = 0.5,
        C: float = 1.0,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: str | float = "scale",
        coef0: float = 0.0,
        tol: float = 1e-3,
        max_iter: int = -1,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...
