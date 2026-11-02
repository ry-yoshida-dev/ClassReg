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
