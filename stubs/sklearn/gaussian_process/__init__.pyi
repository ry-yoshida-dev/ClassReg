import numpy as np
from numpy.typing import ArrayLike, NDArray

from .kernels import Kernel

class GaussianProcessRegressor:
    kernel: Kernel | None
    normalize_y: bool

    def __init__(
        self,
        kernel: Kernel | None = None,
        *,
        alpha: float = 1e-10,
        n_restarts_optimizer: int = 0,
        normalize_y: bool = False,
        random_state: int | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...

class GaussianProcessClassifier:
    kernel: Kernel | None
    max_iter_predict: int

    def __init__(
        self,
        kernel: Kernel | None = None,
        *,
        n_restarts_optimizer: int = 0,
        max_iter_predict: int = 100,
        random_state: int | None = None,
        n_jobs: int | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...
