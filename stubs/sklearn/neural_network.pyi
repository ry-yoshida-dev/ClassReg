import numpy as np
from numpy.typing import ArrayLike, NDArray

class MLPClassifier:
    hidden_layer_sizes: tuple[int, ...]
    activation: str
    solver: str
    early_stopping: bool

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (100,),
        activation: str = "relu",
        *,
        solver: str = "adam",
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        max_iter: int = 200,
        random_state: int | None = None,
        tol: float = 1e-4,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...

class MLPRegressor:
    hidden_layer_sizes: tuple[int, ...]
    activation: str
    solver: str
    early_stopping: bool

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (100,),
        activation: str = "relu",
        *,
        solver: str = "adam",
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        max_iter: int = 200,
        random_state: int | None = None,
        tol: float = 1e-4,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...
