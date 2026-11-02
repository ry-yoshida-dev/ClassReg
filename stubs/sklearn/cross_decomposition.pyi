import numpy as np
from numpy.typing import ArrayLike, NDArray

class PLSRegression:
    n_components: int
    scale: bool

    def __init__(
        self,
        n_components: int = 2,
        *,
        scale: bool = True,
        max_iter: int = 500,
        tol: float = 1e-6,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...
