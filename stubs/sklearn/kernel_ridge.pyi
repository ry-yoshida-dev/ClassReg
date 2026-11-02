import numpy as np
from numpy.typing import ArrayLike, NDArray

class KernelRidge:
    alpha: float
    kernel: str
    gamma: float | None

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        kernel: str = "linear",
        gamma: float | None = None,
        degree: int = 3,
        coef0: float = 1.0,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...
