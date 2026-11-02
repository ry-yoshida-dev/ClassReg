import numpy as np
from numpy.typing import ArrayLike, NDArray

class LinearDiscriminantAnalysis:
    solver: str
    shrinkage: float | None

    def __init__(
        self,
        solver: str = "svd",
        shrinkage: float | None = None,
        tol: float = 1e-4,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...

class QuadraticDiscriminantAnalysis:
    reg_param: float

    def __init__(
        self,
        *,
        reg_param: float = 0.0,
        tol: float = 1e-4,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...
