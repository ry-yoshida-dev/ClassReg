import numpy as np
from numpy.typing import ArrayLike, NDArray

class IsotonicRegression:
    y_min: float | None
    y_max: float | None
    increasing: bool | str
    out_of_bounds: str

    def __init__(
        self,
        *,
        y_min: float | None = None,
        y_max: float | None = None,
        increasing: bool | str = True,
        out_of_bounds: str = "nan",
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...
