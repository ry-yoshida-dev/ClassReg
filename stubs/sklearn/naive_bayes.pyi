from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

class GaussianNB:
    var_smoothing: float

    def __init__(self, *, var_smoothing: float = 1e-9) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...
