import numpy as np
from numpy.typing import ArrayLike, NDArray

class TransformedTargetRegressor:
    regressor: object
    transformer: object

    def __init__(
        self,
        regressor: object | None = None,
        *,
        transformer: object | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...
