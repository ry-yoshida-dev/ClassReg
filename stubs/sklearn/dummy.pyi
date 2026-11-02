from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

class DummyClassifier:
    strategy: str

    def __init__(
        self,
        *,
        strategy: str = "prior",
        random_state: int | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...

class DummyRegressor:
    strategy: str

    def __init__(self, *, strategy: str = "mean") -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...
