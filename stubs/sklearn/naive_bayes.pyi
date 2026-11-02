from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

class GaussianNB:
    var_smoothing: float

    def __init__(self, *, var_smoothing: float = 1e-9) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...

class BernoulliNB:
    alpha: float
    binarize: float | None

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        binarize: float | None = 0.0,
        fit_prior: bool = True,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...

class MultinomialNB:
    alpha: float

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        fit_prior: bool = True,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...

class ComplementNB:
    alpha: float
    norm: bool

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        fit_prior: bool = True,
        norm: bool = False,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...

class CategoricalNB:
    alpha: float
    min_categories: int | None

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        fit_prior: bool = True,
        min_categories: int | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...
