from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

class DecisionTreeClassifier:
    criterion: str
    max_depth: int | None

    def __init__(
        self,
        *,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int | None = None,
        max_leaf_nodes: int | None = None,
        class_weight: Literal["balanced"] | None = None,
        ccp_alpha: float = 0.0,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...

class DecisionTreeRegressor:
    criterion: str
    max_depth: int | None

    def __init__(
        self,
        *,
        criterion: str = "squared_error",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int | None = None,
        max_leaf_nodes: int | None = None,
        ccp_alpha: float = 0.0,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...

class ExtraTreeClassifier:
    criterion: str
    max_features: str | None

    def __init__(
        self,
        *,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | None = None,
        random_state: int | None = None,
        max_leaf_nodes: int | None = None,
        ccp_alpha: float = 0.0,
        class_weight: Literal["balanced"] | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...

class ExtraTreeRegressor:
    criterion: str
    max_features: str | None

    def __init__(
        self,
        *,
        criterion: str = "squared_error",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | None = None,
        random_state: int | None = None,
        max_leaf_nodes: int | None = None,
        ccp_alpha: float = 0.0,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...
