from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

class RandomForestClassifier:
    n_estimators: int
    max_features: str | None
    bootstrap: bool

    def __init__(
        self,
        n_estimators: int = 100,
        *,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | None = "sqrt",
        bootstrap: bool = True,
        n_jobs: int | None = None,
        random_state: int | None = None,
        class_weight: Literal["balanced"] | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...

class HistGradientBoostingClassifier:
    learning_rate: float
    early_stopping: bool

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        max_iter: int = 100,
        max_leaf_nodes: int | None = 31,
        max_depth: int | None = None,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        max_bins: int = 255,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
        tol: float = 1e-7,
        random_state: int | None = None,
        class_weight: Literal["balanced"] | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...

class RandomForestRegressor:
    n_estimators: int
    max_features: str | None
    bootstrap: bool

    def __init__(
        self,
        n_estimators: int = 100,
        *,
        criterion: str = "squared_error",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | None = None,
        bootstrap: bool = True,
        n_jobs: int | None = None,
        random_state: int | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...

class HistGradientBoostingRegressor:
    loss: str
    learning_rate: float
    early_stopping: bool

    def __init__(
        self,
        loss: str = "squared_error",
        *,
        learning_rate: float = 0.1,
        max_iter: int = 100,
        max_leaf_nodes: int | None = 31,
        max_depth: int | None = None,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        max_bins: int = 255,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
        tol: float = 1e-7,
        random_state: int | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...

class ExtraTreesClassifier:
    n_estimators: int
    max_features: str | None
    bootstrap: bool

    def __init__(
        self,
        n_estimators: int = 100,
        *,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | None = "sqrt",
        bootstrap: bool = False,
        n_jobs: int | None = None,
        random_state: int | None = None,
        class_weight: Literal["balanced"] | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...

class ExtraTreesRegressor:
    n_estimators: int
    max_features: str | None
    bootstrap: bool

    def __init__(
        self,
        n_estimators: int = 100,
        *,
        criterion: str = "squared_error",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | None = None,
        bootstrap: bool = False,
        n_jobs: int | None = None,
        random_state: int | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...

class GradientBoostingClassifier:
    n_estimators: int
    subsample: float
    n_iter_no_change: int | None

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_depth: int | None = 3,
        max_features: str | None = None,
        validation_fraction: float = 0.1,
        n_iter_no_change: int | None = None,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...

class GradientBoostingRegressor:
    loss: str
    alpha: float
    n_estimators: int
    subsample: float
    n_iter_no_change: int | None

    def __init__(
        self,
        *,
        loss: str = "squared_error",
        alpha: float = 0.9,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_depth: int | None = 3,
        max_features: str | None = None,
        validation_fraction: float = 0.1,
        n_iter_no_change: int | None = None,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...

class AdaBoostClassifier:
    estimator: object
    n_estimators: int
    learning_rate: float

    def __init__(
        self,
        estimator: object | None = None,
        *,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        random_state: int | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...

class AdaBoostRegressor:
    estimator: object
    n_estimators: int
    learning_rate: float
    loss: str

    def __init__(
        self,
        estimator: object | None = None,
        *,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        loss: str = "linear",
        random_state: int | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...

class BaggingClassifier:
    estimator: object
    n_estimators: int
    bootstrap_features: bool

    def __init__(
        self,
        estimator: object | None = None,
        n_estimators: int = 10,
        *,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        n_jobs: int | None = None,
        random_state: int | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...

class BaggingRegressor:
    estimator: object
    n_estimators: int
    bootstrap_features: bool

    def __init__(
        self,
        estimator: object | None = None,
        n_estimators: int = 10,
        *,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        n_jobs: int | None = None,
        random_state: int | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...

class StackingClassifier:
    estimators: list[tuple[str, object]]
    final_estimator: object | None
    cv: int | None
    passthrough: bool

    def __init__(
        self,
        estimators: list[tuple[str, object]],
        final_estimator: object | None = None,
        *,
        cv: int | None = None,
        n_jobs: int | None = None,
        passthrough: bool = False,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...

class StackingRegressor:
    estimators: list[tuple[str, object]]
    final_estimator: object | None
    cv: int | None
    passthrough: bool

    def __init__(
        self,
        estimators: list[tuple[str, object]],
        final_estimator: object | None = None,
        *,
        cv: int | None = None,
        n_jobs: int | None = None,
        passthrough: bool = False,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...

class VotingClassifier:
    estimators: list[tuple[str, object]]
    weights: list[float] | None
    voting: str

    def __init__(
        self,
        estimators: list[tuple[str, object]],
        *,
        voting: str = "hard",
        weights: list[float] | None = None,
        n_jobs: int | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer | np.bool_ | np.str_]: ...

class VotingRegressor:
    estimators: list[tuple[str, object]]
    weights: list[float] | None

    def __init__(
        self,
        estimators: list[tuple[str, object]],
        *,
        weights: list[float] | None = None,
        n_jobs: int | None = None,
    ) -> None: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...
