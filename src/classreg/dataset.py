from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

from .types import IndexArray, NumericArray


@dataclass(frozen=True, eq=False)
class SupervisedDataset[TargetT: np.generic](ABC):
    """
    Validated pair of feature matrix and targets.

    Subclasses define which target types are valid.

    Attributes
    ----------
    X : NumericArray
        Feature matrix with shape (n_samples, n_features).
    y : NDArray[TargetT]
        Targets with shape (n_samples,).

    Raises
    ------
    TypeError
        If the features or targets have an unsupported type.
    ValueError
        If the shapes are invalid or inconsistent, or a target is invalid.
    """

    X: NumericArray
    y: NDArray[TargetT]

    def __post_init__(self) -> None:
        object.__setattr__(self, "X", np.asarray(self.X))
        object.__setattr__(self, "y", np.asarray(self.y))
        self._validate_features()
        self._validate_target_type()
        self._validate_target_shape()
        self._validate_target_values()

    @property
    def num_samples(self) -> int:
        """
        Get the number of samples.

        Returns
        -------
        int
            The number of samples.
        """
        return len(self.X)

    def subset(self, indices: IndexArray) -> Self:
        """
        Select the samples at the given indices.

        Parameters
        ----------
        indices : IndexArray
            The sample indices to select.

        Returns
        -------
        Self
            The selected samples.
        """
        return type(self)(X=self.X[indices], y=self.y[indices])

    def _validate_features(self) -> None:
        if not np.issubdtype(self.X.dtype, np.integer) and not np.issubdtype(
            self.X.dtype, np.floating
        ):
            raise TypeError(f"X must be numeric, got dtype {self.X.dtype}")
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2-D, got {self.X.ndim} dimensions")
        if self.X.shape[0] == 0 or self.X.shape[1] == 0:
            raise ValueError(f"X must not be empty, got shape {self.X.shape}")

    def _validate_target_shape(self) -> None:
        if self.y.ndim != 1:
            raise ValueError(f"y must be 1-D, got {self.y.ndim} dimensions")
        if self.y.shape[0] != self.X.shape[0]:
            raise ValueError(
                f"X and y must have the same length, got {self.X.shape[0]} and {self.y.shape[0]}"
            )

    @abstractmethod
    def _validate_target_type(self) -> None: ...

    def _validate_target_values(self) -> None:
        return None
