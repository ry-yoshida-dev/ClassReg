"""Structural type for the processors that an evaluator trains."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ...types import NumericArray
from .method import MethodLike


class EvaluableProcessorLike[TargetT: np.generic, MetricsT](Protocol):
    """
    Structural type for a processor that can be fitted and measured.
    """

    @property
    def method(self) -> MethodLike:
        """
        Get the method this processor implements.

        Returns
        -------
        MethodLike
            The method.
        """
        ...

    def fit(self, X: NumericArray, y: NDArray[TargetT]) -> None:
        """
        Fit the processor.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.
        y : NDArray[TargetT]
            The targets.
        """
        ...

    def evaluate(self, X: NumericArray, y: NDArray[TargetT]) -> MetricsT:
        """
        Measure the fitted processor on labeled data.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.
        y : NDArray[TargetT]
            The true targets.

        Returns
        -------
        MetricsT
            The performance metrics.
        """
        ...
