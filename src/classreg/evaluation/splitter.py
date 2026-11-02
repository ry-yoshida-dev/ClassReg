from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from ..types import NumericArray
from .data_split import DataSplit


class DatasetSplitter(ABC):
    """
    Base class for strategies that partition a dataset for evaluation.
    """

    @abstractmethod
    def split(self, X: NumericArray, y: NDArray[np.generic]) -> tuple[DataSplit, ...]:
        """
        Partition a dataset into train/test splits.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.
        y : NDArray[np.generic]
            The targets.

        Returns
        -------
        tuple[DataSplit, ...]
            One split per evaluation fold.
        """
