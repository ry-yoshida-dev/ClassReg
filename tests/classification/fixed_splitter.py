from dataclasses import dataclass
from typing import override

import numpy as np
from numpy.typing import NDArray

from classreg.classification import DatasetSplitter, DataSplit
from classreg.classification.types import NumericArray


@dataclass(frozen=True)
class FixedSplitter(DatasetSplitter):
    """
    Splitter that returns predetermined splits.

    Attributes
    ----------
    splits : tuple[DataSplit, ...]
        The splits to return.
    """

    splits: tuple[DataSplit, ...]

    @override
    def split(self, X: NumericArray, y: NDArray[np.generic]) -> tuple[DataSplit, ...]:
        """
        Return the predetermined splits.

        Parameters
        ----------
        X : NumericArray
            The feature matrix, which is ignored.
        y : NDArray[np.generic]
            The targets, which are ignored.

        Returns
        -------
        tuple[DataSplit, ...]
            The predetermined splits.
        """
        return self.splits
