from dataclasses import dataclass
from typing import override

import numpy as np
from numpy.typing import NDArray

from classreg.classification import DatasetSplitter, DataSplit


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
    def split(
        self, X: NDArray[np.generic], y: NDArray[np.generic]
    ) -> tuple[DataSplit, ...]:
        """
        Return the predetermined splits.

        Parameters
        ----------
        X : NDArray[np.generic]
            The samples, which are ignored.
        y : NDArray[np.generic]
            The targets, which are ignored.

        Returns
        -------
        tuple[DataSplit, ...]
            The predetermined splits.
        """
        return self.splits
