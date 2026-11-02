from dataclasses import dataclass
from typing import override

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import LeaveOneOut

from ...types import NumericArray
from ..data_split import DataSplit
from ..splitter import DatasetSplitter


@dataclass(frozen=True)
class LeaveOneOutSplitter(DatasetSplitter):
    """
    Cross-validation that tests on each sample in turn.

    Each test partition has one sample, so metrics that need several samples or
    classes, such as ``r2`` or ``roc_auc``, are undefined in every fold. With
    balanced classes, a class prior baseline scores close to zero, because leaving
    a sample out makes another class the majority of the training samples, so
    improvements over that baseline are overstated.
    """

    @override
    def split(self, X: NumericArray, y: NDArray[np.generic]) -> tuple[DataSplit, ...]:
        """
        Partition a dataset into one fold per sample.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.
        y : NDArray[np.generic]
            The targets, which are ignored.

        Returns
        -------
        tuple[DataSplit, ...]
            One split per sample.

        Raises
        ------
        ValueError
            If the dataset has fewer than two samples.
        """
        return tuple(
            DataSplit(train_indices=train, test_indices=test)
            for train, test in LeaveOneOut().split(X)
        )
