from dataclasses import dataclass
from typing import override

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import GroupKFold

from ...parameter_validator import ParameterValidator
from ...random_state import DEFAULT_RANDOM_STATE
from ...types import GroupArray, NumericArray
from ..data_split import DataSplit
from ..splitter import DatasetSplitter


@dataclass(frozen=True, eq=False)
class GroupKFoldSplitter(DatasetSplitter):
    """
    K-fold cross-validation that never splits a group across train and test.

    Use this when samples are correlated within groups, such as several
    measurements of one subject; random splitting would otherwise leak
    information.

    Attributes
    ----------
    groups : GroupArray
        Group identifier of each sample, with shape (n_samples,).
    n_splits : int
        Number of folds.
    is_shuffled : bool
        Whether groups are shuffled before folding.
    random_state : int | None
        Seed for the shuffle; ignored when is_shuffled is False.
    """

    groups: GroupArray
    n_splits: int = 5
    is_shuffled: bool = True
    random_state: int | None = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", np.asarray(self.groups))
        is_discrete: bool = any(
            np.issubdtype(self.groups.dtype, group_type)
            for group_type in (np.integer, np.str_)
        )
        if not is_discrete:
            raise TypeError(
                f"groups must contain integer or string identifiers, got dtype {self.groups.dtype}"
            )
        ParameterValidator.require_integers(
            n_splits=self.n_splits,
            random_state=self.random_state,
        )
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")

    @override
    def split(self, X: NumericArray, y: NDArray[np.generic]) -> tuple[DataSplit, ...]:
        """
        Partition a dataset into group-preserving folds.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.
        y : NDArray[np.generic]
            The targets, which are ignored.

        Returns
        -------
        tuple[DataSplit, ...]
            One split per fold.

        Raises
        ------
        ValueError
            If groups does not have one entry per sample.
        """
        if self.groups.shape != (len(X),):
            raise ValueError(
                f"groups must have shape ({len(X)},), got {self.groups.shape}"
            )
        splitter: GroupKFold = GroupKFold(
            n_splits=self.n_splits,
            shuffle=self.is_shuffled,
            random_state=self.random_state if self.is_shuffled else None,
        )
        return tuple(
            DataSplit(train_indices=train, test_indices=test)
            for train, test in splitter.split(X, y, self.groups)
        )
