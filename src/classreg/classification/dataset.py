from __future__ import annotations

from typing import override

import numpy as np

from ..dataset import SupervisedDataset
from .types import LabelArray, LabelScalar


class ClassificationDataset(SupervisedDataset[LabelScalar]):
    """
    Validated pair of feature matrix and class labels.

    Labels must be integers, booleans, or strings.
    """

    @property
    def classes(self) -> LabelArray:
        """
        Get the sorted unique class labels.

        Returns
        -------
        LabelArray
            The sorted unique class labels.
        """
        return np.unique(self.y)

    @override
    def _validate_target_type(self) -> None:
        is_discrete: bool = any(
            np.issubdtype(self.y.dtype, label_type)
            for label_type in (np.integer, np.bool_, np.str_)
        )
        if not is_discrete:
            raise TypeError(
                f"y must contain integer, boolean, or string labels, got dtype {self.y.dtype}"
            )
