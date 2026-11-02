from __future__ import annotations

from typing import override

import numpy as np

from ..dataset import SupervisedDataset
from .types import TargetScalar


class RegressionDataset(SupervisedDataset[TargetScalar]):
    """
    Validated pair of feature matrix and target values.

    Targets must be finite numbers.
    """

    @override
    def _validate_target_type(self) -> None:
        if not np.issubdtype(self.y.dtype, np.integer) and not np.issubdtype(
            self.y.dtype, np.floating
        ):
            raise TypeError(f"y must be numeric, got dtype {self.y.dtype}")

    @override
    def _validate_target_values(self) -> None:
        if not np.all(np.isfinite(self.y)):
            raise ValueError("y must not contain NaN or infinite values")
