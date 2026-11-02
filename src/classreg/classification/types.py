"""Type aliases for classification data."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..types import BoolArray, FloatArray, GroupArray, IndexArray, NumericArray

type LabelScalar = np.integer | np.bool_ | np.str_
type LabelArray = NDArray[LabelScalar]
type CountArray = NDArray[np.int64]
type ClassLabel = int | str

__all__ = [
    "BoolArray",
    "ClassLabel",
    "CountArray",
    "FloatArray",
    "GroupArray",
    "IndexArray",
    "LabelArray",
    "LabelScalar",
    "NumericArray",
]
