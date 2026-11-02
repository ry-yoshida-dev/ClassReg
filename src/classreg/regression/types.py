"""Type aliases for regression data."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..types import BoolArray, FloatArray, GroupArray, IndexArray, NumericArray

type TargetScalar = np.integer | np.floating
type TargetArray = NDArray[TargetScalar]

__all__ = [
    "BoolArray",
    "FloatArray",
    "GroupArray",
    "IndexArray",
    "NumericArray",
    "TargetArray",
    "TargetScalar",
]
