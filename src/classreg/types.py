"""Type aliases shared by classification and regression."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

type NumericArray = NDArray[np.integer | np.floating]
type FloatArray = NDArray[np.floating]
type BoolArray = NDArray[np.bool_]
type IndexArray = NDArray[np.intp]
type GroupArray = NDArray[np.integer | np.str_]
