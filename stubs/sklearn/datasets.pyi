import numpy as np
from numpy.typing import NDArray

class Bunch:
    data: NDArray[np.float64]
    target: NDArray[np.int64]

def load_breast_cancer() -> Bunch: ...
def load_diabetes() -> Bunch: ...
def load_iris() -> Bunch: ...
