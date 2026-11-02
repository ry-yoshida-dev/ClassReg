from enum import Enum


class AdaBoostLoss(Enum):
    """
    Loss used to update sample weights in scikit-learn's AdaBoostRegressor.

    Attributes
    ----------
    LINEAR
        Linear in the relative error.
    SQUARE
        Quadratic in the relative error.
    EXPONENTIAL
        Exponential in the relative error.
    """

    LINEAR = "linear"
    SQUARE = "square"
    EXPONENTIAL = "exponential"
