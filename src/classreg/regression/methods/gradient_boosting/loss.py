from enum import Enum


class GradientBoostingLoss(Enum):
    """
    Loss function for scikit-learn's GradientBoostingRegressor.

    Attributes
    ----------
    SQUARED_ERROR
        Squared error.
    ABSOLUTE_ERROR
        Absolute error, robust to outliers.
    HUBER
        Squared error for small residuals and absolute error for large ones.
    QUANTILE
        Pinball loss, which targets a quantile.
    """

    SQUARED_ERROR = "squared_error"
    ABSOLUTE_ERROR = "absolute_error"
    HUBER = "huber"
    QUANTILE = "quantile"
