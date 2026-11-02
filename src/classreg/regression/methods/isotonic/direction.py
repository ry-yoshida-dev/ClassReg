from enum import Enum


class IsotonicDirection(Enum):
    """
    Monotonic direction for scikit-learn's IsotonicRegression.

    Attributes
    ----------
    INCREASING
        Predictions never decrease as the feature grows.
    DECREASING
        Predictions never increase as the feature grows.
    AUTO
        Chosen from the sign of the Spearman correlation.
    """

    INCREASING = True
    DECREASING = False
    AUTO = "auto"
