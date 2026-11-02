from enum import Enum


class StochasticGradientDescentLoss(Enum):
    """
    Loss function for scikit-learn's SGDRegressor.

    Attributes
    ----------
    SQUARED_ERROR
        Squared error.
    HUBER
        Squared error for small residuals and absolute error for large ones.
    EPSILON_INSENSITIVE
        Absolute error beyond the epsilon tube.
    SQUARED_EPSILON_INSENSITIVE
        Squared error beyond the epsilon tube.
    """

    SQUARED_ERROR = "squared_error"
    HUBER = "huber"
    EPSILON_INSENSITIVE = "epsilon_insensitive"
    SQUARED_EPSILON_INSENSITIVE = "squared_epsilon_insensitive"
