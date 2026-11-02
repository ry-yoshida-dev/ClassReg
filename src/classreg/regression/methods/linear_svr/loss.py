from enum import Enum


class LinearSVRLoss(Enum):
    """
    Loss function for scikit-learn's LinearSVR.

    Attributes
    ----------
    EPSILON_INSENSITIVE
        Absolute error beyond the epsilon tube.
    SQUARED_EPSILON_INSENSITIVE
        Squared error beyond the epsilon tube.
    """

    EPSILON_INSENSITIVE = "epsilon_insensitive"
    SQUARED_EPSILON_INSENSITIVE = "squared_epsilon_insensitive"
