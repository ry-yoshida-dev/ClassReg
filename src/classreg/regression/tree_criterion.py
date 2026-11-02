from enum import Enum


class TreeCriterion(Enum):
    """
    Split quality criterion, shared by decision trees and tree ensembles.

    Attributes
    ----------
    SQUARED_ERROR
        Mean squared error.
    ABSOLUTE_ERROR
        Mean absolute error.
    POISSON
        Poisson deviance; requires non-negative targets.
    """

    SQUARED_ERROR = "squared_error"
    ABSOLUTE_ERROR = "absolute_error"
    POISSON = "poisson"
