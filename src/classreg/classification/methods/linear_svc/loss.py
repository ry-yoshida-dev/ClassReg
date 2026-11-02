from enum import Enum


class LinearSVCLoss(Enum):
    """
    Loss function for scikit-learn's LinearSVC.

    Attributes
    ----------
    HINGE
        Hinge loss.
    SQUARED_HINGE
        Quadratically penalized hinge loss.
    """

    HINGE = "hinge"
    SQUARED_HINGE = "squared_hinge"
