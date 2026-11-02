from enum import Enum


class LinearSVCPenalty(Enum):
    """
    Regularization penalty for scikit-learn's LinearSVC.

    Attributes
    ----------
    L1
        L1 penalty, which gives sparse coefficients.
    L2
        L2 penalty.
    """

    L1 = "l1"
    L2 = "l2"
