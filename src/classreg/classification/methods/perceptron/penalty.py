from enum import Enum


class PerceptronPenalty(Enum):
    """
    Regularization penalty for scikit-learn's Perceptron.

    Attributes
    ----------
    L1
        L1 penalty, which gives sparse coefficients.
    L2
        L2 penalty.
    ELASTIC_NET
        Combined L1 and L2 penalty.
    """

    L1 = "l1"
    L2 = "l2"
    ELASTIC_NET = "elasticnet"
