from enum import Enum


class StochasticGradientDescentLoss(Enum):
    """
    Loss function for scikit-learn's SGDClassifier.

    Attributes
    ----------
    HINGE
        Hinge loss, which gives a linear support vector machine.
    LOG_LOSS
        Logistic loss, which gives logistic regression.
    MODIFIED_HUBER
        Smoothed hinge loss, tolerant to outliers.
    SQUARED_HINGE
        Quadratically penalized hinge loss.
    PERCEPTRON
        Linear loss of the perceptron algorithm.
    """

    HINGE = "hinge"
    LOG_LOSS = "log_loss"
    MODIFIED_HUBER = "modified_huber"
    SQUARED_HINGE = "squared_hinge"
    PERCEPTRON = "perceptron"
