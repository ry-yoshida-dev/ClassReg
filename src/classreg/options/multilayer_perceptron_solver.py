from enum import Enum


class MultilayerPerceptronSolver(Enum):
    """
    Weight optimizer of scikit-learn's MLPClassifier and MLPRegressor.

    Attributes
    ----------
    LBFGS
        Limited-memory BFGS, which suits small datasets.
    SGD
        Stochastic gradient descent.
    ADAM
        Adam, a stochastic optimizer with adaptive step sizes.
    """

    LBFGS = "lbfgs"
    SGD = "sgd"
    ADAM = "adam"
