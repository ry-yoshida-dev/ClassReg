from enum import Enum


class MultilayerPerceptronActivation(Enum):
    """
    Hidden layer activation of scikit-learn's MLPClassifier and MLPRegressor.

    Attributes
    ----------
    IDENTITY
        No transformation.
    LOGISTIC
        Logistic sigmoid.
    TANH
        Hyperbolic tangent.
    RELU
        Rectified linear unit.
    """

    IDENTITY = "identity"
    LOGISTIC = "logistic"
    TANH = "tanh"
    RELU = "relu"
