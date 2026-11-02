from enum import Enum


class SVMKernel(Enum):
    """
    Kernel function for scikit-learn's SVC and SVR.

    Attributes
    ----------
    LINEAR
        Linear kernel.
    POLYNOMIAL
        Polynomial kernel.
    RBF
        Radial basis function kernel.
    SIGMOID
        Sigmoid kernel.
    """

    LINEAR = "linear"
    POLYNOMIAL = "poly"
    RBF = "rbf"
    SIGMOID = "sigmoid"
