from enum import Enum


class LinearDiscriminantAnalysisSolver(Enum):
    """
    Solver for scikit-learn's LinearDiscriminantAnalysis.

    Attributes
    ----------
    SVD
        Singular value decomposition; does not support shrinkage.
    LSQR
        Least squares.
    EIGEN
        Eigenvalue decomposition.
    """

    SVD = "svd"
    LSQR = "lsqr"
    EIGEN = "eigen"
