from enum import Enum


class GeneralizedLinearSolver(Enum):
    """
    Solver for scikit-learn's generalized linear regressors.

    Attributes
    ----------
    LBFGS
        Limited-memory BFGS.
    NEWTON_CHOLESKY
        Newton's method with a Cholesky decomposition; suits few features.
    """

    LBFGS = "lbfgs"
    NEWTON_CHOLESKY = "newton-cholesky"
