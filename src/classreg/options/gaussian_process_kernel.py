from __future__ import annotations

from enum import Enum

from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    Kernel,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)


class GaussianProcessKernel(Enum):
    """
    Covariance shape of a Gaussian process.

    Attributes
    ----------
    RBF
        Radial basis function, for very smooth functions.
    MATERN
        Matern kernel with smoothness 2.5, for moderately smooth functions.
    RATIONAL_QUADRATIC
        Mixture of radial basis functions over many length scales.
    """

    RBF = "rbf"
    MATERN = "matern"
    RATIONAL_QUADRATIC = "rational_quadratic"

    def build(self) -> Kernel:
        """
        Build the shape multiplied by a fitted signal variance plus fitted noise.

        Scale and noise then do not have to be chosen by hand for regression.

        Returns
        -------
        Kernel
            The unfitted kernel.
        """
        return ConstantKernel() * self.build_shape() + WhiteKernel()

    def build_shape(self) -> Kernel:
        """
        Build the covariance shape alone.

        Returns
        -------
        Kernel
            The unfitted kernel.
        """
        match self:
            case GaussianProcessKernel.RBF:
                return RBF()
            case GaussianProcessKernel.MATERN:
                return Matern(nu=2.5)
            case GaussianProcessKernel.RATIONAL_QUADRATIC:
                return RationalQuadratic()
