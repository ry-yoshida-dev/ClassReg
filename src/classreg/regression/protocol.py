"""Structural type for regressors used by RegressionProcessor."""

from __future__ import annotations

from typing import Protocol

from .types import FloatArray, NumericArray, TargetArray


class RegressorLike(Protocol):
    """
    Structural type for regressors.

    Any object with these methods can be wrapped, including custom scikit-learn
    compatible estimators.
    """

    def fit(self, X: NumericArray, y: TargetArray) -> object:
        """
        Fit the regressor.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.
        y : TargetArray
            The target values.

        Returns
        -------
        object
            The result of fitting, which is ignored.
        """
        ...

    def predict(self, X: NumericArray) -> FloatArray:
        """
        Predict target values.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.

        Returns
        -------
        FloatArray
            The predicted values with shape (n_samples,).
        """
        ...
