from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .values import MetricValues


class RegressionMetric(Enum):
    """
    Scalar performance metric of a regressor.

    Attributes
    ----------
    MAE
        Mean absolute error.
    MSE
        Mean squared error.
    RMSE
        Root mean squared error.
    MEDIAN_ABSOLUTE_ERROR
        Median absolute error, robust to outliers.
    MAX_ERROR
        Largest absolute error.
    R2
        Coefficient of determination; undefined for constant targets.
    EXPLAINED_VARIANCE
        Explained variance score, which ignores systematic bias; undefined for
        constant targets.
    MAPE
        Mean absolute percentage error as a fraction; undefined when a target is
        zero.
    """

    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    MEDIAN_ABSOLUTE_ERROR = "median_absolute_error"
    MAX_ERROR = "max_error"
    R2 = "r2"
    EXPLAINED_VARIANCE = "explained_variance"
    MAPE = "mape"

    @property
    def is_higher_better(self) -> bool:
        """
        Check the direction in which this metric improves.

        Returns
        -------
        bool
            True if larger values indicate better performance.
        """
        match self:
            case RegressionMetric.R2 | RegressionMetric.EXPLAINED_VARIANCE:
                return True
            case _:
                return False

    def read_from(self, metrics: MetricValues) -> float | None:
        """
        Read the value of this metric.

        Parameters
        ----------
        metrics : MetricValues
            The metrics to read from.

        Returns
        -------
        float | None
            The value, or None when the metric is undefined for this evaluation.
        """
        match self:
            case RegressionMetric.MAE:
                return metrics.mae
            case RegressionMetric.MSE:
                return metrics.mse
            case RegressionMetric.RMSE:
                return metrics.rmse
            case RegressionMetric.MEDIAN_ABSOLUTE_ERROR:
                return metrics.median_absolute_error
            case RegressionMetric.MAX_ERROR:
                return metrics.max_error
            case RegressionMetric.R2:
                return metrics.r2
            case RegressionMetric.EXPLAINED_VARIANCE:
                return metrics.explained_variance
            case RegressionMetric.MAPE:
                return metrics.mape
