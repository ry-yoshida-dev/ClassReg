"""Structural type for the metric enums of classification and regression."""

from __future__ import annotations

from typing import Protocol


class MetricLike[MetricsT](Protocol):
    """
    Structural type for a scalar metric readable from a metrics object.

    Tying a metric to its metrics type makes the type checker reject, for
    example, a regression metric passed to a classification report.
    """

    @property
    def value(self) -> str:
        """
        Get the name of the metric.

        Returns
        -------
        str
            The name of the metric.
        """
        ...

    @property
    def is_higher_better(self) -> bool:
        """
        Check the direction in which this metric improves.

        Returns
        -------
        bool
            True if larger values indicate better performance.
        """
        ...

    def read_from(self, metrics: MetricsT) -> float | None:
        """
        Read the value of this metric.

        Parameters
        ----------
        metrics : MetricsT
            The metrics to read from.

        Returns
        -------
        float | None
            The value, or None when the metric is undefined for this evaluation.
        """
        ...
