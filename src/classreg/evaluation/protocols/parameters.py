"""Structural type for the parameter classes that an evaluator consumes."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .processor import EvaluableProcessorLike


class EvaluableParametersLike[TargetT: np.generic, MetricsT](Protocol):
    """
    Structural type for parameters that build a fresh processor on each call.
    """

    def build_processor(self) -> EvaluableProcessorLike[TargetT, MetricsT]:
        """
        Build an unfitted processor.

        Returns
        -------
        EvaluableProcessorLike[TargetT, MetricsT]
            The processor.
        """
        ...
