from __future__ import annotations

import math
from numbers import Integral

import numpy as np

from .types import GroupArray


class ParameterValidator:
    """
    Runtime checks that range comparisons cannot express.

    NaN compares false to everything, and non-integers pass integer range
    checks, so both would otherwise only fail once an estimator is fitted.
    """

    @staticmethod
    def require_integers(**values: object) -> None:
        """
        Check that values are integers, or None for optional parameters.

        Parameters
        ----------
        **values : object
            The values by parameter name.

        Raises
        ------
        TypeError
            If a value is neither None nor an integer; booleans are rejected.
        """
        for name, value in values.items():
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer, got {value!r}")

    @staticmethod
    def require_not_nan(**values: float) -> None:
        """
        Check that values are not NaN.

        Parameters
        ----------
        **values : float
            The values by parameter name.

        Raises
        ------
        ValueError
            If a value is NaN.
        """
        for name, value in values.items():
            if math.isnan(value):
                raise ValueError(f"{name} must not be NaN")

    @staticmethod
    def require_group_identifiers(groups: GroupArray) -> None:
        """
        Check that group identifiers are discrete.

        Parameters
        ----------
        groups : GroupArray
            The group identifier of each sample.

        Raises
        ------
        TypeError
            If groups does not contain integer or string identifiers.
        """
        is_discrete: bool = any(
            np.issubdtype(groups.dtype, group_type)
            for group_type in (np.integer, np.str_)
        )
        if not is_discrete:
            raise TypeError(
                f"groups must contain integer or string identifiers, got dtype {groups.dtype}"
            )
