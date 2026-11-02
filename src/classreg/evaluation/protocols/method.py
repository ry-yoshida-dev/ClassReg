"""Structural type for the method enums of classification and regression."""

from __future__ import annotations

from typing import Protocol


class MethodLike(Protocol):
    """
    Structural type for an enum member that names a learning method.
    """

    @property
    def value(self) -> str:
        """
        Get the name of the method.

        Returns
        -------
        str
            The name of the method.
        """
        ...
