from enum import Enum


class IsotonicOutOfBounds(Enum):
    """
    Handling of samples outside the training range of IsotonicRegression.

    Attributes
    ----------
    CLIP
        Use the prediction at the nearest training value.
    RAISE
        Raise ValueError during prediction.
    """

    CLIP = "clip"
    RAISE = "raise"
