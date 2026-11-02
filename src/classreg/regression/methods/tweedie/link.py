from enum import Enum


class TweedieLink(Enum):
    """
    Link function for scikit-learn's TweedieRegressor.

    Attributes
    ----------
    AUTO
        Identity for a power of 0 and logarithm otherwise.
    IDENTITY
        Identity link.
    LOG
        Logarithmic link, which keeps predictions positive.
    """

    AUTO = "auto"
    IDENTITY = "identity"
    LOG = "log"
