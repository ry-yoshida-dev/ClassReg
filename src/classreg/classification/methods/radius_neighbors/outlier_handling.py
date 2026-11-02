from enum import Enum
from typing import Literal


class RadiusNeighborsOutlierHandling(Enum):
    """
    Handling of samples without neighbors for RadiusNeighborsClassifier.

    Attributes
    ----------
    RAISE
        Raise ValueError during prediction.
    MOST_FREQUENT
        Predict the most frequent training class.
    """

    RAISE = "raise"
    MOST_FREQUENT = "most_frequent"

    def to_estimator_argument(self) -> Literal["most_frequent"] | None:
        """
        Convert to the scikit-learn ``outlier_label`` argument.

        Returns
        -------
        Literal["most_frequent"] | None
            The argument value.
        """
        match self:
            case RadiusNeighborsOutlierHandling.RAISE:
                return None
            case RadiusNeighborsOutlierHandling.MOST_FREQUENT:
                return "most_frequent"
