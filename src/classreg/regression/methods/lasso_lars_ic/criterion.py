from enum import Enum


class LassoLarsCriterion(Enum):
    """
    Information criterion for scikit-learn's LassoLarsIC.

    Attributes
    ----------
    AIC
        Akaike information criterion.
    BIC
        Bayesian information criterion, which favors sparser models.
    """

    AIC = "aic"
    BIC = "bic"
