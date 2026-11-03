from dataclasses import dataclass
from typing import override

from data_scaler import FeatureScaling
from sklearn.linear_model import LassoLarsIC

from ....parameter_validator import ParameterValidator
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor
from .criterion import LassoLarsCriterion


@dataclass(frozen=True)
class LassoLarsInformationCriterionParameters(RegressionParameters):
    """
    Parameters for scikit-learn's LassoLarsIC.

    The penalty is chosen from training data by an information criterion, without
    cross-validation. Unless ``noise_variance`` is given, fitting raises ValueError
    when samples do not outnumber features.

    Attributes
    ----------
    criterion : LassoLarsCriterion
        The information criterion.
    max_iter : int
        Maximum number of iterations.
    is_positive : bool
        Whether coefficients are forced to be non-negative.
    noise_variance : float | None
        Noise variance; estimated from an ordinary least squares fit when None.
    is_intercept_fitted : bool
        Whether to fit an intercept.
    scaling : FeatureScaling
        Feature scaling; the penalty depends on feature scale.
    """

    criterion: LassoLarsCriterion = LassoLarsCriterion.AIC
    max_iter: int = 500
    is_positive: bool = False
    noise_variance: float | None = None
    is_intercept_fitted: bool = True
    scaling: FeatureScaling = FeatureScaling.STANDARD

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        ParameterValidator.require_integers(
            max_iter=self.max_iter,
        )
        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")
        if self.noise_variance is not None:
            ParameterValidator.require_not_nan(noise_variance=self.noise_variance)
            if self.noise_variance <= 0:
                raise ValueError("noise_variance must be greater than 0 when set")

    @override
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted lasso least angle regression processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator: LassoLarsIC = LassoLarsIC(
            criterion=self.criterion.value,
            max_iter=self.max_iter,
            positive=self.is_positive,
            noise_variance=self.noise_variance,
            fit_intercept=self.is_intercept_fitted,
        )
        return RegressionProcessor(
            method=RegressionMethod.LASSO_LARS_INFORMATION_CRITERION,
            estimator=self.scaling.build_pipeline(estimator),
        )
