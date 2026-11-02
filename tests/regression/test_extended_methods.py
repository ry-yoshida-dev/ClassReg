from __future__ import annotations

import math
import unittest
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import load_diabetes
from sklearn.ensemble import BaggingRegressor, StackingRegressor, VotingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from classreg import KFoldSplitter
from classreg.regression import (
    AutomaticRelevanceDeterminationParameters,
    BaggingParameters,
    BayesianRidgeParameters,
    DecisionTreeParameters,
    ExtraTreeParameters,
    GammaParameters,
    GeneralizedLinearSolver,
    IsotonicDirection,
    IsotonicOutOfBounds,
    IsotonicParameters,
    KernelRidgeParameters,
    LarsParameters,
    LassoLarsCriterion,
    LassoLarsInformationCriterionParameters,
    LassoLarsParameters,
    MetricValues,
    NuSVRParameters,
    OrthogonalMatchingPursuitParameters,
    PartialLeastSquaresParameters,
    RadiusNeighborsParameters,
    RegressionEvaluator,
    RegressionParameters,
    RegressionProcessor,
    RidgeParameters,
    StackingParameters,
    SVMKernel,
    SVRParameters,
    TheilSenParameters,
    TweedieLink,
    TweedieParameters,
    VotingParameters,
)
from classreg.regression.types import NumericArray
from tests.regression.helpers import linear_data

LINEAR_SIGNAL_PARAMETERS: tuple[RegressionParameters, ...] = (
    AutomaticRelevanceDeterminationParameters(),
    BayesianRidgeParameters(),
    ExtraTreeParameters(),
    TweedieParameters(alpha=0.0),
    KernelRidgeParameters(),
    LarsParameters(),
    LassoLarsParameters(alpha=0.01),
    LassoLarsInformationCriterionParameters(),
    NuSVRParameters(C=10.0),
    OrthogonalMatchingPursuitParameters(n_nonzero_coefs=3),
    PartialLeastSquaresParameters(n_components=3),
    RadiusNeighborsParameters(radius=2.0),
    TheilSenParameters(),
    BaggingParameters(estimator=RidgeParameters()),
    StackingParameters(estimators=(RidgeParameters(), SVRParameters(C=10.0))),
    VotingParameters(estimators=(RidgeParameters(), SVRParameters(C=10.0))),
)


def final_estimator(processor: RegressionProcessor) -> object:
    """Return the estimator step of the pipeline inside a processor."""
    pipeline: object = processor.estimator
    assert isinstance(pipeline, Pipeline)
    return pipeline.named_steps["estimator"]


class TestExtendedMethods(unittest.TestCase):
    def test_learns_linear_signal(self) -> None:
        X, y = linear_data()
        for parameters in LINEAR_SIGNAL_PARAMETERS:
            with self.subTest(parameters=type(parameters).__name__):
                processor: RegressionProcessor = parameters.build_processor()
                processor.fit(X, y)
                self.assertEqual(processor.predict(X).shape, y.shape)
                metrics: MetricValues = processor.evaluate(X, y)
                assert metrics.r2 is not None
                self.assertGreater(metrics.r2, 0.7)

    def test_positive_target_models(self) -> None:
        X, y = linear_data()
        positive: NDArray[np.float64] = np.exp(y / 4.0)
        candidates: tuple[RegressionParameters, ...] = (
            GammaParameters(alpha=0.0),
            GammaParameters(alpha=0.0, solver=GeneralizedLinearSolver.NEWTON_CHOLESKY),
            TweedieParameters(power=1.5, alpha=0.0),
            TweedieParameters(power=0.0, link=TweedieLink.LOG, alpha=0.0),
        )
        for parameters in candidates:
            with self.subTest(parameters=parameters):
                processor: RegressionProcessor = parameters.build_processor()
                processor.fit(X, positive)
                metrics: MetricValues = processor.evaluate(X, positive)
                assert metrics.r2 is not None
                self.assertGreater(metrics.r2, 0.9)

    def test_gamma_rejects_non_positive_targets(self) -> None:
        X, y = linear_data()
        processor: RegressionProcessor = GammaParameters().build_processor()
        with self.assertRaises(ValueError):
            processor.fit(X, y)

    def test_isotonic_clips_outside_the_training_range(self) -> None:
        X: NumericArray = np.arange(20.0).reshape(-1, 1)
        y: NDArray[np.float64] = 2.0 * np.arange(20.0)
        processor: RegressionProcessor = IsotonicParameters().build_processor()
        processor.fit(X, y)
        self.assertEqual(
            processor.predict(np.array([[-5.0], [100.0]])).tolist(), [0.0, 38.0]
        )
        self.assertEqual(processor.evaluate(X, y).r2, 1.0)

    def test_isotonic_out_of_bounds_can_raise(self) -> None:
        X: NumericArray = np.arange(20.0).reshape(-1, 1)
        y: NDArray[np.float64] = np.arange(20.0)
        processor: RegressionProcessor = IsotonicParameters(
            out_of_bounds=IsotonicOutOfBounds.RAISE
        ).build_processor()
        processor.fit(X, y)
        with self.assertRaises(ValueError):
            _ = processor.predict(np.array([[100.0]]))

    def test_isotonic_directions(self) -> None:
        X: NumericArray = np.arange(20.0).reshape(-1, 1)
        y: NDArray[np.float64] = -np.arange(20.0)
        for direction in (IsotonicDirection.DECREASING, IsotonicDirection.AUTO):
            with self.subTest(direction=direction):
                processor: RegressionProcessor = IsotonicParameters(
                    direction=direction
                ).build_processor()
                processor.fit(X, y)
                self.assertEqual(processor.evaluate(X, y).r2, 1.0)

    def test_isotonic_rejects_several_features(self) -> None:
        X, y = linear_data()
        processor: RegressionProcessor = IsotonicParameters().build_processor()
        with self.assertRaises(ValueError):
            processor.fit(X, y)

    def test_kernel_ridge_centering_compensates_for_the_missing_intercept(self) -> None:
        diabetes = load_diabetes()
        X: NumericArray = diabetes.data
        y: NDArray[np.float64] = diabetes.target.astype(np.float64)
        centered: RegressionProcessor = KernelRidgeParameters().build_processor()
        raw: RegressionProcessor = KernelRidgeParameters(
            is_target_centered=False
        ).build_processor()
        centered.fit(X, y)
        raw.fit(X, y)
        centered_r2: float | None = centered.evaluate(X, y).r2
        raw_r2: float | None = raw.evaluate(X, y).r2
        assert centered_r2 is not None and raw_r2 is not None
        self.assertGreater(centered_r2, 0.4)
        self.assertLess(raw_r2, 0.0)

    def test_radius_neighbors_without_neighbors_cannot_be_evaluated(self) -> None:
        X, y = linear_data()
        far: NumericArray = np.full((1, 3), 1000.0)
        processor: RegressionProcessor = RadiusNeighborsParameters().build_processor()
        processor.fit(X, y)
        with self.assertWarns(UserWarning):
            predictions: NDArray[np.floating] = processor.predict(far)
        self.assertTrue(np.isnan(predictions).all())
        with self.assertWarns(UserWarning), self.assertRaises(ValueError):
            _ = processor.evaluate(far, np.array([1.0]))

    def test_lasso_lars_information_criterion_needs_enough_samples(self) -> None:
        X, y = linear_data()
        processor: RegressionProcessor = (
            LassoLarsInformationCriterionParameters().build_processor()
        )
        with self.assertRaises(ValueError):
            processor.fit(X[:4], y[:4])

    def test_ensembles_evaluate(self) -> None:
        X, y = linear_data()
        evaluator: RegressionEvaluator = RegressionEvaluator(KFoldSplitter(n_splits=3))
        report = evaluator.evaluate(
            StackingParameters(
                estimators=(RidgeParameters(), DecisionTreeParameters()), n_splits=3
            ),
            X,
            y,
        )
        self.assertEqual(len(report.model.folds), 3)


class TestExtendedForwarding(unittest.TestCase):
    def test_forwarded_values(self) -> None:
        cases: tuple[tuple[RegressionParameters, dict[str, object]], ...] = (
            (
                AutomaticRelevanceDeterminationParameters(threshold_lambda=5.0),
                {"threshold_lambda": 5.0},
            ),
            (BayesianRidgeParameters(lambda_2=0.5), {"lambda_2": 0.5}),
            (ExtraTreeParameters(max_depth=3), {"max_depth": 3}),
            (
                GammaParameters(solver=GeneralizedLinearSolver.NEWTON_CHOLESKY),
                {"solver": "newton-cholesky"},
            ),
            (
                TweedieParameters(power=1.5, link=TweedieLink.LOG),
                {"power": 1.5, "link": "log"},
            ),
            (
                IsotonicParameters(
                    direction=IsotonicDirection.AUTO, y_min=1.0, y_max=2.0
                ),
                {
                    "increasing": "auto",
                    "out_of_bounds": "clip",
                    "y_min": 1.0,
                    "y_max": 2.0,
                },
            ),
            (LarsParameters(n_nonzero_coefs=2), {"n_nonzero_coefs": 2}),
            (
                LassoLarsParameters(alpha=0.5, is_positive=True),
                {"alpha": 0.5, "positive": True},
            ),
            (
                LassoLarsInformationCriterionParameters(
                    criterion=LassoLarsCriterion.BIC, noise_variance=2.0
                ),
                {"criterion": "bic", "noise_variance": 2.0},
            ),
            (NuSVRParameters(nu=0.25, C=2.0), {"nu": 0.25, "C": 2.0}),
            (OrthogonalMatchingPursuitParameters(tol=2.0), {"tol": 2.0}),
            (
                PartialLeastSquaresParameters(n_components=1, is_scaled=False),
                {"n_components": 1, "scale": False},
            ),
            (RadiusNeighborsParameters(radius=2.0), {"radius": 2.0}),
            (TheilSenParameters(n_subsamples=9), {"n_subsamples": 9}),
        )
        for parameters, expected in cases:
            with self.subTest(parameters=type(parameters).__name__):
                estimator: object = final_estimator(parameters.build_processor())
                for name, value in expected.items():
                    self.assertEqual(getattr(estimator, name), value)

    def test_kernel_ridge_structure(self) -> None:
        centered: object = final_estimator(
            KernelRidgeParameters(kernel=SVMKernel.RBF, gamma=0.5).build_processor()
        )
        assert isinstance(centered, TransformedTargetRegressor)
        inner: object = centered.regressor
        assert isinstance(inner, KernelRidge)
        self.assertEqual(inner.kernel, "rbf")
        self.assertEqual(inner.gamma, 0.5)
        self.assertIsInstance(
            final_estimator(
                KernelRidgeParameters(is_target_centered=False).build_processor()
            ),
            KernelRidge,
        )

    def test_ensembles(self) -> None:
        ridge: RidgeParameters = RidgeParameters()
        bagging: object = final_estimator(
            BaggingParameters(estimator=ridge, n_estimators=3).build_processor()
        )
        assert isinstance(bagging, BaggingRegressor)
        self.assertEqual(bagging.n_estimators, 3)
        self.assertIsInstance(bagging.estimator, Pipeline)
        stacking: object = final_estimator(
            StackingParameters(
                estimators=(ridge, ridge), n_splits=3, is_feature_passthrough=True
            ).build_processor()
        )
        assert isinstance(stacking, StackingRegressor)
        self.assertEqual(
            [name for name, _ in stacking.estimators], ["Ridge_0", "Ridge_1"]
        )
        self.assertEqual(stacking.cv, 3)
        self.assertTrue(stacking.passthrough)
        final: object = stacking.final_estimator
        assert isinstance(final, Pipeline)
        self.assertIsInstance(final.named_steps["estimator"], Ridge)
        voting: object = final_estimator(
            VotingParameters(
                estimators=(ridge, ridge), weights=(1.0, 2.0)
            ).build_processor()
        )
        assert isinstance(voting, VotingRegressor)
        self.assertEqual(voting.weights, [1.0, 2.0])


class TestExtendedValidation(unittest.TestCase):
    def test_invalid_values_are_rejected(self) -> None:
        ridge: RidgeParameters = RidgeParameters()
        cases: dict[str, Callable[[], object]] = {
            "ard threshold": lambda: AutomaticRelevanceDeterminationParameters(
                threshold_lambda=0.0
            ),
            "ard nan": lambda: AutomaticRelevanceDeterminationParameters(
                alpha_1=math.nan
            ),
            "bayesian tol": lambda: BayesianRidgeParameters(tol=0.0),
            "extra tree leaf": lambda: ExtraTreeParameters(min_samples_leaf=0),
            "gamma alpha": lambda: GammaParameters(alpha=-1.0),
            "gamma tol": lambda: GammaParameters(tol=0.0),
            "tweedie power": lambda: TweedieParameters(power=0.5),
            "isotonic bounds": lambda: IsotonicParameters(y_min=2.0, y_max=1.0),
            "isotonic nan": lambda: IsotonicParameters(y_min=math.nan),
            "kernel ridge alpha": lambda: KernelRidgeParameters(alpha=-1.0),
            "kernel ridge gamma": lambda: KernelRidgeParameters(gamma=0.0),
            "lars count": lambda: LarsParameters(n_nonzero_coefs=0),
            "lasso lars alpha": lambda: LassoLarsParameters(alpha=-1.0),
            "lasso lars ic noise": lambda: LassoLarsInformationCriterionParameters(
                noise_variance=0.0
            ),
            "nu svr nu": lambda: NuSVRParameters(nu=1.5),
            "nu svr c": lambda: NuSVRParameters(C=0.0),
            "omp exclusive": lambda: OrthogonalMatchingPursuitParameters(
                n_nonzero_coefs=1, tol=1.0
            ),
            "omp tol": lambda: OrthogonalMatchingPursuitParameters(tol=0.0),
            "pls components": lambda: PartialLeastSquaresParameters(n_components=0),
            "radius": lambda: RadiusNeighborsParameters(radius=0.0),
            "theil sen subsamples": lambda: TheilSenParameters(n_subsamples=0),
            "bagging count": lambda: BaggingParameters(n_estimators=0),
            "bagging samples": lambda: BaggingParameters(max_samples=1.5),
            "stacking empty": lambda: StackingParameters(estimators=()),
            "stacking folds": lambda: StackingParameters(
                estimators=(ridge,), n_splits=1
            ),
            "voting empty": lambda: VotingParameters(estimators=()),
            "voting weights": lambda: VotingParameters(
                estimators=(ridge,), weights=(1.0, 2.0)
            ),
        }
        for name, build in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    _ = build()


if __name__ == "__main__":
    _ = unittest.main()
