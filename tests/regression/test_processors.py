from __future__ import annotations

import unittest

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import NotFittedError

from classreg.regression import (
    AdaBoostParameters,
    BaselineParameters,
    BaselineStrategy,
    DecisionTreeParameters,
    ElasticNetParameters,
    ExtraTreesParameters,
    GaussianProcessKernel,
    GaussianProcessParameters,
    GeneralizedLinearSolver,
    GradientBoostingLoss,
    GradientBoostingParameters,
    HistGradientBoostingParameters,
    HuberParameters,
    KNearestNeighborsParameters,
    LassoParameters,
    LinearRegressionParameters,
    LinearSVRParameters,
    MetricValues,
    MultilayerPerceptronParameters,
    MultilayerPerceptronSolver,
    PoissonParameters,
    QuantileParameters,
    RandomForestParameters,
    RegressionMethod,
    RegressionParameters,
    RegressionProcessor,
    RidgeParameters,
    StochasticGradientDescentParameters,
    SVRParameters,
)
from tests.regression.helpers import linear_data

ALL_PARAMETERS: tuple[RegressionParameters, ...] = (
    LinearRegressionParameters(),
    RidgeParameters(),
    LassoParameters(alpha=0.01),
    ElasticNetParameters(alpha=0.01),
    SVRParameters(C=10.0),
    KNearestNeighborsParameters(),
    DecisionTreeParameters(),
    RandomForestParameters(n_estimators=10),
    HistGradientBoostingParameters(min_samples_leaf=5),
    ExtraTreesParameters(n_estimators=10),
    GradientBoostingParameters(),
    AdaBoostParameters(),
    HuberParameters(),
    LinearSVRParameters(max_iter=5000),
    StochasticGradientDescentParameters(),
    QuantileParameters(alpha=0.01),
    GaussianProcessParameters(),
    MultilayerPerceptronParameters(
        solver=MultilayerPerceptronSolver.LBFGS, max_iter=2000
    ),
)


class TestAllProcessors(unittest.TestCase):
    def test_learns_linear_signal(self) -> None:
        X, y = linear_data()
        for parameters in ALL_PARAMETERS:
            with self.subTest(parameters=type(parameters).__name__):
                processor: RegressionProcessor = parameters.build_processor()
                processor.fit(X, y)
                self.assertEqual(processor.predict(X).shape, y.shape)
                metrics: MetricValues = processor.evaluate(X, y)
                assert metrics.r2 is not None
                self.assertGreater(metrics.r2, 0.85)

    def test_predict_before_fit_raises(self) -> None:
        X, _ = linear_data()
        for parameters in ALL_PARAMETERS:
            with self.subTest(parameters=type(parameters).__name__):
                with self.assertRaises(NotFittedError):
                    _ = parameters.build_processor().predict(X)

    def test_failed_refit_clears_fitted_state(self) -> None:
        X, y = linear_data()
        processor: RegressionProcessor = LinearRegressionParameters().build_processor()
        processor.fit(X, y)
        with self.assertRaises(ValueError):
            processor.fit(X, y[:-1])
        with self.assertRaises(NotFittedError):
            _ = processor.predict(X)

    def test_linear_regression_recovers_coefficients(self) -> None:
        X, y = linear_data(noise_scale=0.0)
        processor: RegressionProcessor = LinearRegressionParameters().build_processor()
        processor.fit(X, y)
        np.testing.assert_allclose(processor.predict(X), y)

    def test_nan_features_are_supported_by_hist_gradient_boosting(self) -> None:
        X, y = linear_data()
        X[0, 0] = np.nan
        processor: RegressionProcessor = HistGradientBoostingParameters(
            max_iter=5, min_samples_leaf=2
        ).build_processor()
        processor.fit(X, y)
        self.assertEqual(processor.predict(X).shape, y.shape)

    def test_poisson_learns_counts(self) -> None:
        X, _ = linear_data()
        rates: NDArray[np.float64] = np.exp(0.5 * X[:, 0] + 1.0)
        for solver in GeneralizedLinearSolver:
            with self.subTest(solver=solver):
                processor: RegressionProcessor = PoissonParameters(
                    alpha=0.01, solver=solver
                ).build_processor()
                processor.fit(X, rates)
                metrics: MetricValues = processor.evaluate(X, rates)
                assert metrics.r2 is not None
                self.assertGreater(metrics.r2, 0.95)

    def test_poisson_rejects_negative_targets(self) -> None:
        X, y = linear_data()
        with self.assertRaises(ValueError):
            PoissonParameters().build_processor().fit(X, y - y.max() - 1.0)

    def test_quantile_predicts_requested_quantile(self) -> None:
        X, y = linear_data(noise_scale=1.0)
        processor: RegressionProcessor = QuantileParameters(
            quantile=0.9, alpha=0.0
        ).build_processor()
        processor.fit(X, y)
        self.assertAlmostEqual(
            float(np.mean(y <= processor.predict(X))), 0.9, delta=0.03
        )

    def test_huber_resists_outliers(self) -> None:
        X, y = linear_data()
        corrupted: NDArray[np.float64] = y.copy()
        corrupted[:5] += 100.0
        processor: RegressionProcessor = HuberParameters().build_processor()
        processor.fit(X, corrupted)
        metrics: MetricValues = processor.evaluate(X, y)
        assert metrics.r2 is not None
        self.assertGreater(metrics.r2, 0.95)

    def test_gaussian_process_kernels(self) -> None:
        X, y = linear_data(num_samples=40, noise_scale=0.2)
        for kernel in GaussianProcessKernel:
            with self.subTest(kernel=kernel):
                processor: RegressionProcessor = GaussianProcessParameters(
                    kernel=kernel
                ).build_processor()
                processor.fit(X, y)
                metrics: MetricValues = processor.evaluate(X, y)
                assert metrics.r2 is not None
                self.assertGreater(metrics.r2, 0.85)

    def test_gradient_boosting_losses(self) -> None:
        X, y = linear_data()
        for loss in GradientBoostingLoss:
            with self.subTest(loss=loss):
                processor: RegressionProcessor = GradientBoostingParameters(
                    loss=loss, alpha=0.5
                ).build_processor()
                processor.fit(X, y)
                metrics: MetricValues = processor.evaluate(X, y)
                assert metrics.r2 is not None
                self.assertGreater(metrics.r2, 0.85)

    def test_method_is_reported(self) -> None:
        self.assertEqual(SVRParameters().build_processor().method, RegressionMethod.SVR)


class TestBaseline(unittest.TestCase):
    def test_predicts_training_statistic(self) -> None:
        X = np.zeros((3, 1))
        y: NDArray[np.float64] = np.array([1.0, 2.0, 9.0])
        for strategy, expected in (
            (BaselineStrategy.MEAN, 4.0),
            (BaselineStrategy.MEDIAN, 2.0),
        ):
            with self.subTest(strategy=strategy):
                processor: RegressionProcessor = BaselineParameters(
                    strategy=strategy
                ).build_processor()
                processor.fit(X, y)
                self.assertEqual(processor.predict(X).tolist(), [expected] * 3)


if __name__ == "__main__":
    _ = unittest.main()
