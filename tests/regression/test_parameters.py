from __future__ import annotations

import math
import unittest

from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    LinearRegression,
    PoissonRegressor,
    QuantileRegressor,
    Ridge,
    SGDRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor

from classreg.regression import (
    AdaBoostLoss,
    AdaBoostParameters,
    DecisionTreeParameters,
    ElasticNetParameters,
    ExtraTreesParameters,
    FeatureScaling,
    GaussianProcessKernel,
    GaussianProcessParameters,
    GeneralizedLinearSolver,
    GradientBoostingLoss,
    GradientBoostingParameters,
    HistGradientBoostingLoss,
    HistGradientBoostingParameters,
    HuberParameters,
    KNearestNeighborsAlgorithm,
    KNearestNeighborsMetric,
    KNearestNeighborsParameters,
    KNearestNeighborsWeighting,
    LassoParameters,
    LinearSVRLoss,
    LinearSVRParameters,
    MultilayerPerceptronActivation,
    MultilayerPerceptronParameters,
    PoissonParameters,
    QuantileParameters,
    RandomForestMaxFeatures,
    RandomForestParameters,
    RegressionProcessor,
    RidgeParameters,
    StochasticGradientDescentLoss,
    StochasticGradientDescentParameters,
    SVMGamma,
    SVRParameters,
)


class TestParameterValidation(unittest.TestCase):
    def test_ridge(self) -> None:
        with self.assertRaises(ValueError):
            _ = RidgeParameters(alpha=-1.0)
        with self.assertRaises(ValueError):
            _ = RidgeParameters(max_iter=0)
        self.assertEqual(RidgeParameters(alpha=0.0).alpha, 0.0)

    def test_lasso(self) -> None:
        with self.assertRaises(ValueError):
            _ = LassoParameters(alpha=0.0)
        with self.assertRaises(ValueError):
            _ = LassoParameters(max_iter=0)

    def test_elastic_net(self) -> None:
        with self.assertRaises(ValueError):
            _ = ElasticNetParameters(alpha=0.0)
        with self.assertRaises(ValueError):
            _ = ElasticNetParameters(l1_ratio=1.5)

    def test_svr(self) -> None:
        with self.assertRaises(ValueError):
            _ = SVRParameters(C=0.0)
        with self.assertRaises(ValueError):
            _ = SVRParameters(epsilon=-0.1)
        with self.assertRaises(ValueError):
            _ = SVRParameters(gamma=0.0)

    def test_k_nearest_neighbors(self) -> None:
        with self.assertRaises(ValueError):
            _ = KNearestNeighborsParameters(n_neighbors=0)
        with self.assertRaises(ValueError):
            _ = KNearestNeighborsParameters(
                metric=KNearestNeighborsMetric.COSINE,
                algorithm=KNearestNeighborsAlgorithm.BALL_TREE,
            )

    def test_decision_tree(self) -> None:
        with self.assertRaises(ValueError):
            _ = DecisionTreeParameters(max_depth=0)
        with self.assertRaises(ValueError):
            _ = DecisionTreeParameters(min_samples_split=1)

    def test_random_forest(self) -> None:
        with self.assertRaises(ValueError):
            _ = RandomForestParameters(n_estimators=0)

    def test_hist_gradient_boosting(self) -> None:
        with self.assertRaises(ValueError):
            _ = HistGradientBoostingParameters(learning_rate=0.0)
        with self.assertRaises(ValueError):
            _ = HistGradientBoostingParameters(max_bins=256)

    def test_extra_trees(self) -> None:
        with self.assertRaises(ValueError):
            _ = ExtraTreesParameters(n_estimators=0)
        with self.assertRaises(TypeError):
            _ = ExtraTreesParameters(n_estimators=True)

    def test_gradient_boosting(self) -> None:
        with self.assertRaises(ValueError):
            _ = GradientBoostingParameters(learning_rate=math.nan)
        with self.assertRaises(ValueError):
            _ = GradientBoostingParameters(subsample=0.0)
        with self.assertRaises(ValueError):
            _ = GradientBoostingParameters(alpha=1.0)

    def test_ada_boost(self) -> None:
        with self.assertRaises(ValueError):
            _ = AdaBoostParameters(n_estimators=0)
        with self.assertRaises(ValueError):
            _ = AdaBoostParameters(learning_rate=0.0)

    def test_huber(self) -> None:
        with self.assertRaises(ValueError):
            _ = HuberParameters(epsilon=0.5)
        with self.assertRaises(ValueError):
            _ = HuberParameters(epsilon=math.nan)
        with self.assertRaises(ValueError):
            _ = HuberParameters(max_iter=0)

    def test_linear_svr(self) -> None:
        with self.assertRaises(ValueError):
            _ = LinearSVRParameters(C=0.0)
        with self.assertRaises(ValueError):
            _ = LinearSVRParameters(epsilon=-1.0)
        with self.assertRaises(ValueError):
            _ = LinearSVRParameters(tol=0.0)
        with self.assertRaises(ValueError):
            _ = LinearSVRParameters(max_iter=0)

    def test_stochastic_gradient_descent(self) -> None:
        with self.assertRaises(ValueError):
            _ = StochasticGradientDescentParameters(alpha=0.0)
        with self.assertRaises(ValueError):
            _ = StochasticGradientDescentParameters(l1_ratio=1.5)
        with self.assertRaises(ValueError):
            _ = StochasticGradientDescentParameters(epsilon=-1.0)
        with self.assertRaises(ValueError):
            _ = StochasticGradientDescentParameters(max_iter=0)

    def test_quantile(self) -> None:
        for quantile in (0.0, 1.0, math.nan):
            with self.subTest(quantile=quantile):
                with self.assertRaises(ValueError):
                    _ = QuantileParameters(quantile=quantile)
        with self.assertRaises(ValueError):
            _ = QuantileParameters(alpha=-1.0)

    def test_poisson(self) -> None:
        with self.assertRaises(ValueError):
            _ = PoissonParameters(alpha=-1.0)
        with self.assertRaises(ValueError):
            _ = PoissonParameters(max_iter=0)

    def test_gaussian_process(self) -> None:
        with self.assertRaises(ValueError):
            _ = GaussianProcessParameters(alpha=math.nan)
        with self.assertRaises(ValueError):
            _ = GaussianProcessParameters(n_restarts_optimizer=-1)
        with self.assertRaises(TypeError):
            _ = GaussianProcessParameters(n_restarts_optimizer=True)

    def test_multilayer_perceptron(self) -> None:
        with self.assertRaises(ValueError):
            _ = MultilayerPerceptronParameters(hidden_layer_sizes=())
        with self.assertRaises(ValueError):
            _ = MultilayerPerceptronParameters(hidden_layer_sizes=(0,))
        with self.assertRaises(TypeError):
            _ = MultilayerPerceptronParameters(hidden_layer_sizes=(True,))


class TestParameterForwarding(unittest.TestCase):
    def test_ridge_scales_by_default(self) -> None:
        processor: RegressionProcessor = RidgeParameters(alpha=2.0).build_processor()
        assert isinstance(processor.estimator, Pipeline)
        self.assertIsInstance(processor.estimator.named_steps["scaler"], StandardScaler)
        estimator: object = processor.estimator.named_steps["estimator"]
        assert isinstance(estimator, Ridge)
        self.assertEqual(estimator.alpha, 2.0)

    def test_elastic_net(self) -> None:
        processor: RegressionProcessor = ElasticNetParameters(
            l1_ratio=0.25
        ).build_processor()
        assert isinstance(processor.estimator, Pipeline)
        estimator: object = processor.estimator.named_steps["estimator"]
        assert isinstance(estimator, ElasticNet)
        self.assertEqual(estimator.l1_ratio, 0.25)

    def test_svr_gamma_rule_and_value(self) -> None:
        for gamma, expected in ((SVMGamma.AUTO, "auto"), (0.25, 0.25)):
            with self.subTest(gamma=gamma):
                processor: RegressionProcessor = SVRParameters(
                    gamma=gamma
                ).build_processor()
                assert isinstance(processor.estimator, Pipeline)
                estimator: object = processor.estimator.named_steps["estimator"]
                assert isinstance(estimator, SVR)
                self.assertEqual(estimator.gamma, expected)
                self.assertEqual(estimator.max_iter, -1)

    def test_k_nearest_neighbors(self) -> None:
        processor: RegressionProcessor = KNearestNeighborsParameters(
            n_neighbors=3, weighting=KNearestNeighborsWeighting.DISTANCE
        ).build_processor()
        assert isinstance(processor.estimator, Pipeline)
        estimator: object = processor.estimator.named_steps["estimator"]
        assert isinstance(estimator, KNeighborsRegressor)
        self.assertEqual(estimator.n_neighbors, 3)
        self.assertEqual(estimator.weights, "distance")

    def test_random_forest_max_features(self) -> None:
        for max_features, expected in (
            (None, None),
            (RandomForestMaxFeatures.SQRT, "sqrt"),
        ):
            with self.subTest(max_features=max_features):
                processor: RegressionProcessor = RandomForestParameters(
                    max_features=max_features
                ).build_processor()
                assert isinstance(processor.estimator, Pipeline)
                estimator: object = processor.estimator.named_steps["estimator"]
                assert isinstance(estimator, RandomForestRegressor)
                self.assertEqual(estimator.max_features, expected)

    def test_hist_gradient_boosting_loss(self) -> None:
        processor: RegressionProcessor = HistGradientBoostingParameters(
            loss=HistGradientBoostingLoss.ABSOLUTE_ERROR
        ).build_processor()
        assert isinstance(processor.estimator, Pipeline)
        estimator: object = processor.estimator.named_steps["estimator"]
        assert isinstance(estimator, HistGradientBoostingRegressor)
        self.assertEqual(estimator.loss, "absolute_error")
        self.assertFalse(estimator.early_stopping)

    def test_extra_trees_uses_all_features_without_bootstrap(self) -> None:
        processor: RegressionProcessor = ExtraTreesParameters(
            n_estimators=7
        ).build_processor()
        assert isinstance(processor.estimator, Pipeline)
        estimator: object = processor.estimator.named_steps["estimator"]
        assert isinstance(estimator, ExtraTreesRegressor)
        self.assertEqual(estimator.n_estimators, 7)
        self.assertIsNone(estimator.max_features)
        self.assertFalse(estimator.bootstrap)

    def test_gradient_boosting(self) -> None:
        for is_enabled, expected in ((False, None), (True, 4)):
            with self.subTest(is_enabled=is_enabled):
                processor: RegressionProcessor = GradientBoostingParameters(
                    loss=GradientBoostingLoss.HUBER,
                    alpha=0.8,
                    is_early_stopping_enabled=is_enabled,
                    n_iter_no_change=4,
                ).build_processor()
                assert isinstance(processor.estimator, Pipeline)
                estimator: object = processor.estimator.named_steps["estimator"]
                assert isinstance(estimator, GradientBoostingRegressor)
                self.assertEqual(estimator.loss, "huber")
                self.assertEqual(estimator.alpha, 0.8)
                self.assertEqual(estimator.n_iter_no_change, expected)

    def test_ada_boost(self) -> None:
        processor: RegressionProcessor = AdaBoostParameters(
            loss=AdaBoostLoss.SQUARE, max_depth=2
        ).build_processor()
        assert isinstance(processor.estimator, Pipeline)
        estimator: object = processor.estimator.named_steps["estimator"]
        assert isinstance(estimator, AdaBoostRegressor)
        self.assertEqual(estimator.loss, "square")
        tree: object = estimator.estimator
        assert isinstance(tree, DecisionTreeRegressor)
        self.assertEqual(tree.max_depth, 2)

    def test_robust_linear_models_scale_by_default(self) -> None:
        processor: RegressionProcessor = HuberParameters(epsilon=2.0).build_processor()
        assert isinstance(processor.estimator, Pipeline)
        self.assertIsInstance(processor.estimator.named_steps["scaler"], StandardScaler)
        huber: object = processor.estimator.named_steps["estimator"]
        assert isinstance(huber, HuberRegressor)
        self.assertEqual(huber.epsilon, 2.0)

        processor = QuantileParameters(quantile=0.25).build_processor()
        assert isinstance(processor.estimator, Pipeline)
        quantile: object = processor.estimator.named_steps["estimator"]
        assert isinstance(quantile, QuantileRegressor)
        self.assertEqual(quantile.quantile, 0.25)

    def test_linear_svr_forwarding(self) -> None:
        processor: RegressionProcessor = LinearSVRParameters(
            C=2.0,
            epsilon=0.5,
            loss=LinearSVRLoss.SQUARED_EPSILON_INSENSITIVE,
        ).build_processor()
        assert isinstance(processor.estimator, Pipeline)
        estimator: object = processor.estimator.named_steps["estimator"]
        assert isinstance(estimator, LinearSVR)
        self.assertEqual(estimator.C, 2.0)
        self.assertEqual(estimator.epsilon, 0.5)
        self.assertEqual(estimator.loss, "squared_epsilon_insensitive")
        self.assertEqual(estimator.dual, "auto")

    def test_stochastic_gradient_descent_forwarding(self) -> None:
        processor: RegressionProcessor = StochasticGradientDescentParameters(
            loss=StochasticGradientDescentLoss.HUBER,
            l1_ratio=0.25,
            epsilon=5.0,
        ).build_processor()
        assert isinstance(processor.estimator, Pipeline)
        estimator: object = processor.estimator.named_steps["estimator"]
        assert isinstance(estimator, SGDRegressor)
        self.assertEqual(estimator.loss, "huber")
        self.assertEqual(estimator.penalty, "elasticnet")
        self.assertEqual(estimator.l1_ratio, 0.25)
        self.assertEqual(estimator.epsilon, 5.0)

    def test_poisson_solver(self) -> None:
        processor: RegressionProcessor = PoissonParameters(
            solver=GeneralizedLinearSolver.NEWTON_CHOLESKY
        ).build_processor()
        assert isinstance(processor.estimator, Pipeline)
        estimator: object = processor.estimator.named_steps["estimator"]
        assert isinstance(estimator, PoissonRegressor)
        self.assertEqual(estimator.solver, "newton-cholesky")

    def test_gaussian_process_normalizes_targets_by_default(self) -> None:
        processor: RegressionProcessor = GaussianProcessParameters(
            kernel=GaussianProcessKernel.MATERN
        ).build_processor()
        assert isinstance(processor.estimator, Pipeline)
        estimator: object = processor.estimator.named_steps["estimator"]
        assert isinstance(estimator, GaussianProcessRegressor)
        self.assertTrue(estimator.normalize_y)
        self.assertIn("Matern", repr(estimator.kernel))

    def test_multilayer_perceptron(self) -> None:
        processor: RegressionProcessor = MultilayerPerceptronParameters(
            hidden_layer_sizes=(8, 4),
            activation=MultilayerPerceptronActivation.TANH,
        ).build_processor()
        assert isinstance(processor.estimator, Pipeline)
        self.assertIsInstance(processor.estimator.named_steps["scaler"], StandardScaler)
        estimator: object = processor.estimator.named_steps["estimator"]
        assert isinstance(estimator, MLPRegressor)
        self.assertEqual(estimator.hidden_layer_sizes, (8, 4))
        self.assertEqual(estimator.activation, "tanh")


class TestFeatureScaling(unittest.TestCase):
    def test_scaler_types(self) -> None:
        estimator: LinearRegression = LinearRegression()
        self.assertIsInstance(
            FeatureScaling.MIN_MAX.build_pipeline(estimator).named_steps["scaler"],
            MinMaxScaler,
        )
        self.assertIsInstance(
            FeatureScaling.ROBUST.build_pipeline(estimator).named_steps["scaler"],
            RobustScaler,
        )
        self.assertIsInstance(
            FeatureScaling.NONE.build_pipeline(estimator).named_steps["scaler"],
            FunctionTransformer,
        )


if __name__ == "__main__":
    _ = unittest.main()
