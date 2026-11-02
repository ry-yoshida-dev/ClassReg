from __future__ import annotations

import unittest

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from classreg.classification import (
    AdaBoostParameters,
    ClassWeighting,
    ExtraTreesParameters,
    GradientBoostingParameters,
    KNearestNeighborsMetric,
    KNearestNeighborsParameters,
    KNearestNeighborsWeighting,
    LinearDiscriminantAnalysisParameters,
    LinearDiscriminantAnalysisSolver,
    LinearSVCLoss,
    LinearSVCParameters,
    LinearSVCPenalty,
    LogisticRegressionParameters,
    MarginClassificationProcessor,
    MultilayerPerceptronActivation,
    MultilayerPerceptronParameters,
    MultilayerPerceptronSolver,
    ProbabilisticClassificationProcessor,
    RandomForestMaxFeatures,
    RandomForestParameters,
    StochasticGradientDescentLoss,
    StochasticGradientDescentParameters,
    SVCParameters,
    SVMGamma,
    SVMKernel,
)


class TestParameterForwarding(unittest.TestCase):
    def test_logistic_regression(self) -> None:
        processor: ProbabilisticClassificationProcessor = LogisticRegressionParameters(
            C=0.5,
            class_weighting=ClassWeighting.BALANCED,
        ).build_processor()
        assert isinstance(processor, ProbabilisticClassificationProcessor)
        assert isinstance(processor.estimator, Pipeline)
        estimator: object = processor.estimator.named_steps["estimator"]
        assert isinstance(estimator, LogisticRegression)
        self.assertEqual(estimator.C, 0.5)
        self.assertEqual(estimator.class_weight, "balanced")

    def test_svc_gamma_rule_and_value(self) -> None:
        for gamma, expected in ((SVMGamma.AUTO, "auto"), (0.25, 0.25)):
            with self.subTest(gamma=gamma):
                processor: MarginClassificationProcessor = SVCParameters(
                    gamma=gamma, kernel=SVMKernel.POLYNOMIAL
                ).build_processor()
                assert isinstance(processor, MarginClassificationProcessor)
                assert isinstance(processor.estimator, Pipeline)
                estimator: object = processor.estimator.named_steps["estimator"]
                assert isinstance(estimator, SVC)
                self.assertEqual(estimator.gamma, expected)
                self.assertEqual(estimator.kernel, "poly")

    def test_svc_unlimited_iterations_map_to_negative_one(self) -> None:
        processor: MarginClassificationProcessor = SVCParameters().build_processor()
        assert isinstance(processor.estimator, Pipeline)
        estimator: object = processor.estimator.named_steps["estimator"]
        assert isinstance(estimator, SVC)
        self.assertEqual(estimator.max_iter, -1)

    def test_k_nearest_neighbors(self) -> None:
        processor: ProbabilisticClassificationProcessor = KNearestNeighborsParameters(
            n_neighbors=3,
            weighting=KNearestNeighborsWeighting.DISTANCE,
            metric=KNearestNeighborsMetric.MANHATTAN,
        ).build_processor()
        assert isinstance(processor, ProbabilisticClassificationProcessor)
        assert isinstance(processor.estimator, Pipeline)
        estimator: object = processor.estimator.named_steps["estimator"]
        assert isinstance(estimator, KNeighborsClassifier)
        self.assertEqual(estimator.n_neighbors, 3)
        self.assertEqual(estimator.weights, "distance")
        self.assertEqual(estimator.metric, "manhattan")

    def test_random_forest_all_features(self) -> None:
        for max_features, expected in (
            (None, None),
            (RandomForestMaxFeatures.LOG2, "log2"),
        ):
            with self.subTest(max_features=max_features):
                processor: ProbabilisticClassificationProcessor = (
                    RandomForestParameters(max_features=max_features).build_processor()
                )
                assert isinstance(processor, ProbabilisticClassificationProcessor)
                assert isinstance(processor.estimator, Pipeline)
                estimator: object = processor.estimator.named_steps["estimator"]
                assert isinstance(estimator, RandomForestClassifier)
                self.assertEqual(estimator.max_features, expected)

    def test_extra_trees(self) -> None:
        processor: ProbabilisticClassificationProcessor = ExtraTreesParameters(
            n_estimators=7, class_weighting=ClassWeighting.BALANCED
        ).build_processor()
        assert isinstance(processor.estimator, Pipeline)
        estimator: object = processor.estimator.named_steps["estimator"]
        assert isinstance(estimator, ExtraTreesClassifier)
        self.assertEqual(estimator.n_estimators, 7)
        self.assertEqual(estimator.max_features, "sqrt")
        self.assertFalse(estimator.bootstrap)

    def test_gradient_boosting_early_stopping(self) -> None:
        for is_enabled, expected in ((False, None), (True, 4)):
            with self.subTest(is_enabled=is_enabled):
                processor: ProbabilisticClassificationProcessor = (
                    GradientBoostingParameters(
                        subsample=0.5,
                        is_early_stopping_enabled=is_enabled,
                        n_iter_no_change=4,
                    ).build_processor()
                )
                assert isinstance(processor.estimator, Pipeline)
                estimator: object = processor.estimator.named_steps["estimator"]
                assert isinstance(estimator, GradientBoostingClassifier)
                self.assertEqual(estimator.subsample, 0.5)
                self.assertEqual(estimator.n_iter_no_change, expected)

    def test_ada_boost_tree_depth(self) -> None:
        processor: ProbabilisticClassificationProcessor = AdaBoostParameters(
            max_depth=2
        ).build_processor()
        assert isinstance(processor.estimator, Pipeline)
        estimator: object = processor.estimator.named_steps["estimator"]
        assert isinstance(estimator, AdaBoostClassifier)
        tree: object = estimator.estimator
        assert isinstance(tree, DecisionTreeClassifier)
        self.assertEqual(tree.max_depth, 2)

    def test_linear_discriminant_analysis(self) -> None:
        processor: ProbabilisticClassificationProcessor = (
            LinearDiscriminantAnalysisParameters(
                solver=LinearDiscriminantAnalysisSolver.LSQR, shrinkage=0.5
            ).build_processor()
        )
        assert isinstance(processor.estimator, Pipeline)
        estimator: object = processor.estimator.named_steps["estimator"]
        assert isinstance(estimator, LinearDiscriminantAnalysis)
        self.assertEqual(estimator.solver, "lsqr")
        self.assertEqual(estimator.shrinkage, 0.5)

    def test_multilayer_perceptron(self) -> None:
        processor: ProbabilisticClassificationProcessor = (
            MultilayerPerceptronParameters(
                hidden_layer_sizes=(8, 4),
                activation=MultilayerPerceptronActivation.TANH,
                solver=MultilayerPerceptronSolver.LBFGS,
            ).build_processor()
        )
        assert isinstance(processor.estimator, Pipeline)
        self.assertIsInstance(processor.estimator.named_steps["scaler"], StandardScaler)
        estimator: object = processor.estimator.named_steps["estimator"]
        assert isinstance(estimator, MLPClassifier)
        self.assertEqual(estimator.hidden_layer_sizes, (8, 4))
        self.assertEqual(estimator.activation, "tanh")
        self.assertEqual(estimator.solver, "lbfgs")

    def test_stochastic_gradient_descent(self) -> None:
        processor: MarginClassificationProcessor = StochasticGradientDescentParameters(
            loss=StochasticGradientDescentLoss.LOG_LOSS,
            l1_ratio=0.25,
            class_weighting=ClassWeighting.BALANCED,
        ).build_processor()
        assert isinstance(processor.estimator, Pipeline)
        estimator: object = processor.estimator.named_steps["estimator"]
        assert isinstance(estimator, SGDClassifier)
        self.assertEqual(estimator.loss, "log_loss")
        self.assertEqual(estimator.penalty, "elasticnet")
        self.assertEqual(estimator.l1_ratio, 0.25)
        self.assertEqual(estimator.class_weight, "balanced")

    def test_linear_svc(self) -> None:
        processor: MarginClassificationProcessor = LinearSVCParameters(
            C=2.0,
            penalty=LinearSVCPenalty.L1,
            loss=LinearSVCLoss.SQUARED_HINGE,
            class_weighting=ClassWeighting.BALANCED,
        ).build_processor()
        assert isinstance(processor.estimator, Pipeline)
        estimator: object = processor.estimator.named_steps["estimator"]
        assert isinstance(estimator, LinearSVC)
        self.assertEqual(estimator.C, 2.0)
        self.assertEqual(estimator.penalty, "l1")
        self.assertEqual(estimator.loss, "squared_hinge")
        self.assertEqual(estimator.dual, "auto")
        self.assertEqual(estimator.class_weight, "balanced")


if __name__ == "__main__":
    _ = unittest.main()
