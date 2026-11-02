from __future__ import annotations

import unittest

import numpy as np
from sklearn.exceptions import NotFittedError

from classreg.classification import (
    AdaBoostParameters,
    ClassificationMethod,
    ClassificationParameters,
    DecisionTreeParameters,
    ExtraTreesParameters,
    GaussianNaiveBayesParameters,
    GradientBoostingParameters,
    HistGradientBoostingParameters,
    KNearestNeighborsParameters,
    LinearDiscriminantAnalysisParameters,
    LinearSVCParameters,
    LogisticRegressionParameters,
    MarginClassificationProcessor,
    MetricValues,
    MultilayerPerceptronParameters,
    ProbabilisticClassificationProcessor,
    QuadraticDiscriminantAnalysisParameters,
    RandomForestParameters,
    StochasticGradientDescentParameters,
    SVCParameters,
)
from classreg.classification.processor import ClassificationProcessor
from classreg.classification.types import LabelArray, NumericArray
from tests.classification.helpers import (
    separable_blobs,
)

ALL_PARAMETERS: tuple[ClassificationParameters, ...] = (
    LogisticRegressionParameters(),
    SVCParameters(),
    KNearestNeighborsParameters(),
    DecisionTreeParameters(),
    RandomForestParameters(n_estimators=10),
    HistGradientBoostingParameters(max_iter=10, min_samples_leaf=2),
    GaussianNaiveBayesParameters(),
    ExtraTreesParameters(n_estimators=10),
    GradientBoostingParameters(n_estimators=10),
    AdaBoostParameters(n_estimators=10, max_depth=2),
    LinearDiscriminantAnalysisParameters(),
    QuadraticDiscriminantAnalysisParameters(),
    MultilayerPerceptronParameters(max_iter=2000),
)


class TestAllProcessors(unittest.TestCase):
    def test_learns_separable_data(self) -> None:
        X, y = separable_blobs()
        for parameters in ALL_PARAMETERS:
            with self.subTest(parameters=type(parameters).__name__):
                processor: ClassificationProcessor = parameters.build_processor()
                processor.fit(X, y)
                self.assertEqual(processor.classes.tolist(), [0, 1, 2])
                self.assertEqual(processor.predict(X).tolist(), y.tolist())
                metrics: MetricValues = processor.evaluate(X, y)
                self.assertEqual(metrics.accuracy, 1.0)
                self.assertEqual(metrics.roc_auc, 1.0)

    def test_predict_before_fit_raises(self) -> None:
        X, _ = separable_blobs()
        for parameters in ALL_PARAMETERS:
            with self.subTest(parameters=type(parameters).__name__):
                with self.assertRaises(NotFittedError):
                    _ = parameters.build_processor().predict(X)

    def test_string_labels(self) -> None:
        X, y = separable_blobs(num_classes=2)
        labels: LabelArray = np.array(["cat", "dog"])[y]
        for parameters in ALL_PARAMETERS:
            with self.subTest(parameters=type(parameters).__name__):
                processor: ClassificationProcessor = parameters.build_processor()
                processor.fit(X, labels)
                self.assertEqual(processor.evaluate(X, labels).accuracy, 1.0)

    def test_single_class_is_rejected(self) -> None:
        X, y = separable_blobs()
        processor: ProbabilisticClassificationProcessor = (
            LogisticRegressionParameters().build_processor()
        )
        with self.assertRaises(ValueError):
            processor.fit(X, np.zeros_like(y))

    def test_failed_refit_clears_fitted_state(self) -> None:
        X, y = separable_blobs()
        processor: ProbabilisticClassificationProcessor = (
            LogisticRegressionParameters().build_processor()
        )
        processor.fit(X, y)
        with self.assertRaises(ValueError):
            processor.fit(X, np.zeros_like(y))
        with self.assertRaises(NotFittedError):
            _ = processor.classes

    def test_unseen_label_in_evaluate_raises(self) -> None:
        X, y = separable_blobs(num_classes=2)
        processor: ProbabilisticClassificationProcessor = (
            LogisticRegressionParameters().build_processor()
        )
        processor.fit(X, y)
        with self.assertRaises(ValueError):
            _ = processor.evaluate(X, y + 5)

    def test_method_is_reported(self) -> None:
        processor: MarginClassificationProcessor = SVCParameters().build_processor()
        self.assertEqual(processor.method, ClassificationMethod.SVC)

    def test_processors_compare_by_identity(self) -> None:
        X, y = separable_blobs()
        first: ClassificationProcessor = (
            GaussianNaiveBayesParameters().build_processor()
        )
        second: ClassificationProcessor = (
            GaussianNaiveBayesParameters().build_processor()
        )
        first.fit(X, y)
        second.fit(X, y)
        self.assertNotEqual(first, second)
        self.assertEqual(first, first)

    def test_fitted_classes_are_read_only(self) -> None:
        X, y = separable_blobs()
        processor: ClassificationProcessor = (
            GaussianNaiveBayesParameters().build_processor()
        )
        processor.fit(X, y)
        self.assertFalse(processor.classes.flags.writeable)

    def test_one_vs_rest_linear_models_learn_linearly_separable_classes(self) -> None:
        rng: np.random.Generator = np.random.default_rng(0)
        centers: list[list[float]] = [[0.0, 0.0], [16.0, 0.0], [8.0, 16.0]]
        X: NumericArray = np.vstack(
            [rng.standard_normal((30, 2)) + center for center in centers]
        )
        y: LabelArray = np.repeat(np.arange(3), 30)
        for parameters in (
            LinearSVCParameters(),
            StochasticGradientDescentParameters(),
        ):
            with self.subTest(parameters=type(parameters).__name__):
                processor: MarginClassificationProcessor = parameters.build_processor()
                with self.assertRaises(NotFittedError):
                    _ = processor.predict(X)
                processor.fit(X, y)
                self.assertEqual(processor.evaluate(X, y).accuracy, 1.0)
                self.assertEqual(
                    processor.decision_function(X).values.shape, (y.size, 3)
                )

    def test_one_vs_rest_linear_models_produce_margins(self) -> None:
        X, y = separable_blobs(num_classes=2)
        for parameters in (
            LinearSVCParameters(),
            StochasticGradientDescentParameters(),
        ):
            with self.subTest(parameters=type(parameters).__name__):
                processor: MarginClassificationProcessor = parameters.build_processor()
                processor.fit(X, y)
                self.assertEqual(
                    processor.decision_function(X).values.shape, (y.size, 2)
                )
                self.assertIsNone(processor.evaluate(X, y).log_loss)

    def test_score_kind_matches_scores(self) -> None:
        X, y = separable_blobs(num_classes=2)
        svc: MarginClassificationProcessor = SVCParameters().build_processor()
        tree: ProbabilisticClassificationProcessor = (
            DecisionTreeParameters().build_processor()
        )
        svc.fit(X, y)
        tree.fit(X, y)
        self.assertEqual(svc.score_kind, svc.decision_function(X).kind)
        self.assertEqual(tree.score_kind, tree.predict_proba(X).kind)


if __name__ == "__main__":
    _ = unittest.main()
