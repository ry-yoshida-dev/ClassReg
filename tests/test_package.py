from __future__ import annotations

import unittest

import numpy as np
from numpy.typing import NDArray

import classreg
from classreg import classification, regression


class TestPackageLayout(unittest.TestCase):
    def test_shared_components_are_the_same_objects(self) -> None:
        self.assertIs(classification.FeatureScaling, classreg.FeatureScaling)
        self.assertIs(regression.FeatureScaling, classreg.FeatureScaling)
        self.assertIs(classification.MetricSummary, regression.MetricSummary)
        self.assertIs(classification.SVMKernel, regression.SVMKernel)

    def test_same_named_methods_stay_task_specific(self) -> None:
        self.assertIsNot(
            classification.RandomForestParameters,
            regression.RandomForestParameters,
        )

    def test_evaluators_share_the_base_class(self) -> None:
        self.assertIsInstance(
            classification.ClassificationEvaluator(), classreg.SupervisedEvaluator
        )
        self.assertIsInstance(
            regression.RegressionEvaluator(), classreg.SupervisedEvaluator
        )

    def test_shared_splitter_works_for_classification(self) -> None:
        rng: np.random.Generator = np.random.default_rng(0)
        X: NDArray[np.float64] = rng.standard_normal((40, 2))
        y: NDArray[np.int64] = np.array([0, 1] * 20, dtype=np.int64)
        report: classification.EvaluationReport[classification.MetricValues] = (
            classification.ClassificationEvaluator(
                splitter=classreg.HoldoutSplitter()
            ).evaluate(classification.LogisticRegressionParameters(), X, y)
        )
        self.assertEqual(len(report.model.folds), 1)

    def test_reports_list_only_their_own_metrics(self) -> None:
        rng: np.random.Generator = np.random.default_rng(0)
        X: NDArray[np.float64] = rng.standard_normal((40, 2))
        table: str = (
            regression.RegressionEvaluator()
            .evaluate(regression.RidgeParameters(), X, X[:, 0])
            .format_table()
        )
        self.assertIn("rmse", table)
        self.assertNotIn("accuracy", table)


if __name__ == "__main__":
    _ = unittest.main()
