from __future__ import annotations

import unittest

from classreg.classification import (
    ClassificationEvaluator,
    ClassificationMetric,
    DataSplit,
    EvaluationReport,
    GaussianNaiveBayesParameters,
    LeaveOneOutSplitter,
    MetricValues,
    ModelComparison,
)
from tests.classification.helpers import separable_blobs


class TestLeaveOneOutWithClassification(unittest.TestCase):
    def test_each_sample_is_tested_once(self) -> None:
        X, y = separable_blobs(num_samples_per_class=5)
        splits: tuple[DataSplit, ...] = LeaveOneOutSplitter().split(X, y)
        self.assertEqual(
            sorted(split.test_indices.item() for split in splits),
            list(range(y.size)),
        )

    def test_evaluation_renders_with_undefined_metrics(self) -> None:
        X, y = separable_blobs(num_samples_per_class=5)
        report: EvaluationReport[MetricValues] = ClassificationEvaluator(
            splitter=LeaveOneOutSplitter()
        ).evaluate(GaussianNaiveBayesParameters(), X, y)
        self.assertEqual(
            report.model.summarize(ClassificationMetric.ACCURACY).mean, 1.0
        )
        self.assertIn("accuracy", report.format_table())
        for metric in (
            ClassificationMetric.AVERAGE_PRECISION,
            ClassificationMetric.MATTHEWS_CORRELATION,
            ClassificationMetric.COHEN_KAPPA,
            ClassificationMetric.ROC_AUC,
        ):
            with self.subTest(metric=metric):
                with self.assertRaises(ValueError):
                    _ = report.model.summarize(metric)

    def test_class_prior_baseline_is_wrong_on_balanced_classes(self) -> None:
        X, y = separable_blobs(num_samples_per_class=5)
        report: EvaluationReport[MetricValues] = ClassificationEvaluator(
            splitter=LeaveOneOutSplitter()
        ).evaluate(GaussianNaiveBayesParameters(), X, y)
        assert report.baseline is not None
        self.assertEqual(
            report.baseline.summarize(ClassificationMetric.ACCURACY).mean, 0.0
        )

    def test_comparison_ranks_by_defined_metric(self) -> None:
        X, y = separable_blobs(num_samples_per_class=5)
        comparison: ModelComparison[MetricValues] = ClassificationEvaluator(
            splitter=LeaveOneOutSplitter()
        ).compare(
            {"bayes": GaussianNaiveBayesParameters()},
            X,
            y,
            ClassificationMetric.ACCURACY,
        )
        self.assertEqual(comparison.best.name, "bayes")


if __name__ == "__main__":
    _ = unittest.main()
