from __future__ import annotations

import math
import unittest

import numpy as np
from numpy.typing import NDArray

from classreg.regression import (
    DataSplit,
    EvaluationReport,
    GroupKFoldSplitter,
    HoldoutSplitter,
    KFoldSplitter,
    LeaveOneOutSplitter,
    MetricValues,
    ModelComparison,
    RegressionEvaluator,
    RegressionMetric,
    RepeatedHoldoutSplitter,
    RepeatedKFoldSplitter,
    RidgeParameters,
    TimeSeriesSplitter,
)
from tests.regression.helpers import linear_data


class TestHoldoutSplitter(unittest.TestCase):
    def test_partition_sizes(self) -> None:
        X, y = linear_data(num_samples=100)
        splits: tuple[DataSplit, ...] = HoldoutSplitter(test_fraction=0.25).split(X, y)
        self.assertEqual(len(splits), 1)
        self.assertEqual(splits[0].test_indices.size, 25)
        self.assertEqual(splits[0].train_indices.size, 75)

    def test_invalid_fraction_raises(self) -> None:
        for test_fraction in (0.0, 1.0):
            with self.subTest(test_fraction=test_fraction):
                with self.assertRaises(ValueError):
                    _ = HoldoutSplitter(test_fraction=test_fraction)


class TestKFoldSplitter(unittest.TestCase):
    def test_test_partitions_cover_all_samples_once(self) -> None:
        X, y = linear_data()
        splits: tuple[DataSplit, ...] = KFoldSplitter(n_splits=3).split(X, y)
        test_indices: NDArray[np.intp] = np.concatenate(
            [split.test_indices for split in splits]
        )
        self.assertEqual(np.sort(test_indices).tolist(), list(range(y.size)))
        for split in splits:
            self.assertEqual(
                np.intersect1d(split.train_indices, split.test_indices).size, 0
            )

    def test_is_reproducible(self) -> None:
        X, y = linear_data()
        first: tuple[DataSplit, ...] = KFoldSplitter(random_state=1).split(X, y)
        second: tuple[DataSplit, ...] = KFoldSplitter(random_state=1).split(X, y)
        np.testing.assert_array_equal(first[0].test_indices, second[0].test_indices)

    def test_unshuffled_ignores_random_state(self) -> None:
        X, y = linear_data()
        splits: tuple[DataSplit, ...] = KFoldSplitter(
            is_shuffled=False, random_state=1
        ).split(X, y)
        self.assertEqual(len(splits), 5)

    def test_invalid_n_splits_raises(self) -> None:
        with self.assertRaises(ValueError):
            _ = KFoldSplitter(n_splits=1)


class TestGroupKFoldSplitter(unittest.TestCase):
    def test_groups_never_cross_partitions(self) -> None:
        X, y = linear_data()
        groups: NDArray[np.int64] = np.arange(y.size) % 12
        splits: tuple[DataSplit, ...] = GroupKFoldSplitter(
            groups=groups, n_splits=3
        ).split(X, y)
        for split in splits:
            self.assertEqual(
                np.intersect1d(
                    groups[split.train_indices], groups[split.test_indices]
                ).size,
                0,
            )

    def test_group_length_mismatch_raises(self) -> None:
        X, y = linear_data()
        splitter: GroupKFoldSplitter = GroupKFoldSplitter(groups=np.array([0, 1]))
        with self.assertRaises(ValueError):
            _ = splitter.split(X, y)


class TestTimeSeriesSplitter(unittest.TestCase):
    def test_training_precedes_test_with_gap(self) -> None:
        X, y = linear_data()
        splits: tuple[DataSplit, ...] = TimeSeriesSplitter(n_splits=3, gap=2).split(
            X, y
        )
        for split in splits:
            self.assertEqual(
                int(split.train_indices.max()) + 3, int(split.test_indices.min())
            )

    def test_invalid_arguments_raise(self) -> None:
        with self.assertRaises(ValueError):
            _ = TimeSeriesSplitter(n_splits=1)
        with self.assertRaises(ValueError):
            _ = TimeSeriesSplitter(gap=-1)
        with self.assertRaises(ValueError):
            _ = TimeSeriesSplitter(test_size=0)


class TestRepeatedKFoldSplitter(unittest.TestCase):
    def test_each_repeat_covers_all_samples_once(self) -> None:
        X, y = linear_data()
        splits: tuple[DataSplit, ...] = RepeatedKFoldSplitter(
            n_splits=3, n_repeats=2
        ).split(X, y)
        self.assertEqual(len(splits), 6)
        for repeat in (splits[:3], splits[3:]):
            test_indices: NDArray[np.intp] = np.concatenate(
                [split.test_indices for split in repeat]
            )
            self.assertEqual(np.sort(test_indices).tolist(), list(range(y.size)))

    def test_repeats_differ_and_are_reproducible(self) -> None:
        X, y = linear_data()
        first: tuple[DataSplit, ...] = RepeatedKFoldSplitter(
            n_splits=3, n_repeats=2, random_state=1
        ).split(X, y)
        second: tuple[DataSplit, ...] = RepeatedKFoldSplitter(
            n_splits=3, n_repeats=2, random_state=1
        ).split(X, y)
        np.testing.assert_array_equal(first[0].test_indices, second[0].test_indices)
        self.assertFalse(np.array_equal(first[0].test_indices, first[3].test_indices))

    def test_invalid_arguments_raise(self) -> None:
        with self.assertRaises(ValueError):
            _ = RepeatedKFoldSplitter(n_splits=1)
        with self.assertRaises(ValueError):
            _ = RepeatedKFoldSplitter(n_repeats=0)
        with self.assertRaises(TypeError):
            _ = RepeatedKFoldSplitter(n_repeats=True)

    def test_evaluator_reports_every_fold(self) -> None:
        X, y = linear_data()
        report: EvaluationReport[MetricValues] = RegressionEvaluator(
            splitter=RepeatedKFoldSplitter(n_splits=3, n_repeats=2)
        ).evaluate(RidgeParameters(), X, y)
        self.assertEqual(len(report.model.folds), 6)


class TestRepeatedHoldoutSplitter(unittest.TestCase):
    def test_partition_sizes(self) -> None:
        X, y = linear_data(num_samples=100)
        splits: tuple[DataSplit, ...] = RepeatedHoldoutSplitter(
            n_repeats=3, test_fraction=0.25
        ).split(X, y)
        self.assertEqual(len(splits), 3)
        for split in splits:
            self.assertEqual(split.test_indices.size, 25)
            self.assertEqual(split.train_indices.size, 75)
            self.assertEqual(
                np.intersect1d(split.train_indices, split.test_indices).size, 0
            )
        self.assertFalse(np.array_equal(splits[0].test_indices, splits[1].test_indices))

    def test_invalid_arguments_raise(self) -> None:
        for test_fraction in (0.0, 1.0, math.nan):
            with self.subTest(test_fraction=test_fraction):
                with self.assertRaises(ValueError):
                    _ = RepeatedHoldoutSplitter(test_fraction=test_fraction)
        with self.assertRaises(ValueError):
            _ = RepeatedHoldoutSplitter(n_repeats=0)
        with self.assertRaises(TypeError):
            _ = RepeatedHoldoutSplitter(n_repeats=True)


class TestLeaveOneOutSplitter(unittest.TestCase):
    def test_each_sample_is_tested_once(self) -> None:
        X, y = linear_data(num_samples=10)
        splits: tuple[DataSplit, ...] = LeaveOneOutSplitter().split(X, y)
        self.assertEqual(
            sorted(split.test_indices.item() for split in splits), list(range(10))
        )
        for split in splits:
            self.assertEqual(split.train_indices.size, 9)

    def test_single_sample_raises(self) -> None:
        X, y = linear_data(num_samples=1)
        with self.assertRaises(ValueError):
            _ = LeaveOneOutSplitter().split(X, y)

    def test_evaluation_renders_with_undefined_metrics(self) -> None:
        X, y = linear_data(num_samples=20)
        report: EvaluationReport[MetricValues] = RegressionEvaluator(
            splitter=LeaveOneOutSplitter()
        ).evaluate(RidgeParameters(), X, y)
        self.assertEqual(len(report.model.folds), 20)
        self.assertIsNone(report.model.folds[0].test_metrics.r2)
        self.assertIn("rmse", report.format_table())

    def test_comparison_ranks_by_defined_metric(self) -> None:
        X, y = linear_data(num_samples=20)
        comparison: ModelComparison[MetricValues] = RegressionEvaluator(
            splitter=LeaveOneOutSplitter()
        ).compare({"ridge": RidgeParameters()}, X, y, RegressionMetric.RMSE)
        self.assertEqual(comparison.best.name, "ridge")


if __name__ == "__main__":
    _ = unittest.main()
