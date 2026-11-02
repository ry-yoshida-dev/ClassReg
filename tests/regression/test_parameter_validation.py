from __future__ import annotations

import math
import unittest

import numpy as np

from classreg.regression import (
    DecisionTreeParameters,
    ElasticNetParameters,
    GroupKFoldSplitter,
    HistGradientBoostingParameters,
    HoldoutSplitter,
    KFoldSplitter,
    KNearestNeighborsParameters,
    LassoParameters,
    LinearRegressionParameters,
    RandomForestParameters,
    RidgeParameters,
    SVRParameters,
    TimeSeriesSplitter,
)


class TestParameterValidator(unittest.TestCase):
    def test_nan_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _ = RidgeParameters(alpha=math.nan)
        with self.assertRaises(ValueError):
            _ = LassoParameters(tol=math.nan)
        with self.assertRaises(ValueError):
            _ = ElasticNetParameters(l1_ratio=math.nan)
        with self.assertRaises(ValueError):
            _ = SVRParameters(epsilon=math.nan)
        with self.assertRaises(ValueError):
            _ = SVRParameters(gamma=math.nan)
        with self.assertRaises(ValueError):
            _ = DecisionTreeParameters(ccp_alpha=math.nan)
        with self.assertRaises(ValueError):
            _ = HistGradientBoostingParameters(learning_rate=math.nan)
        with self.assertRaises(ValueError):
            _ = HistGradientBoostingParameters(validation_fraction=math.nan)

    def test_non_integers_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            _ = LinearRegressionParameters(n_jobs=True)
        with self.assertRaises(TypeError):
            _ = RidgeParameters(random_state=True)
        with self.assertRaises(TypeError):
            _ = KNearestNeighborsParameters(n_neighbors=True)
        with self.assertRaises(TypeError):
            _ = RandomForestParameters(n_estimators=True)
        with self.assertRaises(TypeError):
            _ = SVRParameters(degree=True)

    def test_none_is_accepted_for_optional_integers(self) -> None:
        self.assertIsNone(RandomForestParameters(max_depth=None).max_depth)
        self.assertIsNone(LinearRegressionParameters(n_jobs=None).n_jobs)

    def test_splitters_validate_integers(self) -> None:
        with self.assertRaises(TypeError):
            _ = HoldoutSplitter(random_state=True)
        with self.assertRaises(TypeError):
            _ = KFoldSplitter(n_splits=True)
        with self.assertRaises(TypeError):
            _ = TimeSeriesSplitter(gap=True)
        with self.assertRaises(TypeError):
            _ = GroupKFoldSplitter(groups=np.array([0, 1]), n_splits=True)

    def test_float_groups_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            _ = GroupKFoldSplitter(groups=np.array([0.5, 1.5]))


if __name__ == "__main__":
    _ = unittest.main()
