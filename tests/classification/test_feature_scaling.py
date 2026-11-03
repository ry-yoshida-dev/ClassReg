from __future__ import annotations

import unittest

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from classreg.classification import (
    FeatureScaling,
)


class TestFeatureScaling(unittest.TestCase):
    def test_scaler_types(self) -> None:
        estimator: GaussianNB = GaussianNB()
        self.assertIsInstance(
            FeatureScaling.STANDARD.build_pipeline(estimator).named_steps["scaler"],
            StandardScaler,
        )
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
