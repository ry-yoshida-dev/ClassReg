from __future__ import annotations

import unittest
from collections.abc import Callable

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import BaggingClassifier, StackingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline

from classreg import FeatureScaling
from classreg.classification import (
    BaggingParameters,
    BernoulliNaiveBayesParameters,
    CategoricalNaiveBayesParameters,
    ClassificationEvaluator,
    ClassificationParameters,
    ClassificationProcessor,
    ClassScoreKind,
    ComplementNaiveBayesParameters,
    DecisionTreeParameters,
    ExtraTreeParameters,
    GaussianProcessKernel,
    GaussianProcessParameters,
    KNearestNeighborsParameters,
    LogisticRegressionParameters,
    MarginClassificationProcessor,
    MetricValues,
    MultinomialNaiveBayesParameters,
    NearestCentroidParameters,
    NuSVCParameters,
    PerceptronParameters,
    PerceptronPenalty,
    ProbabilisticClassificationProcessor,
    RadiusNeighborsOutlierHandling,
    RadiusNeighborsParameters,
    RidgeClassifierParameters,
    StackingParameters,
    StratifiedKFoldSplitter,
    SVCParameters,
    VotingParameters,
)
from classreg.classification.types import LabelArray, NumericArray
from tests.classification.helpers import separable_blobs

ONE_VS_REST_PARAMETERS: tuple[ClassificationParameters, ...] = (
    BernoulliNaiveBayesParameters(),
    NuSVCParameters(),
    PerceptronParameters(),
    RidgeClassifierParameters(),
    NearestCentroidParameters(),
    ExtraTreeParameters(),
    GaussianProcessParameters(),
    RadiusNeighborsParameters(radius=1.0),
    BaggingParameters(),
    BaggingParameters(estimator=SVCParameters()),
    StackingParameters(estimators=(DecisionTreeParameters(), SVCParameters())),
    VotingParameters(
        estimators=(LogisticRegressionParameters(), DecisionTreeParameters())
    ),
)
MARGIN_PARAMETERS: tuple[ClassificationParameters, ...] = (
    NuSVCParameters(),
    PerceptronParameters(),
    RidgeClassifierParameters(),
)


def triangular_blobs() -> tuple[NumericArray, LabelArray]:
    """Return three classes that one linear boundary each can separate."""
    rng: np.random.Generator = np.random.default_rng(0)
    centers: list[list[float]] = [[0.0, 0.0], [16.0, 0.0], [8.0, 16.0]]
    features: NumericArray = np.vstack(
        [rng.standard_normal((30, 2)) + center for center in centers]
    )
    return features, np.repeat(np.arange(3), 30)


def final_estimator(processor: ClassificationProcessor) -> object:
    """Return the estimator step of the pipeline inside a processor."""
    pipeline: object = processor.underlying_estimator
    assert isinstance(pipeline, Pipeline)
    return pipeline.named_steps["estimator"]


class TestExtendedMethods(unittest.TestCase):
    def test_learns_separable_classes(self) -> None:
        X, y = triangular_blobs()
        for parameters in ONE_VS_REST_PARAMETERS:
            with self.subTest(parameters=type(parameters).__name__):
                processor: ClassificationProcessor = parameters.build_processor()
                processor.fit(X, y)
                metrics: MetricValues = processor.evaluate(X, y)
                self.assertEqual(metrics.accuracy, 1.0)
                self.assertEqual(metrics.roc_auc, 1.0)

    def test_score_kinds(self) -> None:
        X, y = triangular_blobs()
        for parameters in ONE_VS_REST_PARAMETERS:
            with self.subTest(parameters=type(parameters).__name__):
                processor: ClassificationProcessor = parameters.build_processor()
                expected: ClassScoreKind = (
                    ClassScoreKind.MARGIN
                    if parameters in MARGIN_PARAMETERS
                    else ClassScoreKind.PROBABILITY
                )
                self.assertIs(processor.score_kind, expected)
                processor.fit(X, y)
                metrics: MetricValues = processor.evaluate(X, y)
                if expected is ClassScoreKind.MARGIN:
                    self.assertIsNone(metrics.log_loss)
                else:
                    self.assertIsNotNone(metrics.log_loss)

    def test_string_labels_and_binary_margins(self) -> None:
        X, y = separable_blobs(num_classes=2)
        labels: LabelArray = np.array(["cat", "dog"])[y]
        for parameters in ONE_VS_REST_PARAMETERS:
            with self.subTest(parameters=type(parameters).__name__):
                processor: ClassificationProcessor = parameters.build_processor()
                processor.fit(X, labels)
                self.assertEqual(processor.predict(X).tolist(), labels.tolist())
                self.assertEqual(processor.evaluate(X, labels).accuracy, 1.0)

    def test_count_naive_bayes_methods_learn_counts(self) -> None:
        rng: np.random.Generator = np.random.default_rng(0)
        X: NumericArray = np.vstack(
            [
                rng.poisson([20.0, 1.0, 1.0], (40, 3)),
                rng.poisson([1.0, 20.0, 1.0], (40, 3)),
                rng.poisson([1.0, 1.0, 20.0], (40, 3)),
            ]
        ).astype(np.float64)
        y: LabelArray = np.repeat(np.arange(3), 40)
        parameters: tuple[ClassificationParameters, ...] = (
            MultinomialNaiveBayesParameters(),
            ComplementNaiveBayesParameters(),
        )
        for candidate in parameters:
            with self.subTest(parameters=type(candidate).__name__):
                processor: ClassificationProcessor = candidate.build_processor()
                processor.fit(X, y)
                self.assertEqual(processor.evaluate(X, y).accuracy, 1.0)

    def test_count_naive_bayes_rejects_negative_features(self) -> None:
        X, y = separable_blobs()
        processor: ClassificationProcessor = (
            MultinomialNaiveBayesParameters().build_processor()
        )
        with self.assertRaises(ValueError):
            processor.fit(X - 100.0, y)

    def test_categorical_naive_bayes_needs_known_categories(self) -> None:
        X: NumericArray = np.array([[0.0], [1.0]] * 10)
        y: LabelArray = np.array([0, 1] * 10)
        unseen: NumericArray = np.array([[5.0]])
        processor: ClassificationProcessor = (
            CategoricalNaiveBayesParameters().build_processor()
        )
        processor.fit(X, y)
        self.assertEqual(processor.predict(X[:2]).tolist(), [0, 1])
        with self.assertRaises(IndexError):
            _ = processor.predict(unseen)
        covering: ClassificationProcessor = CategoricalNaiveBayesParameters(
            min_categories=6
        ).build_processor()
        covering.fit(X, y)
        self.assertEqual(covering.predict(unseen).shape, (1,))

    def test_radius_neighbors_outlier_handling(self) -> None:
        X, y = triangular_blobs()
        far: NumericArray = np.array([[1000.0, 1000.0]])
        raising: ClassificationProcessor = RadiusNeighborsParameters(
            radius=0.5
        ).build_processor()
        raising.fit(X, y)
        with self.assertRaises(ValueError):
            _ = raising.predict(far)
        assigning: ClassificationProcessor = RadiusNeighborsParameters(
            radius=0.5,
            outlier_handling=RadiusNeighborsOutlierHandling.MOST_FREQUENT,
        ).build_processor()
        assigning.fit(X, y)
        self.assertEqual(assigning.predict(far).shape, (1,))

    def test_gaussian_process_probabilities_are_finite(self) -> None:
        iris = load_iris()
        X: NumericArray = iris.data
        y: LabelArray = iris.target
        for kernel in (GaussianProcessKernel.RBF, GaussianProcessKernel.MATERN):
            with self.subTest(kernel=kernel):
                processor: ClassificationProcessor = GaussianProcessParameters(
                    kernel=kernel
                ).build_processor()
                assert isinstance(processor, ProbabilisticClassificationProcessor)
                processor.fit(X, y)
                self.assertTrue(np.isfinite(processor.predict_proba(X).values).all())
                accuracy: float = processor.evaluate(X, y).accuracy
                self.assertGreater(accuracy, 0.9)

    def test_ensembles_of_margin_methods_give_probabilities(self) -> None:
        cancer = load_breast_cancer()
        X: NumericArray = cancer.data
        y: LabelArray = cancer.target
        candidates: tuple[ClassificationParameters, ...] = (
            BaggingParameters(estimator=SVCParameters()),
            StackingParameters(estimators=(SVCParameters(), PerceptronParameters())),
        )
        for parameters in candidates:
            with self.subTest(parameters=type(parameters).__name__):
                processor: ClassificationProcessor = parameters.build_processor()
                assert isinstance(processor, ProbabilisticClassificationProcessor)
                processor.fit(X, y)
                values: NumericArray = processor.predict_proba(X).values
                self.assertEqual(values.shape, (y.size, 2))
                self.assertTrue(np.allclose(values.sum(axis=1), 1.0))
                accuracy: float = processor.evaluate(X, y).accuracy
                self.assertGreater(accuracy, 0.9)

    def test_stacking_evaluates_across_folds(self) -> None:
        cancer = load_breast_cancer()
        evaluator: ClassificationEvaluator = ClassificationEvaluator(
            StratifiedKFoldSplitter(n_splits=3)
        )
        report = evaluator.evaluate(
            StackingParameters(
                estimators=(KNearestNeighborsParameters(), SVCParameters())
            ),
            cancer.data,
            cancer.target,
        )
        self.assertEqual(len(report.model.folds), 3)

    def test_each_build_gets_a_new_estimator(self) -> None:
        X, y = triangular_blobs()
        parameters: BaggingParameters = BaggingParameters()
        first: ClassificationProcessor = parameters.build_processor()
        second: ClassificationProcessor = parameters.build_processor()
        self.assertIsNot(first.underlying_estimator, second.underlying_estimator)
        first.fit(X, y)
        self.assertEqual(second.score_kind, ClassScoreKind.PROBABILITY)


class TestExtendedForwarding(unittest.TestCase):
    def test_margin_processors(self) -> None:
        for parameters in MARGIN_PARAMETERS:
            with self.subTest(parameters=type(parameters).__name__):
                self.assertIsInstance(
                    parameters.build_processor(), MarginClassificationProcessor
                )

    def test_forwarded_values(self) -> None:
        cases: tuple[tuple[ClassificationParameters, dict[str, object]], ...] = (
            (
                BernoulliNaiveBayesParameters(alpha=2.0, binarize_threshold=None),
                {"alpha": 2.0, "binarize": None},
            ),
            (MultinomialNaiveBayesParameters(alpha=2.0), {"alpha": 2.0}),
            (
                ComplementNaiveBayesParameters(is_normalized=True),
                {"norm": True},
            ),
            (
                CategoricalNaiveBayesParameters(min_categories=4),
                {"min_categories": 4},
            ),
            (ExtraTreeParameters(max_depth=3), {"max_depth": 3}),
            (
                NearestCentroidParameters(shrink_threshold=0.5, is_prior_fitted=True),
                {"shrink_threshold": 0.5, "priors": "empirical"},
            ),
            (NuSVCParameters(nu=0.25), {"nu": 0.25}),
            (
                PerceptronParameters(
                    penalty=PerceptronPenalty.ELASTIC_NET, learning_rate=0.5
                ),
                {"penalty": "elasticnet", "eta0": 0.5},
            ),
            (
                RadiusNeighborsParameters(
                    radius=2.0,
                    outlier_handling=RadiusNeighborsOutlierHandling.MOST_FREQUENT,
                ),
                {"radius": 2.0, "outlier_label": "most_frequent"},
            ),
            (RidgeClassifierParameters(alpha=2.0), {"alpha": 2.0}),
        )
        for parameters, expected in cases:
            with self.subTest(parameters=type(parameters).__name__):
                estimator: object = final_estimator(parameters.build_processor())
                for name, value in expected.items():
                    self.assertEqual(getattr(estimator, name), value)

    def test_bagging_wraps_the_inner_pipeline(self) -> None:
        estimator: object = final_estimator(
            BaggingParameters(
                estimator=KNearestNeighborsParameters(),
                n_estimators=3,
                is_feature_bootstrapped=True,
            ).build_processor()
        )
        assert isinstance(estimator, BaggingClassifier)
        self.assertEqual(estimator.n_estimators, 3)
        self.assertTrue(estimator.bootstrap_features)
        self.assertIsInstance(estimator.estimator, Pipeline)

    def test_stacking_and_voting_name_estimators_uniquely(self) -> None:
        duplicates: tuple[ClassificationParameters, ...] = (
            DecisionTreeParameters(),
            DecisionTreeParameters(max_depth=2),
        )
        stacking: object = final_estimator(
            StackingParameters(
                estimators=duplicates, n_splits=3, is_feature_passthrough=True
            ).build_processor()
        )
        assert isinstance(stacking, StackingClassifier)
        self.assertEqual(
            [name for name, _ in stacking.estimators],
            ["DecisionTree_0", "DecisionTree_1"],
        )
        self.assertEqual(stacking.cv, 3)
        self.assertTrue(stacking.passthrough)
        voting: object = final_estimator(
            VotingParameters(
                estimators=duplicates, weights=(1.0, 2.0)
            ).build_processor()
        )
        assert isinstance(voting, VotingClassifier)
        self.assertEqual(voting.voting, "soft")
        self.assertEqual(voting.weights, [1.0, 2.0])


class TestExtendedValidation(unittest.TestCase):
    def test_invalid_values_are_rejected(self) -> None:
        tree: DecisionTreeParameters = DecisionTreeParameters()
        cases: dict[str, Callable[[], object]] = {
            "bernoulli alpha": lambda: BernoulliNaiveBayesParameters(alpha=0.0),
            "bernoulli nan": lambda: BernoulliNaiveBayesParameters(
                binarize_threshold=float("nan")
            ),
            "multinomial scaling": lambda: MultinomialNaiveBayesParameters(
                scaling=FeatureScaling.STANDARD
            ),
            "complement scaling": lambda: ComplementNaiveBayesParameters(
                scaling=FeatureScaling.ROBUST
            ),
            "categorical min": lambda: CategoricalNaiveBayesParameters(
                min_categories=0
            ),
            "extra tree depth": lambda: ExtraTreeParameters(max_depth=0),
            "gp restarts": lambda: GaussianProcessParameters(n_restarts_optimizer=-1),
            "gp iterations": lambda: GaussianProcessParameters(max_iter_predict=0),
            "centroid shrink": lambda: NearestCentroidParameters(shrink_threshold=0.0),
            "nu svc nu": lambda: NuSVCParameters(nu=0.0),
            "nu svc gamma": lambda: NuSVCParameters(gamma=0.0),
            "perceptron ratio": lambda: PerceptronParameters(l1_ratio=1.5),
            "perceptron step": lambda: PerceptronParameters(learning_rate=0.0),
            "radius": lambda: RadiusNeighborsParameters(radius=0.0),
            "ridge alpha": lambda: RidgeClassifierParameters(alpha=-1.0),
            "bagging count": lambda: BaggingParameters(n_estimators=0),
            "bagging samples": lambda: BaggingParameters(max_samples=0.0),
            "bagging features": lambda: BaggingParameters(max_features=1.5),
            "stacking empty": lambda: StackingParameters(estimators=()),
            "stacking folds": lambda: StackingParameters(
                estimators=(tree,), n_splits=1
            ),
            "stacking margin final": lambda: StackingParameters(
                estimators=(tree,), final_estimator=SVCParameters()
            ),
            "voting empty": lambda: VotingParameters(estimators=()),
            "voting margin": lambda: VotingParameters(estimators=(SVCParameters(),)),
            "voting weight count": lambda: VotingParameters(
                estimators=(tree,), weights=(1.0, 2.0)
            ),
            "voting negative weight": lambda: VotingParameters(
                estimators=(tree,), weights=(-1.0,)
            ),
            "voting zero weights": lambda: VotingParameters(
                estimators=(tree,), weights=(0.0,)
            ),
        }
        for name, build in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    _ = build()

    def test_min_max_scaling_keeps_count_naive_bayes_valid(self) -> None:
        X, y = separable_blobs(num_classes=2)
        processor: ClassificationProcessor = MultinomialNaiveBayesParameters(
            scaling=FeatureScaling.MIN_MAX
        ).build_processor()
        processor.fit(X, y)
        self.assertEqual(processor.predict(X).shape, y.shape)


if __name__ == "__main__":
    _ = unittest.main()
