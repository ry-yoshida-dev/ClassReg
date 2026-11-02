# ClassReg

## Overview

ClassReg (`classreg`) is a Python package that wraps scikit-learn classifiers and regressors behind a small, consistent API and adds a performance evaluation layer. Parameter dataclasses validate their fields and build processors with `fit`, `predict`, and `evaluate`; evaluators estimate generalization performance by cross-validation and compare configurations.

| Task | Package | Methods |
| --- | --- | --- |
| Classification | `classreg.classification` | Logistic regression, SVC, k-nearest neighbors, decision tree, random forest, histogram gradient boosting, Gaussian naive Bayes |
| Regression | `classreg.regression` | Linear regression, ridge, lasso, elastic net, SVR, k-nearest neighbors, decision tree, random forest, histogram gradient boosting |

Each task package exports everything needed for that task. Same-named classes such as `RandomForestParameters` are task-specific, so the top-level `classreg` package exports only the shared components.

For module-level design notes, see [src/classreg/README.md](src/classreg/README.md).

## Installation

From the package root (the directory containing `pyproject.toml`):

```bash
pip install .
```

For development:

```bash
pip install -e .
```

## Quick example

Train a classifier and measure it on separate test data:

```python
from classreg.classification import RandomForestParameters

processor = RandomForestParameters(n_estimators=200).build_processor()
processor.fit(X_train, y_train)
labels = processor.predict(X_test)
probabilities = processor.predict_proba(X_test)

metrics = processor.evaluate(X_test, y_test)
print(metrics.macro_f1, metrics.roc_auc)
print(metrics.confusion_matrix.counts)
```

Cross-validate a classifier against a feature-independent baseline:

```python
from classreg.classification import (
    ClassificationEvaluator,
    ClassificationMetric,
    LogisticRegressionParameters,
)

report = ClassificationEvaluator().evaluate(LogisticRegressionParameters(), X, y)
print(report.format_table())
print(report.model.summarize(ClassificationMetric.MACRO_F1).mean)
print(report.get_baseline_improvement(ClassificationMetric.MACRO_F1))
print(ClassificationEvaluator.combine_confusion_matrices(report.model).counts)
```

```text
metric                train            test             baseline test
--------------------  ---------------  ---------------  ---------------
accuracy              0.9881 ± 0.0046  0.9789 ± 0.0100  0.6274 ± 0.0044
macro_f1              0.9873 ± 0.0049  0.9773 ± 0.0108  0.3855 ± 0.0017
...
```

Rank regressors on identical splits, by RMSE unless another metric is given:

```python
from classreg.regression import (
    LassoParameters,
    RandomForestParameters,
    RegressionEvaluator,
    RidgeParameters,
)

comparison = RegressionEvaluator().compare(
    {
        "ridge": RidgeParameters(),
        "lasso": LassoParameters(alpha=0.1),
        "forest": RandomForestParameters(),
    },
    X,
    y,
)
print(comparison.best.name)
print(comparison.format_table())
```

Keep correlated samples together, or respect temporal order, by choosing a splitter:

```python
from classreg.classification import (
    ClassificationEvaluator,
    StratifiedGroupKFoldSplitter,
)
from classreg.regression import RegressionEvaluator, TimeSeriesSplitter

by_subject = ClassificationEvaluator(
    splitter=StratifiedGroupKFoldSplitter(groups=subject_ids)
)
forward = RegressionEvaluator(splitter=TimeSeriesSplitter(n_splits=5, gap=1))
```

## Notes

- Scaling is part of the estimator pipeline, and every fold builds a fresh processor, so test samples never influence preprocessing. Scale-sensitive methods default to `FeatureScaling.STANDARD`.
- Classification labels must be integers, booleans, or strings; regression targets must be finite numbers. Both must have shape `(n_samples,)`, so multi-label and multi-output targets are not supported.
- Metrics that cannot be computed are `None`, not a misleading number: for example `log_loss` for SVC, which produces margins rather than probabilities, `r2` for constant targets, and `mape` when a target is zero.
- Macro-averaged ROC AUC and average precision are one-vs-rest, also for binary problems; for the score of the positive class, read `metrics.per_class`.
- Evaluating on a label the classifier was not fitted on raises `ValueError`, as does a split that trains on a single class.
- `MetricSummary.std` is `None` for a single holdout split: the variability is unknown, not zero.
- Means are only compared on the same folds. When a metric is undefined in only some folds, summaries average the remaining folds and report their coverage, such as `(1/5 folds)`. `compare` refuses to rank by such a metric, `get_baseline_improvement` raises `ValueError` when the model and the baseline define it on different folds, and the comparison table omits such a baseline.
- `compare` rejects a metric that a candidate's kind of scores can never define, such as `log_loss` for SVC, before any training.
- Parameter and splitter objects reject NaN and non-integer values at construction, with `ValueError` and `TypeError` respectively.

## Development

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run basedpyright src tests
uv run pytest
```

The same checks run in CI on every push to `main` and every pull request, along with the tests on the minimum supported dependency versions.

## License

Released under the [MIT License](LICENSE).
