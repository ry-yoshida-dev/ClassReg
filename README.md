# ClassReg

## Overview

ClassReg (`classreg`) wraps scikit-learn classifiers and regressors behind a small, consistent API and adds a performance evaluation layer. Parameter dataclasses validate their fields and build processors with `fit`, `predict`, and `evaluate`; evaluators estimate generalization performance by cross-validation against a feature-independent baseline and compare configurations on identical splits.

| Task | Package | Methods |
| --- | --- | --- |
| Classification | `classreg.classification` | [30 methods](src/classreg/classification/methods/README.md), from logistic regression and SVC to tree ensembles, naive Bayes, and stacking |
| Regression | `classreg.regression` | [38 methods](src/classreg/regression/methods/README.md), from linear and generalized linear models to SVR, tree ensembles, and stacking |

Each task package exports everything needed for that task; the top-level `classreg` package exports only the shared components. Scaling comes from [DataScaler](https://github.com/ry-yoshida-dev/DataScaler) and splitting from [DataSplitter](https://github.com/ry-yoshida-dev/DataSplitter); their main classes are re-exported here.

## Installation

```bash
pip install .
```

## Usage

Train a classifier and measure it on separate test data:

```python
from classreg.classification import RandomForestParameters

processor = RandomForestParameters(n_estimators=200).build_processor()
processor.fit(X_train, y_train)
metrics = processor.evaluate(X_test, y_test)
print(metrics.macro_f1, metrics.roc_auc)
```

Cross-validate a classifier against the baseline:

```python
from classreg.classification import (
    ClassificationEvaluator,
    ClassificationMetric,
    LogisticRegressionParameters,
)

report = ClassificationEvaluator().evaluate(LogisticRegressionParameters(), X, y)
print(report.format_table())
print(report.get_baseline_improvement(ClassificationMetric.MACRO_F1))
```

Rank regressors on identical splits, choosing a splitter that respects temporal order:

```python
from classreg.regression import (
    LassoParameters,
    RegressionEvaluator,
    RidgeParameters,
    TimeSeriesSplitter,
)

evaluator = RegressionEvaluator(splitter=TimeSeriesSplitter(n_splits=5))
comparison = evaluator.compare(
    {"ridge": RidgeParameters(), "lasso": LassoParameters(alpha=0.1)}, X, y
)
print(comparison.best.name)
```
