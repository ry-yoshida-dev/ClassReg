# regression

## Overview

Unit tests for the `classreg.regression` package: data validation, metrics checked by hand and against scikit-learn, parameter validation and forwarding, processor behavior for all methods, splitters, and end-to-end evaluation.

## Components

| File | Description |
| --- | --- |
| [`helpers.py`](helpers.py) | Shared synthetic datasets. |
| [`test_dataset.py`](test_dataset.py) | Tests for `RegressionDataset` validation and subsetting. |
| [`test_metrics.py`](test_metrics.py) | Tests for `MetricValues`, including undefined metrics. |
| [`test_parameters.py`](test_parameters.py) | Tests for parameter validation, forwarding to estimators, and `FeatureScaling`. |
| [`test_parameter_validation.py`](test_parameter_validation.py) | Tests for NaN and non-integer rejection in parameters and splitters. |
| [`test_processors.py`](test_processors.py) | Tests for fit, predict, and evaluate across all methods. |
| [`test_extended_methods.py`](test_extended_methods.py) | Tests for sparse, Bayesian, kernel, generalized linear, and isotonic models, and for ensembles. |
| [`test_splitters.py`](test_splitters.py) | Tests for partition invariants of every splitter. |
| [`test_evaluator.py`](test_evaluator.py) | Tests for `RegressionEvaluator`, reports, comparisons, and absence of test-to-train leakage. |

## Examples

```bash
python -m pytest tests/regression/ -v
```
