# regression

## Overview

Regression on top of the shared foundation. `methods/` holds parameter classes that build processors, `metrics/` computes performance from true and predicted values, and `RegressionEvaluator` adds k-fold defaults to the shared evaluator.

## Components

| Component | Description |
|-----------|-------------|
| [types.py](./types.py) | Type aliases for targets. |
| [method.py](./method.py) | `RegressionMethod` enum of available methods. |
| [dataset.py](./dataset.py) | `RegressionDataset`, the validated pair of features and targets. |
| [tree_criterion.py](./tree_criterion.py) | `TreeCriterion` enum, shared by decision trees and tree ensembles. |
| [generalized_linear_solver.py](./generalized_linear_solver.py) | `GeneralizedLinearSolver` enum, shared by the generalized linear methods. |
| [parameter.py](./parameter.py) | Abstract base class for regression parameters. |
| [processor.py](./processor.py) | `RegressionProcessor`, with validation, fitted state, `predict`, and `evaluate`. |
| [protocol.py](./protocol.py) | `RegressorLike`, the structural type of wrapped estimators. |
| [evaluator.py](./evaluator.py) | `RegressionEvaluator`, with k-fold splits and a mean baseline by default. |
| [methods/](./methods/README.md) | Per-method parameter classes and enums. |
| [metrics/](./metrics/README.md) | Performance metrics computed from true and predicted values. |
