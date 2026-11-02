# classreg

## Overview

Shared foundation for supervised learning. Layers depend only on those listed before them: shared types, constants and validators, `options/`, `dataset.py`, and `feature_scaling.py`; `evaluation/`, which is generic over the task; and the `classification/` and `regression/` task packages, which do not depend on each other.

## Components

| Component | Description |
|-----------|-------------|
| [types.py](./types.py) | Type aliases shared by both tasks. |
| [random_state.py](./random_state.py) | `DEFAULT_RANDOM_STATE`, the default seed of estimators and splitters. |
| [parameter_validator.py](./parameter_validator.py) | `ParameterValidator`, runtime checks for NaN and non-integer parameters and non-discrete group identifiers. |
| [dataset.py](./dataset.py) | `SupervisedDataset`, the abstract validated pair of features and targets. |
| [predictor.py](./predictor.py) | `PredictorLike`, the structural type of estimators that a pipeline wraps. |
| [feature_scaling.py](./feature_scaling.py) | `FeatureScaling` enum that builds the scaler and estimator pipeline. |
| [options/](./options/README.md) | Enums for estimator options that both tasks share. |
| [evaluation/](./evaluation/README.md) | Task-independent splitters, cross-validated evaluation, reports, and model comparison. |
| [classification/](./classification/README.md) | Classification methods, metrics, and evaluator. |
| [regression/](./regression/README.md) | Regression methods, metrics, and evaluator. |
