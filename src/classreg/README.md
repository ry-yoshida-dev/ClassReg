# classreg

## Overview

Shared foundation for supervised learning. Layers depend only on those listed before them: shared types, constants and validators, `options/`, and `dataset.py`; `evaluation/`, which is generic over the task; and the `classification/` and `regression/` task packages, which do not depend on each other.

## Components

| Component | Description |
|-----------|-------------|
| [types.py](./types.py) | Type aliases shared by both tasks. |
| [random_state.py](./random_state.py) | `DEFAULT_RANDOM_STATE`, the default seed of estimators. |
| [parameter_validator.py](./parameter_validator.py) | `ParameterValidator`, runtime checks for NaN and non-integer parameters. |
| [dataset.py](./dataset.py) | `SupervisedDataset`, the abstract validated pair of features and targets. |
| [options/](./options/README.md) | Enums for estimator options that both tasks share. |
| [evaluation/](./evaluation/README.md) | Cross-validated evaluation, reports, and model comparison. |
| [classification/](./classification/README.md) | Classification methods, metrics, and evaluator. |
| [regression/](./regression/README.md) | Regression methods, metrics, and evaluator. |
