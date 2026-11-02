# splitters

## Overview

Concrete `DatasetSplitter` strategies that do not depend on the task. Choose by how samples relate to each other: independent samples suit k-fold or holdout, correlated samples need group folds, and temporal data needs forward chaining. Classification adds stratified variants in `classification/splitters/`.

## Components

| Component | Description |
|-----------|-------------|
| [k_fold.py](./k_fold.py) | `KFoldSplitter`, the regression default. |
| [holdout.py](./holdout.py) | `HoldoutSplitter`, a single partition. |
| [group_k_fold.py](./group_k_fold.py) | `GroupKFoldSplitter`, which keeps each group on one side. |
| [time_series.py](./time_series.py) | `TimeSeriesSplitter`, which always tests on later samples. |
