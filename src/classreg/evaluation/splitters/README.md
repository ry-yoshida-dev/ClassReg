# splitters

## Overview

Concrete `DatasetSplitter` strategies that do not depend on the task. Choose by how samples relate to each other: independent samples suit k-fold, holdout, or their repeated forms, very small datasets suit leave-one-out, correlated samples need group folds or their leave-out and repeated forms, partitions fixed in advance use predefined folds, and temporal data needs forward chaining. Classification adds stratified variants in `classification/splitters/`.

## Components

| Component | Description |
|-----------|-------------|
| [k_fold.py](./k_fold.py) | `KFoldSplitter`, the regression default. |
| [holdout.py](./holdout.py) | `HoldoutSplitter`, a single partition. |
| [group_k_fold.py](./group_k_fold.py) | `GroupKFoldSplitter`, which keeps each group on one side. |
| [repeated_k_fold.py](./repeated_k_fold.py) | `RepeatedKFoldSplitter`, k-fold repeated with new shuffles. |
| [repeated_holdout.py](./repeated_holdout.py) | `RepeatedHoldoutSplitter`, independently drawn random partitions. |
| [leave_one_out.py](./leave_one_out.py) | `LeaveOneOutSplitter`, which tests on each sample in turn. |
| [leave_p_out.py](./leave_p_out.py) | `LeavePOutSplitter`, which tests on every combination of samples. |
| [leave_one_group_out.py](./leave_one_group_out.py) | `LeaveOneGroupOutSplitter`, which tests on each group in turn. |
| [leave_p_groups_out.py](./leave_p_groups_out.py) | `LeavePGroupsOutSplitter`, which tests on every combination of groups. |
| [repeated_group_holdout.py](./repeated_group_holdout.py) | `RepeatedGroupHoldoutSplitter`, random partitions that keep each group on one side. |
| [predefined.py](./predefined.py) | `PredefinedSplitter`, partitions fixed in advance. |
| [time_series.py](./time_series.py) | `TimeSeriesSplitter`, which always tests on later samples. |
