# splitters

## Overview

Splitting strategies that preserve class proportions.

## Components

| Component | Description |
|-----------|-------------|
| [stratified_k_fold.py](./stratified_k_fold.py) | `StratifiedKFoldSplitter`, the classification default. |
| [stratified_holdout.py](./stratified_holdout.py) | `StratifiedHoldoutSplitter`, a single partition. |
| [stratified_group_k_fold.py](./stratified_group_k_fold.py) | `StratifiedGroupKFoldSplitter`, which keeps each group on one side. |
