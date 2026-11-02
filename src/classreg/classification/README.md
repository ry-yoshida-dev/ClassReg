# classification

## Overview

Classification on top of the shared foundation. `methods/` holds parameter classes that build processors, `metrics/` computes performance from labels and scores, and `ClassificationEvaluator` adds stratified defaults to the shared evaluator.

## Components

| Component | Description |
|-----------|-------------|
| [types.py](./types.py) | Type aliases for labels. |
| [method.py](./method.py) | `ClassificationMethod` enum of available methods. |
| [dataset.py](./dataset.py) | `ClassificationDataset`, the validated pair of features and labels. |
| [class_weighting.py](./class_weighting.py) | `ClassWeighting` enum for imbalanced data. |
| [tree_criterion.py](./tree_criterion.py) | `TreeCriterion` enum, shared by decision trees and tree ensembles. |
| [parameter.py](./parameter.py) | Abstract base class for classification parameters. |
| [processor.py](./processor.py) | Abstract base class for processors; implements validation, fitted state, `predict`, and `evaluate`. |
| [evaluator.py](./evaluator.py) | `ClassificationEvaluator`, with stratified folds and a class prior baseline by default. |
| [processors/](./processors/README.md) | Concrete processors, split by the type of per-class scores. |
| [methods/](./methods/README.md) | Per-method parameter classes and enums. |
| [metrics/](./metrics/README.md) | Performance metrics computed from labels and scores. |
| [splitters/](./splitters/README.md) | Stratified splitting strategies. |
