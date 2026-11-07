# evaluation

## Overview

Task-independent estimation of generalization performance. `SupervisedEvaluator` measures a model and a baseline on the same splits, and compares model configurations by one metric.

## Components

| Component | Description |
|-----------|-------------|
| [evaluator.py](./evaluator.py) | `SupervisedEvaluator`, with `evaluate` and `compare`; task packages provide the concrete subclasses. |
| [report.py](./report.py) | `EvaluationReport`, a model next to its baseline. |
| [comparison.py](./comparison.py) | `ModelComparison`, candidates ranked by one metric. |
| [ranked_model.py](./ranked_model.py) | `RankedModel`, one candidate in a comparison. |
| [model_evaluation.py](./model_evaluation.py) | `ModelEvaluation`, one configuration across all folds. |
| [fold.py](./fold.py) | `FoldEvaluation`, train and test metrics of one split. |
| [metric_summary.py](./metric_summary.py) | `MetricSummary`, the aggregate of a metric over folds. |
| [partition.py](./partition.py) | `EvaluationPartition` enum: train or test. |
| [text_table.py](./text_table.py) | `TextTable`, plain text rendering of reports. |
| [protocols/](./protocols/README.md) | Structural types that connect evaluation to the task packages. |
