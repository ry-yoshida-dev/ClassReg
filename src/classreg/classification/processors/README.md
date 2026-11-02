# processors

## Overview

Concrete `ClassificationProcessor` implementations. They differ by what a classifier produces per class, so asking for an output a method lacks is rejected by the type checker.

## Components

| Component | Description |
|-----------|-------------|
| [probabilistic.py](./probabilistic.py) | `ProbabilisticClassificationProcessor`, with `predict_proba`, for all methods except those with margins. |
| [margin.py](./margin.py) | `MarginClassificationProcessor`, with `decision_function`, for SVC, nu SVC, linear SVC, stochastic gradient descent, perceptron, and ridge classifier. |
| [protocols/](./protocols/README.md) | Structural types for the wrapped estimators. |
