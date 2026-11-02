# protocols

## Overview

Structural types through which `evaluation/` uses the task packages without importing them.

## Components

| Component | Description |
|-----------|-------------|
| [metric.py](./metric.py) | `MetricLike`, a metric enum tied to its metrics type. |
| [method.py](./method.py) | `MethodLike`, a method enum. |
| [processor.py](./processor.py) | `EvaluableProcessorLike`, a processor that can be fitted and measured. |
| [parameters.py](./parameters.py) | `EvaluableParametersLike`, parameters that build a processor. |
