# Tests

## Overview

Unit tests for the `classreg` package, organized by task package, with a test of how the shared components connect them.

## Components

| Component | Description |
| --- | --- |
| [`test_package.py`](test_package.py) | Tests for the public API layout and shared components across both tasks. |
| [`classification/`](classification/README.md) | Tests for `classreg.classification`. |
| [`regression/`](regression/README.md) | Tests for `classreg.regression`. |

## Examples

```bash
python -m pytest tests/ -v
```
