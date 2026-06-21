# Mixed task-feature multi-task models

These models use long-format observations with an explicit task-id column.
Unlike Kronecker models, tasks may be observed at different input locations and
missing task observations are allowed.

```text
train_X: [N, d + 1]
train_Y: [N] or [N, 1]
```

Example column layout:

```text
continuous_0 | task_id | category_0 | continuous_1
```

The covariance is

```text
K_mixed(data, data') * K_task(task_id, task_id')
```

The task-id column is modeled by an `IndexKernel`; it must not also be included
in `cat_dims`.

## Models

```python
from bochan.models.regression.gaussian import MixedMultiTaskGP
from bochan.models.classification.binary.base import (
    MultiTaskBinaryClassificationMixedGPModel,
)
from bochan.models.classification.multiclass.base import (
    MultiTaskMulticlassClassificationGPModel,
    MultiTaskMulticlassClassificationMixedGPModel,
)
from bochan.models.ordinal.base import MultiTaskOrdinalMixedGPModel
```

## Input transform

Normalize continuous columns only. Both categorical columns and the task-id
column must remain unchanged.

```python
from botorch.models.transforms.input import Normalize

input_transform = Normalize(
    d=4,
    indices=[0, 3],
)
```

## Binary example

```python
model = MultiTaskBinaryClassificationMixedGPModel(
    train_X=train_X,
    train_Y=train_Y,
    cat_dims=[2],
    num_tasks=3,
    task_feature=1,
    rank=2,
    input_transform=input_transform,
)
```

## Gaussian example

```python
model = MixedMultiTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    cat_dims=[2],
    task_feature=1,
    rank=2,
    input_transform=input_transform,
)
```

BoTorch `MultiTaskGP.posterior` receives the non-task input columns and selected
output tasks:

```python
posterior = model.posterior(
    X_without_task_column,
    output_indices=[0, 1, 2],
)
```

## Candidate optimization

For classification and ordinal task-feature models, pass the task-id column in
candidate tensors and fix it during optimization. Enumerate ordinary categories
with `fixed_features_list`.

```python
fixed_features_list = [
    {task_feature: 1.0, category_feature: 0.0},
    {task_feature: 1.0, category_feature: 1.0},
]
```

## High-level registry

Use `model_type="multitask"` for mixed task-feature models.

```python
ModelConfig(
    task_type="binary",
    input_type="mixed",
    model_type="multitask",
    cat_dims=[2],
    model_kwargs={
        "num_tasks": 3,
        "task_feature": 1,
        "rank": 2,
    },
)
```

`model_type="kronecker"` remains the block-design alternative.
