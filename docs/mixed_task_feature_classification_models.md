# Mixed task-feature classification models

This document covers long-format mixed-input multi-task models for binary,
multiclass, and ordinal targets.

## Data contract

The task id is stored in one explicit input column.

```text
train_X: [N, d + 1]
train_Y: [N] or [N, 1]
```

Example column layout:

```text
continuous_0 | task_id | category_0
```

The covariance is

```text
K_mixed(data, data') * K_task(task_id, task_id')
```

Unlike Kronecker models, tasks may be observed at different input locations and
missing task observations are allowed.

## Models

```python
from bochan.models.classification.binary.base import (
    MultiTaskBinaryClassificationMixedGPModel,
)
from bochan.models.classification.multiclass import (
    MultiTaskMulticlassClassificationMixedGPModel,
)
from bochan.models.ordinal.base import MultiTaskOrdinalMixedGPModel
```

The continuous-input multiclass counterpart is also available:

```python
from bochan.models.classification.multiclass import (
    MultiTaskMulticlassClassificationGPModel,
)
```

## Input transforms

The task-id and categorical columns must not be normalized or perturbed.

```python
from botorch.models.transforms.input import Normalize

input_transform = Normalize(
    d=3,
    indices=[0],
)
```

The task feature must not also be listed in `cat_dims`.

## Binary example

```python
model = MultiTaskBinaryClassificationMixedGPModel(
    train_X=train_X,
    train_Y=train_Y,
    cat_dims=[2],
    num_tasks=2,
    task_feature=1,
    rank=2,
    input_transform=input_transform,
    num_inducing_points=32,
)
```

## Multiclass example

```python
model = MultiTaskMulticlassClassificationMixedGPModel(
    train_X=train_X,
    train_Y=train_Y,
    cat_dims=[2],
    num_classes=3,
    num_tasks=2,
    task_feature=1,
    rank=2,
    input_transform=input_transform,
    num_inducing_points=32,
)
```

Multiclass models learn one task covariance matrix per class logit.

```python
print(model.task_covar_matrix.shape)  # [num_classes, num_tasks, num_tasks]
```

## Ordinal example

```python
model = MultiTaskOrdinalMixedGPModel(
    train_X=train_X,
    train_Y=train_Y,
    cat_dims=[2],
    num_classes=4,
    num_tasks=2,
    task_feature=1,
    rank=2,
    input_transform=input_transform,
    num_inducing=32,
)
```

All ordinal tasks share the same class definition and ordered-logit cutpoints.

## Candidate optimization

Candidate tensors contain the task-id column. Fix the target task and enumerate
ordinary categorical values.

Low-level BoTorch example:

```python
from botorch.optim import optimize_acqf_mixed

fixed_features_list = [
    {1: 1.0, 2: 0.0},
    {1: 1.0, 2: 1.0},
]

candidates, value = optimize_acqf_mixed(
    acq_function=acquisition,
    bounds=bounds,
    q=1,
    num_restarts=10,
    raw_samples=256,
    fixed_features_list=fixed_features_list,
)
```

## High-level API

Use `model_type="multitask"` with mixed input.

```python
from bochan.api import InputTransformConfig, ModelConfig, OptimizeConfig

model_config = ModelConfig(
    task_type="binary",
    input_type="mixed",
    model_type="multitask",
    cat_dims=[2],
    model_kwargs={
        "num_tasks": 2,
        "task_feature": 1,
        "rank": 2,
    },
    input_transform_config=InputTransformConfig(
        normalize=True,
        categorical_idx=[1, 2],
    ),
)
```

At the high-level API, keep the user-facing optimizer name as
`optimizer="optimize_acqf"`. Because the fitted model has categorical dimensions,
`BayesianOptimizer` resolves the mixed optimizer automatically.

```python
opt_config = OptimizeConfig(
    optimizer="optimize_acqf",
    q=1,
    fixed_features={1: 1.0},
    fixed_features_list=[
        {2: 0.0},
        {2: 1.0},
    ],
)
```

The registry supports `model_type="multitask"` for:

```text
binary
multiclass
ordinal
```

Use `model_type="kronecker"` when every task is observed at the same input
locations and `train_Y` has shape `[n, m]`.

## Conditioning

Task-feature models append observations in the same long format.

```python
updated = model.condition_on_observations(
    X=X_new_with_task_id,
    Y=Y_new,
)
```

For ordinal models, `refit=False` reconstructs the model without additional
optimization.

## Task covariance interpretation

Task covariance describes latent GP coupling. It is not raw Pearson
correlation, label agreement, or a confusion matrix.

```python
# binary / ordinal
model.task_covar_matrix  # [num_tasks, num_tasks]

# multiclass
model.task_covar_matrix  # [num_classes, num_tasks, num_tasks]
```
