# Mixed task-feature multi-task models

This guide describes correlated multi-task models for mixed continuous and
categorical inputs. Training observations use long format with an explicit task
id. Unlike Kronecker models, tasks may have different input locations and
missing observations.

## 1. Data contracts

### Task-feature long format

```text
train_X: [N, d + 1]
train_Y: [N] or [N, 1]
```

Example:

```text
continuous_0 | task_id | category_0 | continuous_1
```

The covariance is

```text
K_mixed(data, data') * K_task(task_id, task_id')
```

The task-id column is modeled by an `IndexKernel`. It must not also be included
in `cat_dims`.

### Kronecker block design

```text
train_X: [n, d]
train_Y: [n, m]
```

All tasks must be observed at the same input points.

| Condition | Model type |
|---|---|
| Different input locations per task | `multitask` |
| Missing task observations | `multitask` |
| Complete block design | `kronecker` |
| No task correlation required | independent multi-output |

## 2. Models

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

Kronecker mixed variants:

```python
from bochan.models.regression.gaussian import MixedKroneckerMultiTaskGP
from bochan.models.classification.binary.base import (
    KroneckerMultiTaskBinaryClassificationMixedGPModel,
)
from bochan.models.classification.multiclass.base import (
    KroneckerMultiTaskMulticlassClassificationMixedGPModel,
)
from bochan.models.ordinal.base import KroneckerMultiTaskOrdinalMixedGPModel
```

## 3. Input transforms

Normalize continuous columns only. Both categorical columns and task ids must
remain unchanged.

```python
from botorch.models.transforms.input import Normalize

input_transform = Normalize(
    d=4,
    indices=[0, 3],
)
```

High-level API:

```python
from bochan.api import InputTransformConfig

InputTransformConfig(
    normalize=True,
    categorical_idx=[1, 2],  # task id and ordinary category
)
```

The task feature cannot also be present in `cat_dims`.

## 4. Binary example

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

Candidate tensors contain the task-id column.

## 5. Multiclass example

```python
model = MultiTaskMulticlassClassificationMixedGPModel(
    train_X=train_X,
    train_Y=train_Y,
    cat_dims=[2],
    num_classes=3,
    num_tasks=3,
    task_feature=1,
    rank=2,
    input_transform=input_transform,
)
```

Each class logit has a task covariance matrix.

```python
task_covar = model.task_covar_matrix
print(task_covar.shape)  # [num_classes, num_tasks, num_tasks]
```

All tasks must share the same class set.

## 6. Ordinal example

```python
model = MultiTaskOrdinalMixedGPModel(
    train_X=train_X,
    train_Y=train_Y,
    cat_dims=[2],
    num_classes=4,
    num_tasks=3,
    task_feature=1,
    rank=2,
    input_transform=input_transform,
)
```

All tasks share the ordinal class definition and ordered-logit cutpoints.

## 7. Gaussian example

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

BoTorch `MultiTaskGP.posterior` receives non-task data columns and selected
output tasks.

```python
posterior = model.posterior(
    X_without_task_column,
    output_indices=[0, 1, 2],
)
```

This differs from classification and ordinal task-feature models, whose
candidate and prediction tensors include the task-id column.

## 8. Candidate optimization

For binary, multiclass, and ordinal task-feature models, fix the selected task
and enumerate ordinary categories.

```python
from botorch.optim import optimize_acqf_mixed

fixed_features_list = [
    {task_feature: 1.0, category_feature: 0.0},
    {task_feature: 1.0, category_feature: 1.0},
]

candidates, value = optimize_acqf_mixed(
    acq_function=acq_function,
    bounds=bounds,
    q=1,
    num_restarts=10,
    raw_samples=256,
    fixed_features_list=fixed_features_list,
)
```

High-level API:

```python
from bochan.api import OptimizeConfig

OptimizeConfig(
    optimizer="optimize_acqf_mixed",
    q=1,
    fixed_features={1: 1.0},
    fixed_features_list=[
        {2: 0.0},
        {2: 1.0},
    ],
)
```

## 9. High-level registry

Use `model_type="multitask"` for long-format task-feature models.

```python
from bochan.api import ModelConfig

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

Use `model_type="kronecker"` for complete block design.

```python
ModelConfig(
    task_type="binary",
    input_type="mixed",
    model_type="kronecker",
    cat_dims=[2],
    model_kwargs={"rank": 2},
)
```

The mixed registry supports both keys for regression, multi-objective
regression, binary, multiclass, and ordinal tasks.

## 10. FastAPI

Model creation uses the same configuration fields.

```json
{
  "model_config": {
    "task_type": "binary",
    "input_type": "mixed",
    "model_type": "multitask",
    "cat_dims": [2],
    "model_kwargs": {
      "num_tasks": 2,
      "task_feature": 1,
      "rank": 2
    },
    "input_transform_config": {
      "normalize": true,
      "categorical_idx": [1, 2]
    }
  },
  "fit_config": {
    "num_epochs": 300,
    "lr": 0.01
  },
  "train_X": [
    [0.05, 0, 0],
    [0.20, 0, 1],
    [0.10, 1, 1],
    [0.45, 1, 0]
  ],
  "train_Y": [0, 0, 0, 1],
  "bounds": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
}
```

Candidate request:

```json
{
  "acq_config": {"name": "Entropy"},
  "opt_config": {
    "optimizer": "optimize_acqf_mixed",
    "q": 1,
    "fixed_features": {"1": 1.0},
    "fixed_features_list": [
      {"2": 0.0},
      {"2": 1.0}
    ]
  }
}
```

Gaussian prediction uses task-free inputs and `posterior_kwargs.output_indices`.

```json
{
  "X": [[0.25, 1], [0.75, 0]],
  "return_type": "mean_variance",
  "posterior_kwargs": {
    "output_indices": [0, 1]
  }
}
```

## 11. Conditioning

Task-feature models append observations in long format.

```python
updated = model.condition_on_observations(
    X=X_new_with_task_id,
    Y=Y_new,
)
```

Kronecker models require all tasks in each new row.

```python
updated = model.condition_on_observations(
    X=X_new,
    Y=Y_new_all_tasks,
)
```

Variational classification and ordinal conditioning reconstructs the model and
may require refitting.

## 12. Task covariance interpretation

Task covariance is a latent GP covariance, not raw Pearson correlation, label
agreement, or a confusion matrix.

Binary and ordinal:

```python
task_covar = model.task_covar_matrix  # [m, m]
```

Multiclass:

```python
task_covar = model.task_covar_matrix  # [C, m, m]
```

Convert covariance to correlation with diagonal standardization before comparing
relative task relationships.

## 13. Related documentation and tests

```text
src/bochan/api/README.md
src/bochan/serving/fastapi/README.md
src/bochan/models/regression/gaussian/README.md
src/bochan/models/classification/binary/README.md
src/bochan/models/classification/multiclass/README.md
src/bochan/models/ordinal/README.md
tests/test_mixed_task_feature_multitask_models.py
tests/test_mixed_task_feature_multitask_registry.py
```
