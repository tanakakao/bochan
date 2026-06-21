# bochan

`bochan` is a BoTorch-oriented experimental library for Bayesian optimization, active learning, and level-set estimation across Gaussian regression, binary and multiclass classification, ordinal regression, and hybrid multi-output models.

## Main use cases

- Bayesian optimization with continuous, categorical, and mixed variables
- active learning and level-set estimation
- constrained and multi-objective optimization
- robust optimization with input perturbation and risk aggregation
- high-dimensional modeling with SAAS, PCA, REMBO, DeepGP, and Deep Kernel GP
- independent multi-output and correlated multi-task modeling
- `ask()` / `tell()` loops through `BochanStudy`
- pandas / numpy / CSV workflows through `bochan.tabular`
- HTTP / JSON serving through FastAPI

## Installation

```bash
pip install -e .
pip install -e ".[dev,api,tabular,visualization,evo]"
```

## Package layout

```text
src/bochan/
├── api/
├── acquisition/
├── fit/
├── models/
│   ├── components/
│   ├── transforms/
│   ├── regression/
│   ├── classification/
│   ├── ordinal/
│   └── hybrid/
├── tabular/
├── visualization/
└── serving/fastapi/
```

## Model families

| Family | Purpose |
|---|---|
| `regression/gaussian` | Continuous-output Gaussian GP models |
| `regression/non_gaussian` | Poisson, Beta, Gamma, and Negative Binomial models |
| `classification/binary` | Binary GP classification |
| `classification/multiclass` | Multiclass GP classification |
| `ordinal` | Ordered-label GP models |
| `hybrid` | Heterogeneous multi-output wrappers |

Detailed model documentation:

- `src/bochan/models/regression/gaussian/README.md`
- `src/bochan/models/classification/binary/README.md`
- `src/bochan/models/classification/multiclass/README.md`
- `src/bochan/models/ordinal/README.md`

## Independent multi-output and correlated multi-task

### Independent multi-output

```text
train_X: [n, d]
train_Y: [n, m]
```

Outputs are modeled by independent submodels.

### Task-feature multi-task

```text
train_X: [N, d + 1]
train_Y: [N] or [N, 1]
```

An explicit task-id column is included. Tasks may be observed at different input locations and missing observations are allowed.

```text
K((x,t),(x',t')) = K_data(x,x') * K_task(t,t')
```

### Kronecker multi-task

```text
train_X: [n, d]
train_Y: [n, m]
```

All tasks must be observed at the same input points.

| Data condition | Recommended model |
|---|---|
| Different locations or counts per task | `multitask` |
| Missing task observations | `multitask` |
| Complete block design | `kronecker` |
| No task correlation required | independent multi-output |

## Mixed multi-task models

| Task | `model_type="multitask"` | `model_type="kronecker"` |
|---|---|---|
| regression | `MixedMultiTaskGP` | `MixedKroneckerMultiTaskGP` |
| binary | `MultiTaskBinaryClassificationMixedGPModel` | `KroneckerMultiTaskBinaryClassificationMixedGPModel` |
| multiclass | `MultiTaskMulticlassClassificationMixedGPModel` | `KroneckerMultiTaskMulticlassClassificationMixedGPModel` |
| ordinal | `MultiTaskOrdinalMixedGPModel` | `KroneckerMultiTaskOrdinalMixedGPModel` |

The task-id column is modeled by an `IndexKernel`. It must not be listed in `cat_dims` and must not be normalized or perturbed.

See `docs/mixed_task_feature_multitask_models.md` for the detailed contract.

## High-level model registry

`task_type` values:

```text
regression, multi_objective, binary, multiclass, ordinal, hybrid
```

Common `model_type` values:

```text
base, deepgp, deepkernel, deepgpdeepkernel,
saas, pca, rembo, rrp, hetero,
kronecker, multitask
```

`kronecker` and `multitask` are registered for mixed regression, multi-objective regression, binary, multiclass, and ordinal models.

## Minimal tensor API

```python
import torch

from bochan.api import (
    AcquisitionConfig,
    BayesianOptimizer,
    FitConfig,
    ModelConfig,
    OptimizeConfig,
)

train_X = torch.rand(40, 2, dtype=torch.double)
train_Y = torch.sin(train_X[:, :1] * 6.28)
bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

bo = BayesianOptimizer(
    model_config=ModelConfig(task_type="regression", model_type="base"),
    fit_config=FitConfig(maxiter=128),
    bounds=bounds,
)
bo.fit(train_X, train_Y)

candidates, value = bo.candidate(
    acq_config=AcquisitionConfig(
        name="EI",
        acqf_kwargs={"best_f": train_Y.max()},
    ),
    opt_config=OptimizeConfig(q=3, num_restarts=10, raw_samples=128),
)
```

## Mixed task-feature API example

```python
from bochan.api import InputTransformConfig

model_config = ModelConfig(
    task_type="binary",
    input_type="mixed",
    model_type="multitask",
    cat_dims=[2],
    model_kwargs={
        "num_tasks": 2,
        "task_feature": 1,
        "rank": 2,
        "num_inducing_points": 32,
    },
    input_transform_config=InputTransformConfig(
        normalize=True,
        categorical_idx=[1, 2],
    ),
)
```

For binary, multiclass, and ordinal task-feature models, candidate tensors contain the task id. Fix the selected task and enumerate ordinary categories.

```python
opt_config = OptimizeConfig(
    optimizer="optimize_acqf_mixed",
    q=1,
    fixed_features={1: 1.0},
    fixed_features_list=[{2: 0.0}, {2: 1.0}],
)
```

### Gaussian prediction contract

`MixedMultiTaskGP` follows BoTorch `MultiTaskGP` behavior:

- training `train_X` includes the task-id column;
- prediction `X` excludes the task-id column;
- selected tasks are passed through `output_indices`.

```python
result = bo.predict(
    X_without_task_column,
    return_result=True,
    posterior_kwargs={"output_indices": [0, 1]},
)
```

## Acquisition and optimization

Contextual acquisition aliases include:

```text
EI, PI, UCB, KG
BALD, JointBALD, GreedyJointBALD
Entropy, Variance, NIPV, Margin
Straddle, ICU, BoundaryVariance
ClassEntropy, PoE, LevelSet
EHVI, NEHVI, NParEGO
```

Supported optimizer families include:

```text
optimize_acqf
optimize_acqf_mixed
evo / evo_mixed
torch / torch_mixed
nsgaii
```

## Other APIs

| API | Use when |
|---|---|
| `bochan.api.BayesianOptimizer` | Direct tensor-based fitting and candidate generation |
| `bochan.api.BochanStudy` | Optuna / Ax-style loops and `ask()` / `tell()` |
| `bochan.tabular.TabularBayesianOptimizer` | pandas, numpy, or CSV workflows |
| `bochan.serving.fastapi` | HTTP / JSON serving |

Documentation:

- `src/bochan/api/README.md`
- `src/bochan/api/STUDY_README.md`
- `src/bochan/tabular/README.md`
- `src/bochan/serving/fastapi/README.md`

## Development status

The codebase is under active development. API consistency and correct BoTorch-style behavior are prioritized over backward compatibility.
