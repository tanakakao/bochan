# bochan

`bochan` is a BoTorch-oriented experimental library for Bayesian optimization,
active learning, and level-set estimation across Gaussian regression, binary and
multiclass classification, ordinal regression, and hybrid multi-output models.

The project prioritizes consistent tensor shapes, BoTorch-like posterior APIs,
and reuse of acquisitions, objectives, candidate repair, and optimization
backends across model families.

## Main use cases

- Bayesian optimization with continuous, categorical, and mixed variables
- active learning and level-set estimation
- constrained and multi-objective optimization
- robust optimization with input perturbation and risk aggregation
- high-dimensional modeling with SAAS, PCA, REMBO, DeepGP, and Deep Kernel GP
- independent multi-output and correlated multi-task modeling
- Optuna / Ax-style `ask()` / `tell()` loops through `BochanStudy`
- pandas / numpy / CSV workflows through `bochan.tabular`
- HTTP / JSON serving through FastAPI

## Installation

```bash
pip install -e .
```

Common development setup:

```bash
pip install -e ".[dev,api,tabular,visualization,evo]"
```

All optional extras:

```bash
pip install -e ".[all]"
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
│   │   ├── gaussian/
│   │   └── non_gaussian/
│   ├── classification/
│   │   ├── binary/
│   │   └── multiclass/
│   ├── ordinal/
│   └── hybrid/
├── tabular/
├── visualization/
└── serving/
    └── fastapi/
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
| `components` | Shared likelihoods, posteriors, kernels, and transforms |

Detailed model documentation:

- `src/bochan/models/regression/gaussian/README.md`
- `src/bochan/models/classification/binary/README.md`
- `src/bochan/models/classification/multiclass/README.md`
- `src/bochan/models/ordinal/README.md`

## Independent multi-output and correlated multi-task

### Independent multi-output

Outputs are modeled by independent submodels.

```text
train_X: [n, d]
train_Y: [n, m]
```

### Task-feature multi-task

An explicit task-id column is included in long-format training data. Tasks may
be observed at different input locations and missing task observations are
allowed.

```text
train_X: [N, d + 1]
train_Y: [N] or [N, 1]
```

```text
K((x,t),(x',t')) = K_data(x,x') * K_task(t,t')
```

### Kronecker multi-task

All tasks must be observed at the same input points.

```text
train_X: [n, d]
train_Y: [n, m]
```

| Data condition | Recommended model |
|---|---|
| Different locations or observation counts per task | task-feature `multitask` |
| Missing task observations | task-feature `multitask` |
| Complete block design | `kronecker` |
| No task correlation required | independent multi-output |

## Mixed multi-task models

The mixed registry includes correlated models for continuous and categorical
data.

| Task | `model_type="multitask"` | `model_type="kronecker"` |
|---|---|---|
| regression | `MixedMultiTaskGP` | `MixedKroneckerMultiTaskGP` |
| binary | `MultiTaskBinaryClassificationMixedGPModel` | `KroneckerMultiTaskBinaryClassificationMixedGPModel` |
| multiclass | `MultiTaskMulticlassClassificationMixedGPModel` | `KroneckerMultiTaskMulticlassClassificationMixedGPModel` |
| ordinal | `MultiTaskOrdinalMixedGPModel` | `KroneckerMultiTaskOrdinalMixedGPModel` |

The task-id column is modeled by an `IndexKernel`. It must not also be listed in
`cat_dims`, and it must not be normalized or perturbed.

See `docs/mixed_task_feature_multitask_models.md` for the detailed contract.

## High-level model registry

The API uses these `task_type` values:

```text
regression
multi_objective
binary
multiclass
ordinal
hybrid
```

Common `model_type` values:

```text
base
deepgp
deepkernel
deepgpdeepkernel
saas
pca
rembo
rrp
hetero
kronecker
multitask
```

`kronecker` and `multitask` are registered for mixed regression,
multi-objective regression, binary, multiclass, and ordinal models.

When `cat_dims` is provided and `input_type` is omitted, the API infers
`input_type="mixed"`.

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
    model_config=ModelConfig(
        task_type="regression",
        model_type="base",
    ),
    fit_config=FitConfig(maxiter=128),
    bounds=bounds,
)
bo.fit(train_X, train_Y)

candidates, value = bo.candidate(
    acq_config=AcquisitionConfig(
        name="EI",
        acqf_kwargs={"best_f": train_Y.max()},
    ),
    opt_config=OptimizeConfig(
        q=3,
        num_restarts=10,
        raw_samples=128,
    ),
)
```

## Mixed task-feature API example

The following columns are used:

```text
continuous | task_id | category
```

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

For binary, multiclass, and ordinal task-feature models, candidate tensors
contain the task id. Fix the selected task and enumerate ordinary categories.

```python
opt_config = OptimizeConfig(
    optimizer="optimize_acqf_mixed",
    q=1,
    fixed_features={1: 1.0},
    fixed_features_list=[
        {2: 0.0},
        {2: 1.0},
    ],
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

## Acquisition functions

Acquisitions are organized by task under active learning, level-set estimation,
and Bayesian optimization modules.

Contextual aliases include:

```text
EI, PI, UCB, KG
BALD, JointBALD, GreedyJointBALD
Entropy, PredictiveEntropy
Variance, NIPV, Margin
Straddle, ICU, BoundaryVariance
ClassEntropy, PoE, LevelSet
EHVI, NEHVI, NParEGO
```

Multiclass BO usually requires acquisition-specific arguments such as
`target_class`, `threshold`, and `best_f`.

```python
AcquisitionConfig(
    name="EI",
    acqf_kwargs={
        "target_class": 2,
        "best_f": 0.70,
    },
)
```

## Input perturbation and risk aggregation

```python
from bochan.api import ObjectiveConfig

model_config = ModelConfig(
    task_type="regression",
    model_type="base",
    input_transform_config=InputTransformConfig(
        normalize=True,
        perturbation=True,
        n_w=8,
        std=0.1,
        bounds=bounds,
    ),
)

acq_config = AcquisitionConfig(
    name="EI",
    objective_config=ObjectiveConfig(
        mode="scalar",
        output=0,
        n_w=8,
        risk_type="cvar",
        alpha=0.8,
    ),
    acqf_kwargs={"best_f": train_Y.max()},
)
```

`InputTransformConfig(n_w=...)` and `ObjectiveConfig(n_w=...)` should match when
expanded perturbation samples must be aggregated back to q candidates.

## Candidate optimization and repair

Supported optimizer names include:

```text
optimize_acqf
optimize_acqf_mixed
evo / evo_mixed
torch / torch_mixed
nsgaii
```

Candidate repair:

```python
from bochan.api import CandidateRepairConfig

opt_config = OptimizeConfig(
    q=3,
    repair_config=CandidateRepairConfig(
        bounds=bounds,
        numeric_indices=[0, 1, 2, 3],
        steps=[0.1, 0.1, 0.1, 0.1],
        comp_idx=[0, 1, 2, 3],
        k=2,
        inequality_constraints=ineq_constraints,
        inequality_sense="le",
    ),
)
```

Notes:

- `steps=None` disables grid rounding.
- `comp_idx=None` or `[]` disables k-sparse support selection.
- `support_selection` can be `topk` or `sample`.

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

The codebase is under active development. Backward compatibility is not the
main priority yet; API consistency and correct BoTorch-style behavior are
prioritized.
