# bochan

`bochan` is a BoTorch-oriented experimental library for Bayesian optimization,
active learning, and level-set estimation across multiple surrogate-model
families.

The project focuses on a consistent interface around Gaussian regression,
binary / multiclass classification, ordinal regression, hybrid multi-output
models, tabular data workflows, and ask/tell optimization loops. Acquisition
functions, objectives, candidate repair, and optimization backends are designed
to be reused across those model families.

The codebase is still under active development. Backward compatibility is not
the main priority yet; consistency of API design, tensor shapes, and
BoTorch-like behavior is prioritized.

---

## What this library is for

`bochan` is intended for workflows such as:

- Bayesian optimization with continuous, categorical, and mixed variables
- active learning for regression, binary classification, multiclass classification, ordinal regression, and non-Gaussian response models
- level-set estimation and boundary exploration
- constrained and multi-objective optimization
- robust optimization with input perturbation and risk aggregation
- high-dimensional optimization using PCA, REMBO, SAAS, or related wrappers
- model experimentation around GP, DeepGP, Deep Kernel GP, heteroscedastic GP, and robust relevance pursuit variants
- hybrid multi-output modeling across regression / binary / multiclass / ordinal outputs
- Optuna / Ax-style optimization loops through `BochanStudy`
- human-in-the-loop experiments using `ask()` / `tell()`
- simulation or Python-function optimization with automatic evaluation
- pandas / numpy / CSV based optimization through `bochan.tabular`
- HTTP / JSON model serving through FastAPI

The implementation is designed to stay close to BoTorch concepts where possible:

- model wrappers expose `posterior(X)`;
- latent-response models expose `latent_posterior(X)` when needed;
- acquisition functions operate on q-batch tensors;
- objective classes handle scalarization, probability / utility conversion, input-perturbation aggregation, and risk aggregation;
- BoTorch standard acquisition functions are reused when they already cover the required behavior.

---

## Installation

Core package:

```bash
pip install -e .
```

Common development setup:

```bash
pip install -e ".[dev,api,tabular,visualization,evo]"
```

FastAPI serving only:

```bash
pip install -e ".[api]"
```

Tabular DataFrame / numpy / CSV workflows:

```bash
pip install -e ".[tabular]"
```

Notebook examples:

```bash
pip install -e ".[notebook]"
```

Documentation tooling:

```bash
pip install -e ".[docs]"
```

All optional extras defined in `pyproject.toml`:

```bash
pip install -e ".[all]"
```

---

## Quick start: tabular DataFrame API

For pandas / CSV based optimization, use `bochan.tabular.TabularBayesianOptimizer`.

```python
import pandas as pd

from bochan.tabular import TabularBayesianOptimizer


df = pd.read_csv("data.csv")

bo = TabularBayesianOptimizer(
    task_type="regression",
    model_type="base",
    input_cols=["x1", "x2", "x3"],
    target_cols="y",
    bounds={
        "x1": [0.0, 1.0],
        "x2": [0.0, 1.0],
        "x3": [0.0, 1.0],
    },
)

bo.fit(df)

candidates_df, acq_value = bo.candidate(
    acq_name="EI",
    q=3,
    num_restarts=5,
    raw_samples=64,
)
```

The tabular wrapper also supports public API convenience fields such as `fit_beta`, `evo_method`, and `outcome_constraint_config`.

```python
candidates_df, acq_value = bo.candidate(
    acq_name="NIPV",
    q=5,
    optimizer="evo",
    evo_method="ga",
    numeric_indices=["x1", "x2", "x3"],
    steps={"x1": 0.1, "x2": 0.1, "x3": 0.1},
)
```

Detailed usage is documented in [`src/bochan/tabular/README.md`](src/bochan/tabular/README.md).

---

## Quick start: FastAPI serving

FastAPI serving exposes the high-level API over HTTP / JSON.

```bash
pip install -e ".[api]"
uvicorn bochan.serving.fastapi.app:app --reload
```

Create and fit a model:

```bash
curl -X POST http://127.0.0.1:8000/models \
  -H "Content-Type: application/json" \
  -d '{
    "model_config": {"task_type": "regression", "model_type": "base"},
    "fit_config": {"maxiter": 128},
    "train_X": [[0.0], [0.5], [1.0]],
    "train_Y": [[0.0], [0.25], [1.0]],
    "bounds": [[0.0], [1.0]]
  }'
```

Generate candidates from a fitted model:

```bash
curl -X POST http://127.0.0.1:8000/models/<model_id>/candidates \
  -H "Content-Type: application/json" \
  -d '{
    "acquisition_config": {"name": "EI", "acqf_kwargs": {"best_f": 1.0}},
    "optimize_config": {"q": 1, "num_restarts": 10, "raw_samples": 256}
  }'
```

Detailed HTTP payloads, tensor conversion options, optimizer settings, and constraint examples are documented in [`src/bochan/serving/fastapi/README.md`](src/bochan/serving/fastapi/README.md).

---

## Package layout

```text
src/bochan/
├── api/
├── acquisition/
│   ├── objective/
│   ├── feasible/
│   ├── regression/
│   ├── binary/
│   ├── multiclass/
│   ├── ordinal/
│   └── non_gaussian/
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

### Model layout

Model families use the following broad structure:

```text
models/
├── regression/
│   ├── gaussian/
│   │   ├── base/
│   │   ├── deep/
│   │   ├── high_dim/
│   │   └── robust/
│   └── non_gaussian/
│       ├── poisson/
│       ├── beta/
│       ├── gamma/
│       └── negative_binomial/
├── classification/
│   ├── binary/
│   └── multiclass/
├── ordinal/
└── hybrid/
```

Binary and multiclass classification use the same folder pattern:

```text
classification/binary/
├── base/
├── deep/
├── high_dim/
└── robust/

classification/multiclass/
├── base/
├── deep/
├── high_dim/
└── robust/
```

Major model families:

| Family | Purpose |
|---|---|
| `regression/gaussian` | Standard continuous-output Gaussian regression models. |
| `regression/non_gaussian` | Poisson, Beta, Gamma, and Negative Binomial response models. |
| `classification/binary` | Binary GP classification and related wrappers. |
| `classification/multiclass` | Multiclass GP classification and related wrappers. |
| `ordinal` | Ordered-label / ordinal-regression GP wrappers. |
| `hybrid` | Multi-output wrapper for heterogeneous task families. |
| `components` | Shared likelihoods, posterior wrappers, transforms, decomposition utilities, and helper functions. |
| `transforms` | Input transform builders for Normalize and input perturbation. |

### High-level model registry

The default API registry exposes these `task_type` values:

```python
"regression"
"multi_objective"
"binary"
"multiclass"
"ordinal"
"hybrid"
```

The registered `model_type` values are:

```python
"base"
"deepgp"
"deepkernel"
"deepgpdeepkernel"
"saas"
"pca"
"rembo"
"rrp"
"hetero"
```

For `multiclass`, `deepgpdeepkernel` is not currently registered as a separate
model type. Non-Gaussian models exist under `models/regression/non_gaussian/`,
but they are not part of the default `ModelConfig` registry yet; use `model_cls`,
`model_factory`, or a custom registry to connect them to the high-level API.

If `cat_dims` is provided and `input_type` is omitted, the API infers
`input_type="mixed"`; otherwise it uses `input_type="normal"`.

### Acquisition layout

Acquisition functions are divided by model family and task:

```text
acquisition/<family>/
├── active_learning/
├── levelset_estimation/
└── bayesian_optimization/
```

Main acquisition families:

| Family | Scope |
|---|---|
| `regression` | Standard Gaussian / continuous-output acquisitions. |
| `binary` | Binary probability, feasibility, boundary, and classification BO acquisitions. |
| `multiclass` | Target-class probability, class entropy, multiclass level-set, and multiclass BO acquisitions. |
| `ordinal` | Expected utility, ordered-boundary, ordinal feasibility, and ordinal BO acquisitions. |
| `non_gaussian` | Active-learning and level-set acquisitions for response-scale non-Gaussian models. |
| `objective` | Scalarization, probability / utility conversion, input perturbation, and risk aggregation. |
| `feasible` | Shared feasibility constraints and constrained acquisition wrappers. |

Multiclass acquisitions follow the same single / multi / hetero split as binary
and ordinal:

```text
acquisition/multiclass/
├── active_learning/
│   ├── single_output.py
│   ├── multi_output.py
│   ├── hetero_single_output.py
│   └── hetero_multi_output.py
├── levelset_estimation/
│   ├── single_output.py
│   ├── multi_output.py
│   ├── hetero_single_output.py
│   └── hetero_multi_output.py
└── bayesian_optimization/
    ├── single_output.py
    ├── multi_output.py
    ├── hetero_single_output.py
    └── hetero_multi_output.py
```

Non-Gaussian Bayesian-optimization modules are currently placeholders. Standard
BoTorch qEI / qNEI / qUCB / qPI / qEHVI / qNEHVI / qNParEGO should be reused
when a response-scale posterior and objective are sufficient.
