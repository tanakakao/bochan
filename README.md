# bochan

`bochan` is a BoTorch-oriented experimental library for Bayesian optimization,
active learning, and level-set estimation across multiple surrogate-model
families.

The project focuses on a consistent interface around Gaussian regression,
non-Gaussian regression, binary / multiclass classification, ordinal regression,
tabular data workflows, and ask/tell optimization loops. Acquisition functions,
objectives, candidate repair, and optimization backends are designed to be reused
across those model families.

The codebase is still under active development. Backward compatibility is not
the main priority yet; consistency of API design, tensor shapes, and
BoTorch-like behavior is prioritized.

---

## What this library is for

`bochan` is intended for workflows such as:

- Bayesian optimization with continuous, categorical, and mixed variables
- active learning for regression, classification, ordinal, and non-Gaussian response models
- level-set estimation and boundary exploration
- constrained and multi-objective optimization
- robust optimization with input perturbation and risk aggregation
- high-dimensional optimization using PCA, REMBO, SAAS, or related wrappers
- model experimentation around GP, DeepGP, Deep Kernel GP, heteroscedastic GP, and robust relevance pursuit variants
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

Tabular API with iterative / multiple-imputation style missing-value handling:

```bash
pip install -e ".[impute]"
```

All optional extras:

```bash
pip install -e ".[all]"
```

---

## Package layout

```text
src/bochan/
├── api/
├── acquisition/
│   ├── objective/
│   ├── regression/
│   ├── binary/
│   ├── multiclass/
│   ├── ordinal/
│   └── non_gaussian/
├── models/
│   ├── components/
│   ├── regression/
│   │   ├── gaussian/
│   │   └── non_gaussian/
│   ├── classification/
│   │   ├── binary/
│   │   └── multiclass/
│   ├── hybrid/
│   └── ordinal/
├── tabular/
├── visualization/
└── serving/
    └── fastapi/
```

### Model layout

Binary and multiclass classification use the same folder pattern:

```text
classification/binary/
├── base/
├── deep/
├── high_dim/
└── robust/

classification/multiclass/
├── base/
│   └── models.py
├── deep/
│   ├── deepgp.py
│   └── deepkernel.py
├── high_dim/
│   ├── decomposition.py
│   └── saas.py
└── robust/
    ├── heteroscedastic.py
    └── relevance_pursuit.py
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

### Acquisition layout

Acquisition functions are divided by model family and task:

```text
acquisition/<family>/
├── active_learning/
├── levelset_estimation/
└── bayesian_optimization/
```

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

---

## Which API should I use?

| API | Use when |
|---|---|
| `bochan.api.BayesianOptimizer` | You want direct tensor-based model fitting, prediction, and candidate generation. |
| `bochan.api.BochanStudy` | You want an Optuna / Ax-like optimization loop, `ask()` / `tell()`, save / load, early stopping, or generation scheduling. |
| `bochan.tabular.TabularBayesianOptimizer` | You want to work from pandas DataFrames, numpy arrays, or CSV files with column names. |
| `bochan.serving.fastapi` | You want HTTP / JSON model serving for applications or external systems. |

The lower-level four-step API is still the internal design unit:

```python
bundle = build_model(train_X, train_Y, model_config)
bundle = fit_model(bundle, fit_config)
acqf = build_acquisition(bundle, acq_config, data_context)
candidates, acq_value = optimize_candidates(acqf, bounds, opt_config)
```

`BayesianOptimizer`, `BochanStudy`, `TabularBayesianOptimizer`, and FastAPI are
higher-level wrappers around that design.

---

## Python API: tensor-based optimizer

The high-level tensor API is exposed through `bochan.api.BayesianOptimizer`.

### Single-output multiclass active learning

```python
import torch

from bochan.api import AcquisitionConfig, BayesianOptimizer, FitConfig, ModelConfig, OptimizeConfig

train_X = torch.rand(40, 2, dtype=torch.double)
train_Y = torch.randint(0, 3, (40,), dtype=torch.long)
bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

bo = BayesianOptimizer(
    model_config=ModelConfig(
        task_type="multiclass",
        model_type="base",
        model_kwargs={
            "num_classes": 3,
            "num_inducing_points": 32,
        },
    ),
    fit_config=FitConfig(num_epochs=250, lr=0.03),
    bounds=bounds,
)
bo.fit(train_X, train_Y)

candidates, acq_value = bo.candidate(
    acq_config=AcquisitionConfig(name="entropy"),
    opt_config=OptimizeConfig(q=3, num_restarts=10, raw_samples=128, sequential=True),
)
```

For active learning, these contextual aliases resolve according to `task_type`,
`model_type`, and whether the model is multi-output:

```python
AcquisitionConfig(name="entropy")
AcquisitionConfig(name="BALD")
AcquisitionConfig(name="variance")
AcquisitionConfig(name="margin")
AcquisitionConfig(name="NIPV")
```

### Multiclass level-set estimation

Multiclass level-set estimation is based on target-class probability:

```text
p(target_class | x)
```

Example: explore the boundary where class 2 probability is around 0.5.

```python
candidates, acq_value = bo.candidate(
    acq_config=AcquisitionConfig(
        name="straddle",
        acqf_kwargs={
            "target_class": 2,
            "threshold": 0.5,
        },
    ),
    opt_config=OptimizeConfig(q=3, num_restarts=10, raw_samples=128),
)
```

Useful aliases:

```python
AcquisitionConfig(name="straddle", acqf_kwargs={"target_class": 1, "threshold": 0.5})
AcquisitionConfig(name="ICU", acqf_kwargs={"target_class": 1, "threshold": 0.5})
AcquisitionConfig(name="boundaryvariance", acqf_kwargs={"target_class": 1, "threshold": 0.5})
AcquisitionConfig(name="classentropy")
AcquisitionConfig(name="poe", acqf_kwargs={"target_class": 1, "threshold": 0.5})
AcquisitionConfig(name="levelset", acqf_kwargs={"target_class": 1, "threshold": 0.5})
```

### Multiclass Bayesian optimization

For multiclass Bayesian optimization, `target_class` is required. The objective is:

```text
maximize p(target_class | x)
```

```python
candidates, acq_value = bo.candidate(
    acq_config=AcquisitionConfig(
        name="EI",
        acqf_kwargs={
            "target_class": 2,
            "best_f": 0.70,
            "num_samples": 128,
        },
    ),
    opt_config=OptimizeConfig(q=1, num_restarts=10, raw_samples=128),
)
```

Supported contextual aliases:

```python
AcquisitionConfig(name="EI", acqf_kwargs={"target_class": 2, "best_f": 0.7})
AcquisitionConfig(name="PI", acqf_kwargs={"target_class": 2, "best_f": 0.7})
AcquisitionConfig(name="UCB", acqf_kwargs={"target_class": 2, "beta": 2.0})
AcquisitionConfig(name="PoF", acqf_kwargs={"target_class": 2})
```

### Mixed-input multiclass model

```python
bo = BayesianOptimizer(
    model_config=ModelConfig(
        task_type="multiclass",
        model_type="base",
        cat_dims=[2],
        model_kwargs={"num_classes": 3},
    ),
    fit_config=FitConfig(num_epochs=250, lr=0.03),
    bounds=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]], dtype=torch.double),
)
```

When `cat_dims` is set and `input_type` is omitted, Python API infers
`input_type="mixed"` automatically.

### Multi-output multiclass model

Multi-output multiclass is built from independent submodels and wrapped through
the hybrid multi-output wrapper. Acquisition classes are still resolved to
`qMultiOutputMulticlass...` variants because the fitted bundle records
`multi_output=True`.

```python
from bochan.api import MultiOutputConfig

bo = BayesianOptimizer(
    model_config=ModelConfig(
        task_type="multiclass",
        model_type="base",
        model_kwargs={"num_classes": 3},
        multi_output_config=MultiOutputConfig(
            output_task_types=["multiclass", "multiclass"],
            output_names=["defect_type_a", "defect_type_b"],
            use_hybrid=True,
        ),
    ),
    fit_config=FitConfig(num_epochs=200, lr=0.03),
    bounds=bounds,
)
bo.fit(train_X, train_Y_multi)  # train_Y_multi shape: n x m

candidates, acq_value = bo.candidate(
    acq_config=AcquisitionConfig(name="entropy", acqf_kwargs={"output_reduction": "mean"}),
    opt_config=OptimizeConfig(q=3, num_restarts=10, raw_samples=128),
)
```

For output aggregation, multi-output multiclass acquisitions accept:

```python
output_reduction="mean"
output_reduction="sum"
output_reduction="max"
output_reduction="min"
output_reduction="weighted_mean"
```

### Heteroscedastic multiclass model

```python
bo = BayesianOptimizer(
    model_config=ModelConfig(
        task_type="multiclass",
        model_type="hetero",
        model_kwargs={"num_classes": 3},
    ),
    fit_config=FitConfig(num_epochs=250, lr=0.03),
    bounds=bounds,
)
```

With `model_type="hetero"`, contextual acquisition aliases resolve to
`qHeteroMulticlass...` or `qHeteroMultiOutputMulticlass...` variants.

Noise-aware acquisitions accept:

```python
noise_mode="inverse_linear"
noise_mode="exp"
noise_mode="custom"
noise_mode="none"

noise_combine="multiply"
noise_combine="subtract"
```

---

## Optimization loop API: `BochanStudy`

`BochanStudy` is an Optuna / Ax-style loop wrapper around `BayesianOptimizer`.
It is useful when candidate generation and evaluation are separate operations.

It supports two main patterns:

1. automatic evaluation of a Python objective function;
2. human-in-the-loop or external evaluation through `ask()` / `tell()`.

### Automatic Python-function optimization

```python
import torch

from bochan.api import BochanStudy

bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

study = BochanStudy(
    bounds=bounds,
    n_initial_random=10,
)

study.optimize(
    objective_func=lambda X: X.sum(dim=-1),
    n_trials=20,
    q=2,
    save_path="study.json",
)

train_X, train_Y = study.completed_data()
```

### Human-in-the-loop / simulation workflow

```python
batch = study.ask(q=3, mark_running=True, return_batch=True)

# Send batch.candidates to an experiment, Web UI, or external simulator.
# Register measured values when they become available.

study.tell(batch, measured_values)
study.save("study.json")
```

Resume later by loading the trial history and reinjecting runtime configs:

```python
study = BochanStudy.load(
    "study.json",
    model_config=model_config,
    fit_config=fit_config,
    acq_config=acq_config,
    opt_config=opt_config,
    data_context=data_context,
    bounds=bounds,
)

next_batch = study.ask(q=3, return_batch=True)
```

`BochanStudy` also supports:

- `EarlyStoppingConfig` for target-based or no-improvement stopping;
- `GenerationSchedule` and `GenerationStep` for switching `q`, acquisition, optimization settings, or data context during a run;
- `mark_failed(...)` for failed experiments or simulations;
- `trials_dataframe()` for run history inspection.

See `src/bochan/api/STUDY_README.md` for detailed examples.

---

## Tabular API: pandas / numpy / CSV

`bochan.tabular` is a thin wrapper over the tensor-based API. It converts
DataFrame / numpy / CSV inputs into tensors internally, while allowing users to
work with column names and DataFrame outputs.

### DataFrame example

```python
import pandas as pd

from bochan.tabular import TabularBayesianOptimizer


df = pd.read_csv("data.csv")

bo = TabularBayesianOptimizer(
    task_type="regression",
    model_type="base",
    input_cols=["x1", "x2", "x3", "machine"],
    target_cols="y",
    categorical_cols=["machine"],
    bounds={
        "x1": [0.0, 1.0],
        "x2": [0.0, 1.0],
        "x3": [0.0, 1.0],
        "machine": [0, 2],
    },
)

bo.fit(df)

candidates_df, acq_value = bo.candidate(
    acq_name="NIPV",
    q=10,
    numeric_indices=["x1", "x2", "x3"],
    steps={"x1": 0.1, "x2": 0.1, "x3": 0.1},
    comp_idx=["x1", "x2", "x3"],
    k=2,
)
```

### CSV and numpy examples

```python
bo = TabularBayesianOptimizer.from_csv(
    "data.csv",
    task_type="regression",
    model_type="base",
    input_cols=["x1", "x2", "x3"],
    target_cols="y",
)

bo.fit()
```

```python
bo = TabularBayesianOptimizer(task_type="regression", model_type="base")
bo.fit(
    X_np,
    y_np,
    feature_names=["x1", "x2", "x3"],
    input_cols=["x1", "x2", "x3"],
    target_names=["y"],
)
```

The tabular API supports:

- direct keyword arguments instead of explicit config objects;
- existing `ModelConfig`, `FitConfig`, `AcquisitionConfig`, `OptimizeConfig`, and `CandidateRepairConfig` objects;
- string categorical input columns through label encoding and candidate decoding;
- string categorical target columns for binary / classification workflows;
- missing-value deletion or imputation;
- column-name based `bounds`, `steps`, `comp_idx`, `fixed_features`, and `fixed_features_list`.

See `src/bochan/tabular/README.md` for detailed examples.

---

## FastAPI serving

FastAPI serving lives under:

```text
bochan.serving.fastapi
```

Install and start:

```bash
pip install -e ".[api]"
uvicorn bochan.serving.fastapi.app:app --reload
```

The FastAPI layer mirrors the Python API. It accepts JSON versions of
`ModelConfig`, `FitConfig`, `AcquisitionConfig`, `OptimizeConfig`, and
`DataContext`.

Important endpoints:

| method | path | Purpose |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/models` | Fit a model and store it in memory |
| `GET` | `/models` | List stored model ids |
| `POST` | `/models/{model_id}/predict` | Predict |
| `POST` | `/models/{model_id}/candidates` | Generate candidates |
| `POST` | `/models/{model_id}/ask` | Alias for candidate generation |
| `POST` | `/models/{model_id}/tell` | Add observations and optionally refit |
| `POST` | `/models/{model_id}/refit` | Refit existing optimizer |
| `POST` | `/models/{model_id}/candidates/compare` | Compare multiple acquisitions |
| `GET` | `/acquisitions/names` | List acquisition aliases |

See `src/bochan/serving/fastapi/README.md` for HTTP examples, including
multiclass model fitting and target-class BO.

---

## Core wrapper conventions

### `posterior(X)`

Public prediction API. This should return the prediction object expected by
acquisition functions.

Examples:

- Gaussian regression: continuous response posterior
- Binary classification: probability-scale posterior
- Multiclass classification: class-probability posterior
- Ordinal regression: ordinal class-probability / utility-compatible posterior
- Non-Gaussian regression: response-scale posterior such as rate or mean

### `latent_posterior(X)`

Use this when the model has a latent GP but the public posterior is transformed
through a likelihood or link function.

Typical examples:

- binary classification: latent `f` -> sigmoid probability
- multiclass classification: class-wise latent GP -> softmax probability
- ordinal regression: latent `f` -> cutpoint probabilities
- Poisson regression: latent `f` -> positive rate
- Beta regression: latent `f` -> response mean in `(0, 1)`

### `forward(X)`

For GPyTorch-trained wrappers, `forward(X)` should return the latent GP
distribution used by the likelihood during fitting.

### `make_mll()`

Wrappers should expose `make_mll()` when there is a recommended training
objective such as `ExactMarginalLogLikelihood` or `VariationalELBO`.

### `train_inputs` and `train_inputs_raw`

Use the following distinction:

```text
train_inputs      = inputs actually used by the internal latent / BoTorch model
train_inputs_raw  = original raw search-space inputs
```

This distinction is important for input transforms, high-dimensional wrappers,
mixed variables, and candidate-update logic.

### `condition_on_observations`

When supported, this method should accept raw `X`, prepare `Y` appropriately,
preserve model-family settings, and return a new wrapper instance.

Unsupported options such as Gaussian-style `noise=` for non-Gaussian likelihoods
should raise explicit `NotImplementedError` rather than being ignored.

---

## Documentation map

| File | Contents |
|---|---|
| `docs/theory/README.md` | Theoretical background for GP models, Bayesian optimization, acquisition functions, active learning, level-set estimation, classification / ordinal BO, multi-objective constraints, input perturbation, risk, and tensor shape conventions. |
| `src/bochan/models/README.md` | Model family overview, wrapper API conventions, and model implementation checklist. |
| `src/bochan/acquisition/README.md` | Acquisition family overview, objectives, active learning, level-set estimation, and non-Gaussian acquisitions. |
| `src/bochan/api/README.md` | Tensor-based Python API usage. |
| `src/bochan/api/STUDY_README.md` | `BochanStudy` optimization loop, `ask()` / `tell()`, `optimize()`, save / load, early stopping, and generation schedules. |
| `src/bochan/tabular/README.md` | pandas / numpy / CSV wrapper, column-name based settings, categorical encoding, imputation, and candidate repair. |
| `src/bochan/serving/fastapi/README.md` | HTTP / JSON serving examples. |

---

## Development status

This repository is under active development.

Current priorities:

- keep model wrappers BoTorch-compatible;
- align naming and arguments across regression / binary / multiclass / ordinal / non-Gaussian families;
- keep tensor shapes q-batch safe;
- make optimization-loop APIs usable from both Python functions and human-in-the-loop experiments;
- keep DataFrame / CSV wrappers thin and consistent with the tensor API;
- prefer shared implementation over distribution-specific duplication;
- reuse BoTorch standard functionality whenever possible;
- make placeholder modules explicit when a family or variant is not implemented yet.
