# bochan

`bochan` is a BoTorch-oriented experimental library for Bayesian optimization,
active learning, and level-set estimation across multiple surrogate-model
families.

The project focuses on a consistent interface around Gaussian and non-Gaussian
regression, binary / multiclass classification, ordinal regression, multi-task
and multi-output models, tabular workflows, and ask/tell optimization loops.
Acquisition functions, objectives, candidate repair, and optimization backends
are designed to be reused across those model families.

The codebase is still under active development. Backward support is not the main
priority yet; consistency of API design, tensor shapes, and BoTorch-like behavior
is prioritized.

---

## What this library is for

`bochan` is intended for workflows such as:

- Bayesian optimization with continuous, categorical, mixed, and composition variables
- active learning for regression, binary classification, multiclass classification, ordinal regression, and non-Gaussian response models
- level-set estimation and boundary exploration
- constrained and multi-objective optimization
- robust optimization with input perturbation and risk aggregation
- high-dimensional optimization using PCA, REMBO, SAAS, VAE, or related wrappers
- multi-task, multi-fidelity, and independent multi-output modeling
- external / foundation estimators such as LightGBM, NGBoost, Random Forest, PFN, and TabPFN where supported
- Optuna / Ax-style optimization loops through `BochanStudy`
- human-in-the-loop experiments using `ask()` / `tell()`
- simulation or Python-function optimization with automatic evaluation
- pandas / numpy / CSV based optimization through `bochan.tabular`
- HTTP / JSON model serving through FastAPI
- the React-based Web workbench through `bochan.serving.webapp`

The implementation is designed to stay close to BoTorch concepts where possible:

- model wrappers expose `posterior(X)`;
- latent-response models expose `latent_posterior(X)` when needed;
- acquisition functions operate on q-batch tensors;
- objective classes handle scalarization, probability / utility conversion,
  input-perturbation aggregation, and risk aggregation;
- BoTorch standard acquisition functions are reused when they already cover the
  required behavior.

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

Web workbench dependencies:

```bash
pip install -e ".[web]"
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

## Which API should I use?

| API | Use when |
|---|---|
| `bochan.api.BayesianOptimizer` | You want direct tensor-based model fitting, prediction, and candidate generation. |
| `bochan.api.BochanStudy` | You want an Optuna / Ax-like optimization loop, `ask()` / `tell()`, save / load, early stopping, or generation scheduling. |
| `bochan.tabular.TabularBayesianOptimizer` | You want to work from pandas DataFrames, numpy arrays, or CSV files with column names. |
| `bochan.serving.fastapi` | You want the core Python API through HTTP / JSON. |
| `bochan.serving.webapp` | You want the backend used by the interactive Web workbench. |

The lower-level four-step API is still the internal design unit:

```python
bundle = build_model(train_X, train_Y, model_config)
bundle = fit_model(bundle, fit_config)
acqf = build_acquisition(bundle, acq_config, data_context)
candidates, acq_value = optimize_candidates(acqf, bounds, opt_config)
```

`BayesianOptimizer`, `BochanStudy`, `TabularBayesianOptimizer`, and the serving
adapters are higher-level wrappers around that design.

---

## Quick start: tensor-based optimizer

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

candidates, acq_value = bo.candidate(
    acq_config=AcquisitionConfig(
        name="EI",
        acqf_kwargs={"best_f": train_Y.max()},
    ),
    opt_config=OptimizeConfig(
        q=3,
        num_restarts=10,
        raw_samples=128,
        sequential=True,
    ),
)
```

### `FitConfig.beta`

`FitConfig.beta` is a convenience alias for `mll_kwargs["beta"]`. It is useful
for variational models such as DeepGP and DeepKernel classifiers.

```python
fit_config = FitConfig(
    num_epochs=300,
    lr=0.03,
    beta=0.5,
)
```

When both `beta` and `mll_kwargs["beta"]` are provided, the explicit value in
`mll_kwargs` takes precedence.

### Optimizer backend and `evo_method`

```python
OptimizeConfig(optimizer="optimize_acqf")
OptimizeConfig(optimizer="evo", evo_method="ga")
OptimizeConfig(optimizer="torch")
OptimizeConfig(optimizer="nsgaii")
OptimizeConfig(optimizer="thompson_sampling")
```

Evolutionary methods can also be selected directly:

```python
OptimizeConfig(optimizer="ga")
OptimizeConfig(optimizer="pso")
OptimizeConfig(optimizer="sa")
OptimizeConfig(optimizer="cmaes")
```

Those direct names are normalized to `optimizer="evo"` and stored in
`evo_method`.

---

## Classification, active learning, and level-set estimation

### Multiclass active learning

```python
train_X = torch.rand(40, 2, dtype=torch.double)
train_Y = torch.randint(0, 3, (40,), dtype=torch.long)

bo = BayesianOptimizer(
    model_config=ModelConfig(
        task_type="multiclass",
        model_type="base",
        model_kwargs={"num_classes": 3, "num_inducing_points": 32},
    ),
    fit_config=FitConfig(num_epochs=250, lr=0.03),
    bounds=bounds,
)
bo.fit(train_X, train_Y)

candidates, acq_value = bo.candidate(
    acq_config=AcquisitionConfig(name="entropy"),
    opt_config=OptimizeConfig(q=3, num_restarts=10, raw_samples=128),
)
```

Contextual active-learning aliases include `entropy`, `BALD`, `JointBALD`,
`GreedyJointBALD`, `variance`, `margin`, and `NIPV` where supported by the fitted
model family.

### Multiclass level-set estimation

Multiclass level-set estimation is based on target-class probability. For
example, to explore a boundary where class 2 probability is around 0.5:

```python
candidates, acq_value = bo.candidate(
    acq_config=AcquisitionConfig(
        name="straddle",
        acqf_kwargs={"target_class": 2, "threshold": 0.5},
    ),
    opt_config=OptimizeConfig(q=3, num_restarts=10, raw_samples=128),
)
```

### Multiclass Bayesian optimization

For multiclass Bayesian optimization, `target_class` is required when the
acquisition operates on a selected class probability.

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

`ObjectiveConfig` does not currently auto-build multiclass objectives. Use the
multiclass acquisition's own `target_class`, `threshold`, `best_f`, and related
keyword arguments, or pass `objective` / `objective_factory` explicitly.

---

## Input transforms, risk, constraints, and repair

`InputTransformConfig` can build normalization and input perturbation transforms.
When an acquisition aggregates expanded `q * n_w` samples back to `q`, the
`n_w` used by the input transform and the objective must agree.

```python
from bochan.api import InputTransformConfig, ObjectiveConfig

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

Use `OutcomeConstraintConfig` for user-facing feasibility constraints. Candidate
rounding, k-sparse support, and final constraint repair are configured through
`CandidateRepairConfig` / `OptimizeConfig`.

See `src/bochan/api/README.md` for the full configuration contracts and examples.

---

## Optimization loop API: `BochanStudy`

`BochanStudy` is an Optuna / Ax-style loop wrapper around `BayesianOptimizer`.
It supports both automatic Python-function evaluation and human-in-the-loop /
external evaluation through `ask()` / `tell()`.

```python
from bochan.api import BochanStudy

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
```

For external experiments:

```python
batch = study.ask(q=3, mark_running=True, return_batch=True)
# Run the experiment with batch.candidates.
study.tell(batch, measured_values)
study.save("study.json")
```

See `src/bochan/api/STUDY_README.md` for save / load, early stopping, generation
schedules, failures, and trial-history examples.

---

## Tabular API: pandas / numpy / CSV

`bochan.tabular` is the DataFrame / numpy boundary around the tensor-oriented
`bochan.api` package.

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
```

The tabular API supports:

- direct keyword arguments or canonical config objects;
- direct `beta` for `FitConfig.beta` (`fit_beta` is not a compatibility alias);
- `evo_method` for evolutionary backend selection;
- user-facing outcome constraints;
- string categorical input and target columns;
- missing-value deletion or imputation;
- column-name based bounds, steps, composition indices, and fixed features;
- composition-domain integration through `bochan.composition` and
  `bochan.tabular.composition`.

See `src/bochan/tabular/README.md` and `src/bochan/tabular/ARCHITECTURE.md` for the
canonical tabular contracts.

---

## FastAPI serving

FastAPI serving lives under `bochan.serving.fastapi`.

```bash
pip install -e ".[api]"
uvicorn bochan.serving.fastapi.app:app --reload
```

The default API prefix is `/api/v1`.

| method | path | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Health check |
| `POST` | `/api/v1/models` | Fit a model and store it in memory |
| `GET` | `/api/v1/models` | List stored model ids |
| `POST` | `/api/v1/models/{model_id}/predict` | Predict |
| `POST` | `/api/v1/models/{model_id}/candidates` | Generate candidates |
| `POST` | `/api/v1/models/{model_id}/ask` | Ask for candidates |
| `POST` | `/api/v1/models/{model_id}/tell` | Add observations and optionally refit |
| `POST` | `/api/v1/models/{model_id}/refit` | Refit an existing optimizer |
| `POST` | `/api/v1/models/{model_id}/candidates/compare` | Compare acquisitions |
| `GET` | `/api/v1/acquisitions/names` | List acquisition aliases |

See `src/bochan/serving/fastapi/README.md` for the complete endpoint and payload
reference.

The Web backend is composed separately under `bochan.serving.webapp`. Its app can
include the core FastAPI routes and adds Web-workbench-specific routes without
making `bochan.serving.fastapi` depend on `bochan.serving.webapp`.

---

## Package layout

The current top-level ownership is:

```text
src/bochan/
├── acquisition/       # acquisition functions and objectives
├── api/               # high-level tensor API
├── composition/       # pandas-independent composition domain logic
├── constraints/       # reusable constraint utilities
├── fit/               # fitting helpers
├── inspection/        # feature importance and fitted-model diagnostics
├── llm/               # LLM planning / explanation support
├── models/            # surrogate models
├── optim/             # acquisition optimization backends
├── serving/
│   ├── fastapi/       # core HTTP / JSON adapter
│   ├── webapp/        # Web-workbench backend adapter
│   └── workbench/     # shared workbench application state/services
├── tabular/           # DataFrame / numpy adapter
└── visualization/     # visualization utilities
```

Cross-cutting root modules such as `model_artifact.py`, `tabpfn_assets.py`, and
`tabpfn_preload.py` intentionally live outside the Web adapter because they are
shared by multiple surfaces or deployment-time tooling.

### Model layout

Model code is organized by model family plus cross-cutting strategy packages:

```text
models/
├── regression/
│   ├── gaussian/
│   ├── beta/
│   ├── gamma/
│   ├── count/
│   ├── external/
│   ├── foundation/
│   └── neural/
├── classification/
│   ├── binary/
│   ├── multiclass/
│   └── common/
├── ordinal/
├── hybrid/
├── multitask/
├── multioutput/
├── multifidelity/
├── external/
├── components/
└── transforms/
```

`multitask` owns correlated task/output mechanics, `multioutput` owns wrappers
that aggregate independently fitted outputs, and `multifidelity` owns shared
fidelity-axis abstractions. Concrete likelihood-specific models remain with
their owning task/model family.

See `src/bochan/models/ARCHITECTURE.md` for the ownership rules and
`src/bochan/models/README.md` for model conventions.

### High-level model registry

The exact `model_type` keys are task-dependent and evolve with the registry. The
source of truth is `bochan.api.registry.model`; do not infer availability from a
single flat list in this README.

Representative groups currently include:

- Gaussian GP strategies such as `base`, `kronecker`, `multitask`,
  `multifidelity`, `deepgp`, `deepkernel`, `deepgpdeepkernel`, `saas`, `pca`,
  `rembo`, `vae`, `rrp`, and `hetero` where supported;
- external / neural / foundation estimators such as `lightgbm`, `ngboost`,
  `random_forest`, `deep_ensemble`, `pfn`, and `tabpfn` where supported;
- distribution-specific regression keys prefixed by `beta_`, `gamma_`,
  `poisson_`, and `negative_binomial_`.

If `cat_dims` is provided and `input_type` is omitted, the API can infer mixed
input handling where the requested model family supports it.

---

## Core wrapper conventions

Model families do not all expose uncertainty in the same space. In particular:

- Gaussian regression uses a continuous-response posterior;
- binary classification exposes probability-scale prediction and a latent GP;
- multiclass classification exposes class probabilities and latent class GPs;
- ordinal models keep latent score / cutpoint semantics distinct from class
  probabilities;
- non-Gaussian regression exposes response-scale quantities such as rate or mean;
- hybrid outputs preserve task-specific semantics.

`train_inputs` denotes inputs used by the internal model, while
`train_inputs_raw` denotes the original search-space inputs. See
`src/bochan/models/README.md` and the theory reference for the full posterior and
shape contracts.

---

## Documentation map

| File | Contents |
|---|---|
| `docs/theory/README.md` | Entry point for the mirrored English / Japanese theory reference. |
| `docs/theory/ja/README.md` | Japanese theory chapters and recommended reading paths. |
| `src/bochan/models/README.md` | Model families, registry guidance, and wrapper conventions. |
| `src/bochan/models/ARCHITECTURE.md` | Canonical model-family / cross-cutting ownership rules. |
| `src/bochan/acquisition/README.md` | Acquisition families, objectives, active learning, LSE, and non-Gaussian acquisitions. |
| `src/bochan/api/README.md` | Tensor Python API, configs, objectives, candidate optimization, and repair. |
| `src/bochan/api/STUDY_README.md` | `BochanStudy`, `ask()` / `tell()`, save / load, and scheduling. |
| `src/bochan/tabular/README.md` | DataFrame / numpy / CSV adapter, categorical data, `beta`, repair, and constraints. |
| `src/bochan/tabular/ARCHITECTURE.md` | Canonical tabular package ownership and dependency direction. |
| `src/bochan/serving/fastapi/README.md` | HTTP / JSON endpoints, conversion, payloads, and serving examples. |
| `src/bochan/serving/fastapi/ARCHITECTURE.md` | Transport-layer ownership rules. |

---

## Development status

This repository is under active development. Current priorities include keeping
model wrappers BoTorch-compatible, keeping tensor and response-space contracts
explicit, aligning tabular / serving adapters with canonical APIs, and preferring
shared implementation over compatibility shims or duplicate domain logic.

# Feature importance and fitted-model diagnostics

Use validation data to calculate prediction-performance degradation after a raw
input column is permuted. Here, `permutation` means permutation importance, not
Probability of Improvement.

```python
from bochan.inspection import FeatureImportanceConfig

importance = optimizer.feature_importance(
    X=X_validation,
    y=y_validation,
    config=FeatureImportanceConfig(
        predictive_methods=["permutation"],
        diagnostic_methods=["auto"],
        n_repeats=20,
        random_state=0,
    ),
)
```

`diagnostic_methods=["auto"]` only reads lightweight fitted parameters and module
structure. It does not retrain or run SHAP / Sobol / Integrated Gradients.
Permutation importance is not a causal effect; prefer held-out validation or
cross-validation when interpreting it.
