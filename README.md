# bochan

`bochan` is a BoTorch-oriented experimental library for Bayesian optimization,
active learning, and level-set estimation across multiple surrogate-model
families.

The project focuses on a consistent interface around Gaussian regression,
binary / multiclass classification, ordinal regression, hybrid multi-output
models, tabular data workflows, and ask/tell optimization loops. Acquisition
functions, objectives, candidate repair, and optimization backends are designed
to be reused across those model families.

The codebase is still under active development. Backward support is not
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

## Quick start: tensor-based optimizer

```python
import torch

from bochan.api import AcquisitionConfig, BayesianOptimizer, FitConfig, ModelConfig, OptimizeConfig

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
    acq_config=AcquisitionConfig(name="EI", acqf_kwargs={"best_f": train_Y.max()}),
    opt_config=OptimizeConfig(q=3, num_restarts=10, raw_samples=128, sequential=True),
)
```

### FitConfig.beta

`FitConfig.beta` is a convenience alias for `mll_kwargs["beta"]`. It is useful for
variational models such as DeepGP and DeepKernel classifiers.

```python
fit_config = FitConfig(
    num_epochs=300,
    lr=0.03,
    beta=0.5,
)
```

When both `beta` and `mll_kwargs["beta"]` are provided, the explicit value in
`mll_kwargs` takes precedence.

### Optimizer backend and evo_method

`OptimizeConfig` uses backend-family names. Mixed / non-mixed dispatch is resolved
from categorical settings where possible.

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

## Python API: classification and level-set examples

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
AcquisitionConfig(name="JointBALD")
AcquisitionConfig(name="GreedyJointBALD")
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

`ObjectiveConfig` currently does not auto-build multiclass objectives. Use the
multiclass acquisition's own `target_class`, `threshold`, `best_f`, and related
keyword arguments, or pass `objective` / `objective_factory` explicitly.

---

## Multi-output and heteroscedastic examples

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

For output aggregation, multi-output multiclass acquisitions accept options such
as:

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

Noise-aware acquisitions may accept options such as:

```python
noise_mode="inverse_linear"
noise_mode="exp"
noise_mode="custom"
noise_mode="none"

noise_combine="multiply"
noise_combine="subtract"
```

---

## Input transform and robust objective examples

`InputTransformConfig` can build Normalize and input perturbation transforms from
API settings.

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

`InputTransformConfig(n_w=...)` and `ObjectiveConfig(n_w=...)` should match when
an acquisition needs to aggregate expanded `q * n_w` samples back to `q`.

---

## Outcome constraints

Use `OutcomeConstraintConfig` for user-facing feasibility constraints.

```python
from bochan.api import OutcomeConstraintConfig

acq_config = AcquisitionConfig(
    name="NEHVI",
    outcome_constraint_config=OutcomeConstraintConfig(
        output_indices=[0, 1],
        operators=["ge", "le"],
        thresholds=[0.5, 1.2],
    ),
)
```

For model-dependent feasibility constraints, pass constraint specs through
`outcome_constraint_config.constraints`.

```python
acq_config = AcquisitionConfig(
    name="EI",
    outcome_constraint_config=OutcomeConstraintConfig(
        constraints=[
            {
                "kind": "feasibility",
                "output": "defect",
                "operator": "le",
                "threshold": 0.2,
            }
        ],
        eta=1e-3,
        reduce_constraints="prod",
        reduce_q="mean",
    ),
    acqf_kwargs={"best_f": train_Y.max()},
)
```

Do not specify low-level `constraints` and `outcome_constraint_config` at the same
time.

---

## Candidate optimization and repair

`OptimizeConfig` controls the candidate optimization backend:

```python
OptimizeConfig(optimizer="optimize_acqf")
OptimizeConfig(optimizer="evo", evo_method="ga")
OptimizeConfig(optimizer="torch")
OptimizeConfig(optimizer="nsgaii")
OptimizeConfig(optimizer="thompson_sampling")
```

Mixed optimization can use `fixed_features_list`:

```python
opt_config = OptimizeConfig(
    optimizer="optimize_acqf_mixed",
    q=1,
    fixed_features_list=[
        {2: 0.0},
        {2: 1.0},
    ],
)
```

Candidate repair is configured through `CandidateRepairConfig`:

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
        final_priority="constraints",
    ),
)
```

Notes:

- `steps=None` disables grid rounding.
- `comp_idx=None` or `comp_idx=[]` makes the repair function perform rounding / constraints without k-sparse support selection.
- `support_selection="topk"` selects the largest score entries.
- `support_selection="sample"` samples support entries using `sample_tau` and `sample_eps`.
- `final_priority="grid"` prioritizes final grid alignment.
- `final_priority="constraints"` prioritizes final constraint satisfaction.

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
    optimizer="evo",
    evo_method="ga",
    numeric_indices=["x1", "x2", "x3"],
    steps={"x1": 0.1, "x2": 0.1, "x3": 0.1},
    comp_idx=["x1", "x2", "x3"],
    k=2,
)
```

The tabular API supports:

- direct keyword arguments instead of explicit config objects;
- existing `ModelConfig`, `FitConfig`, `AcquisitionConfig`, `OptimizeConfig`, and `CandidateRepairConfig` objects;
- `fit_beta` / `beta` for `FitConfig.beta`;
- `evo_method` for evolutionary backend selection;
- `outcome_constraint_config` for user-facing outcome constraints;
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
`ModelConfig`, `FitConfig`, `AcquisitionConfig`, `OutcomeConstraintConfig`,
`OptimizeConfig`, and `DataContext`.

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

Example candidate payload:

```json
{
  "acquisition_config": {"name": "EI", "acqf_kwargs": {"best_f": 1.0}},
  "optimize_config": {"q": 1, "num_restarts": 10, "raw_samples": 256}
}
```

The serving layer also accepts `tensor_options` for JSON-to-tensor conversion.

```json
{
  "tensor_options": {"dtype": "float64", "device": "cpu"}
}
```

See `src/bochan/serving/fastapi/README.md` for HTTP examples, including
tensor options, candidate repair, `evo_method`, outcome constraints, multiclass
model fitting, and target-class BO.

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

---

## Core wrapper conventions

### `posterior(X)`

Public prediction API. This should return the prediction object expected by
acquisition functions.

Examples:

- Gaussian regression: continuous response posterior
- Binary classification: probability-scale posterior
- Multiclass classification: class-probability posterior
- Ordinal regression: ordinal class-probability / utility-supported posterior
- Non-Gaussian regression: response-scale posterior such as rate or mean
- Hybrid multi-output: task-aware output collection or objective-space posterior

### `latent_posterior(X)`

Use this when the model has a latent GP but the public posterior is transformed
through a likelihood or link function.

Typical examples:

- binary classification: latent `f` -> sigmoid probability
- multiclass classification: class-wise latent GP -> class probabilities
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
| `src/bochan/models/README.md` | Model family overview, default model registry, wrapper API conventions, and model implementation checklist. |
| `src/bochan/acquisition/README.md` | Acquisition family overview, objectives, feasibility, active learning, level-set estimation, multiclass acquisitions, and non-Gaussian acquisitions. |
| `src/bochan/acquisition/feasible/README.md` | Feasibility constraints and feasibility wrapper usage. |
| `src/bochan/api/README.md` | Tensor-based Python API usage, config objects, registries, objectives, candidate optimization, and repair. |
| `src/bochan/api/STUDY_README.md` | `BochanStudy` optimization loop, `ask()` / `tell()`, `optimize()`, save / load, early stopping, and generation schedules. |
| `src/bochan/tabular/README.md` | pandas / numpy / CSV wrapper, column-name based settings, categorical encoding, imputation, candidate repair, `fit_beta`, `evo_method`, and constraints. |
| `src/bochan/serving/fastapi/README.md` | HTTP / JSON serving examples, tensor conversion, optimizer settings, candidate repair, constraints, and multiclass workflows. |

---

## Development status

This repository is under active development.

Current priorities:

- keep model wrappers BoTorch-supported;
- align naming and arguments across regression / binary / multiclass / ordinal / non-Gaussian families;
- keep tensor shapes q-batch safe;
- make optimization-loop APIs usable from both Python functions and human-in-the-loop experiments;
- keep DataFrame / CSV wrappers thin and consistent with the tensor API;
- keep HTTP / JSON payloads aligned with public `bochan.api` config objects;
- prefer shared implementation over distribution-specific duplication;
- reuse BoTorch standard functionality whenever possible.
# Feature importance and fitted-model diagnostics

Use validation data to calculate prediction-performance degradation after a
raw input column is permuted. Here, `permutation` means permutation importance,
not Probability of Improvement.

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
method = importance.outputs["output_0"].predictive_methods["permutation"]
for name, entry in method.entries.items():
    print(name, entry.importance.mean, entry.importance.std)
diagnostics = importance.outputs["output_0"].model_diagnostics
```

`diagnostic_methods=["auto"]` only reads lightweight fitted parameters and
module structure. It never retrains, optimizes, computes input gradients, or
runs SHAP/Sobol/Integrated Gradients. ARD, projected-model structure, latent
lengthscales, and observation relevance remain diagnostics rather than
predictive importance. Results can be converted with `result.to_dict()`.

For cross-validation, set
`CrossValidationConfig(feature_importance_config=FeatureImportanceConfig(...))`.
Importance is evaluated on each validation fold using its already-fitted fold
model; fold means, between-fold dispersion, ranks, and within-fold repeat
dispersion are retained separately.

Permutation importance is not a causal effect. Correlated features can share or
hide importance, categorical permutation can create unrealistic combinations,
and training-data evaluation is optimistic. Prefer held-out validation or CV.
Joint `FeatureGroup`s preserve within-group row relationships. Runtime scales
approximately with features/groups × repeats × folds. PCA loadings, REMBO
projections, RRP observation relevance, and DeepKernel latent lengthscales are
not raw-feature rankings. Future predictive methods can be added alongside
`permutation`; gradient, SHAP, and Sobol methods are intentionally unsupported.
