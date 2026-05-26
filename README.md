# bochan

`bochan` is a BoTorch-oriented experimental library for Bayesian optimization,
active learning, and level-set estimation across multiple surrogate-model
families.

The project focuses on building a consistent interface around Gaussian
regression, non-Gaussian regression, binary / multiclass classification, and
ordinal regression models, together with acquisition functions and objectives
that can be reused across those model families.

The codebase is still under active development. Backward compatibility is not
the main priority yet; consistency of API design, tensor shapes, and BoTorch-like
behavior is prioritized.

---

## What this library is for

`bochan` is intended for workflows such as:

- Bayesian optimization with continuous, categorical, and mixed variables
- active learning for regression, classification, ordinal, and non-Gaussian
  response models
- level-set estimation and boundary exploration
- constrained and multi-objective optimization
- robust optimization with input perturbation and risk aggregation
- high-dimensional optimization using PCA, REMBO, SAAS, or related wrappers
- model experimentation around GP, DeepGP, Deep Kernel GP, heteroscedastic GP,
  and robust relevance pursuit variants

The implementation is designed to stay close to BoTorch concepts where possible:

- model wrappers expose `posterior(X)`;
- latent-response models expose `latent_posterior(X)` when needed;
- acquisition functions operate on q-batch tensors;
- objective classes handle scalarization, probability / utility conversion,
  input-perturbation aggregation, and risk aggregation;
- BoTorch standard acquisition functions are reused when they already cover the
  required behavior.

---

## Package layout

```text
bochan/
└── src/bochan/
    ├── models/
    │   ├── components/
    │   ├── regression/
    │   │   ├── gaussian/
    │   │   └── non_gaussian/
    │   ├── classification/
    │   └── ordinal/
    │
    └── acquisition/
        ├── objective/
        ├── regression/
        ├── binary/
        ├── ordinal/
        └── non_gaussian/
```

### `models/`

Model wrappers and reusable model components live under `src/bochan/models/`.

Major families:

| Family | Purpose |
|---|---|
| `regression/gaussian` | Standard continuous-output Gaussian regression models. |
| `regression/non_gaussian` | Poisson, Beta, Gamma, and Negative Binomial response models. |
| `classification/binary` | Binary GP classification and related wrappers. |
| `classification/multiclass` | Multiclass classification wrappers. |
| `ordinal` | Ordered-label / ordinal-regression GP wrappers. |
| `components` | Shared likelihoods, posterior wrappers, transforms, decomposition utilities, and helper functions. |

Each model family is organized as much as possible around the same subfolders:

```text
base/
deep/
high_dim/
robust/
```

See `src/bochan/models/README.md` for the detailed model-package design.

### `acquisition/`

Acquisition functions and objective classes live under `src/bochan/acquisition/`.

Major families:

| Family | Purpose |
|---|---|
| `objective` | Scalarization, utility conversion, probability conversion, and risk aggregation. |
| `regression` | Gaussian / standard regression acquisitions. |
| `binary` | Binary classification acquisitions. |
| `ordinal` | Ordinal regression acquisitions. |
| `non_gaussian` | Shared acquisitions for Poisson / Beta / Gamma / Negative Binomial models. |

Acquisition families are further divided by task:

```text
bayesian_optimization/
active_learning/
levelset_estimation/
```

The `non_gaussian` acquisition package is top-level under
`bochan.acquisition.non_gaussian`, not under `bochan.acquisition.regression`.

See `src/bochan/acquisition/README.md` for the detailed acquisition-package
design.

---

## Model families

## 1. Gaussian regression

Gaussian regression models are used for continuous targets with approximately
Gaussian observation noise.

Typical use cases:

- continuous response Bayesian optimization
- regression active learning
- scalar level-set estimation
- multi-output regression through independent or compatible multi-output models

Whenever possible, standard BoTorch acquisition functions such as qEI, qNEI,
qUCB, qPI, qEHVI, qNEHVI, and qNParEGO should be used directly.

## 2. Non-Gaussian regression

Non-Gaussian regression models are used when the observation distribution is not
well represented by a Gaussian likelihood.

Current families:

| Family | Target type |
|---|---|
| Poisson | non-negative count response |
| Beta | bounded continuous response in `(0, 1)` |
| Gamma | positive continuous response |
| Negative Binomial | over-dispersed count response |

These wrappers generally have both a latent GP scale and a response scale:

```text
latent GP f -> likelihood / link -> response distribution
```

Therefore, the public wrapper should expose:

- `latent_posterior(X)` for latent GP uncertainty;
- `posterior(X)` for response-scale prediction;
- response-scale posterior wrappers for acquisition functions.

Custom non-Gaussian active-learning and level-set acquisitions are implemented
under `bochan.acquisition.non_gaussian`. Standard BoTorch BO acquisitions are not
reimplemented when the response-scale posterior is sufficient.

## 3. Binary classification

Binary classification models are used for two-class labels, feasibility labels,
and binary constraints.

Typical use cases:

- feasible / infeasible modeling
- binary constraints in Bayesian optimization
- boundary exploration
- BALD / entropy / margin-based active learning

Public prediction should usually operate on probability scale, while training
uses latent GP quantities.

## 4. Multiclass classification

Multiclass classification models are used for unordered class labels with more
than two classes.

They are separated from ordinal models because class labels have no intrinsic
ordering.

## 5. Ordinal regression

Ordinal regression models are used for ordered labels such as quality levels,
risk levels, or thresholded process outcomes.

Typical use cases:

- ordered categorical response modeling
- ordinal constraints
- utility-based Bayesian optimization
- ordinal boundary exploration
- active learning near ordered class thresholds

Ordinal models generally require cutpoint / threshold handling in addition to
latent GP prediction.

---

## Acquisition and objective design

Acquisition functions are designed to align with model-family semantics.

| Model family | Acquisition naming pattern |
|---|---|
| Gaussian regression | `qRegression...` |
| Binary classification | `qBinary...` |
| Ordinal regression | `qOrdinal...` |
| Non-Gaussian regression | `qNonGaussian...` |
| Multi-output regression | `qMultiOutputRegression...` |
| Multi-output binary classification | `qMultiOutputBinary...` |
| Multi-output ordinal regression | `qMultiOutputOrdinal...` |
| Heteroscedastic variants | `qHetero...` |

Objective classes are used to convert posterior samples or pointwise acquisition
scores into the form required by BoTorch acquisition functions.

Common objective responsibilities:

- scalarize regression outputs;
- convert binary probabilities into scores;
- convert ordinal class probabilities into expected utility;
- aggregate input-perturbation expanded scores from `q * n_w` back to `q`;
- apply mean / VaR / CVaR style risk aggregation;
- provide qEHVI / qNEHVI / qNParEGO compatible multi-output objectives.

---

## Core wrapper conventions

The following conventions are used throughout the library.

### `posterior(X)`

Public prediction API. This should return the prediction object expected by
acquisition functions.

Examples:

- Gaussian regression: continuous response posterior
- Binary classification: probability-scale posterior
- Ordinal regression: ordinal class-probability / utility-compatible posterior
- Non-Gaussian regression: response-scale posterior such as rate or mean

### `latent_posterior(X)`

Use this when the model has a latent GP but the public posterior is transformed
through a likelihood or link function.

Typical examples:

- binary classification: latent `f` -> sigmoid probability
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

## Input transforms, mixed variables, and perturbation

Input transforms should be passed through wrapper constructors where possible.

For mixed continuous / categorical inputs:

- continuous columns may be normalized or projected;
- categorical columns should remain unchanged;
- helpers should check that categorical columns are not modified accidentally;
- `cat_dims` should be stored consistently.

For input perturbation:

- evaluation transforms may expand candidates from `q` to `q * n_w`;
- objective classes should aggregate back to q-batch shape;
- risk modes such as mean / VaR / CVaR should be handled consistently across
  model families.

---

## High-dimensional, deep, and robust variants

High-dimensional wrappers may use PCA, REMBO, SAAS, or related strategies. They
should preserve raw search-space inputs while training internal models on the
appropriate transformed or projected representation.

DeepGP and Deep Kernel GP wrappers should preserve the same public interface:

- `forward(X)` for training-time latent distributions;
- `posterior(X)` for response-scale or probability-scale predictions;
- `latent_posterior(X)` where applicable;
- `input_transform` support before feature extraction where appropriate.

Robust and heteroscedastic variants should expose enough structure for
acquisition functions to distinguish latent uncertainty from observation noise.

---

## Minimal examples

### Standard model training pattern

```python
model = SomeGPModel(
    train_X=train_X,
    train_Y=train_Y,
    input_transform=input_transform,
)

mll = model.make_mll()
fit_func(mll)

posterior = model.posterior(test_X)
```

### Latent and response posterior pattern

```python
latent_post = model.latent_posterior(test_X)
response_post = model.posterior(test_X)
```

Use this pattern for classification, ordinal, and non-Gaussian models.

### Regression acquisition example

```python
from bochan.acquisition.regression.active_learning import qRegressionPosteriorVariance

acqf = qRegressionPosteriorVariance(model=model)
values = acqf(candidates)
```

### Non-Gaussian acquisition example

```python
from bochan.acquisition.non_gaussian.active_learning import qNonGaussianResponseMeanVariance
from bochan.acquisition.non_gaussian.levelset_estimation import qNonGaussianStraddle

active_acqf = qNonGaussianResponseMeanVariance(model=model, num_samples=64)
levelset_acqf = qNonGaussianStraddle(model=model, threshold=1.0, num_samples=64)
```

### Objective example

```python
from bochan.acquisition.objective import RegressionScalarObjective

objective = RegressionScalarObjective(
    n_w=8,
    risk_type=None,
    maximize=True,
)
```

---

## Documentation map

More focused documentation is available in package-level README files:

| File | Contents |
|---|---|
| `docs/theory/README.md` | Theoretical background for GP models, Bayesian optimization, acquisition functions, active learning, level-set estimation, classification / ordinal BO, multi-objective constraints, input perturbation, risk, and tensor shape conventions. |
| `src/bochan/models/README.md` | Model family overview, wrapper API conventions, and model implementation checklist. |
| `src/bochan/acquisition/README.md` | Acquisition family overview, objectives, active learning, level-set estimation, and non-Gaussian acquisitions. |

---

## Development status

This repository is under active development.

Current priorities:

- keep model wrappers BoTorch-compatible;
- align naming and arguments across regression / binary / ordinal /
  non-Gaussian families;
- keep tensor shapes q-batch safe;
- prefer shared implementation over distribution-specific duplication;
- reuse BoTorch standard functionality whenever possible;
- make placeholder modules explicit when a family or variant is not implemented
  yet.
