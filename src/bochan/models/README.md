# models package

`bochan.models` contains BoTorch-style surrogate model wrappers used by the
acquisition package and higher-level optimization / active-learning workflows.

The package is organized around model families rather than one-off experiments.
Each family should expose a consistent wrapper API so that acquisition functions,
fit helpers, visualization utilities, tabular wrappers, serving APIs, and
candidate-generation code can treat models uniformly.

---

## Design goals

The main goals of this package are:

1. Keep a BoTorch-like public interface.
2. Separate raw search-space inputs from transformed / latent-model inputs.
3. Use consistent naming across regression, binary classification, multiclass
   classification, ordinal regression, and non-Gaussian regression.
4. Make advanced model variants discoverable through a regular directory layout.
5. Avoid adding custom model-specific acquisition logic when BoTorch-compatible
   posterior / objective behavior is enough.
6. Keep high-level API registry behavior explicit, especially for model families
   that exist in `models/` but are not yet exposed by the default API registry.

A model wrapper should generally provide:

- `posterior(X, ...)`
- `latent_posterior(X, ...)` when the model has a latent GP layer
- `condition_on_observations(X, Y, ...)` when supported
- `make_mll()` for the recommended marginal log-likelihood / ELBO
- `train_inputs`
- `train_inputs_raw`
- `train_targets`
- `input_transform` support where applicable

---

## Current layout

```text
models/
├── components/
│   ├── beta.py
│   ├── decomposition.py
│   ├── gamma.py
│   ├── negative_binomial.py
│   └── poisson.py
│
├── transforms/
│   └── input.py
│
├── regression/
│   ├── gaussian/
│   │   ├── base/
│   │   ├── deep/
│   │   ├── high_dim/
│   │   └── robust/
│   └── non_gaussian/
│       ├── poisson/
│       │   ├── base/
│       │   ├── deep/
│       │   ├── high_dim/
│       │   └── robust/
│       ├── beta/
│       │   ├── base/
│       │   ├── deep/
│       │   ├── high_dim/
│       │   └── robust/
│       ├── gamma/
│       │   ├── base/
│       │   ├── deep/
│       │   ├── high_dim/
│       │   └── robust/
│       └── negative_binomial/
│           ├── base/
│           ├── deep/
│           ├── high_dim/
│           └── robust/
│
├── classification/
│   ├── binary/
│   │   ├── base/
│   │   ├── deep/
│   │   ├── high_dim/
│   │   └── robust/
│   └── multiclass/
│       ├── base/
│       ├── deep/
│       ├── high_dim/
│       └── robust/
│
├── ordinal/
│   ├── base/
│   ├── deep/
│   ├── high_dim/
│   └── robust/
│
└── hybrid/
    ├── multi_output.py
    ├── prediction.py
    └── specs.py
```

Notes:

- `regression/gaussian/` is the standard continuous-output regression family.
- `regression/non_gaussian/` contains response distributions such as Poisson,
  Beta, Gamma, and Negative Binomial.
- `classification/binary/` and `classification/multiclass/` are separated because
  their likelihoods, posterior semantics, target labels, and acquisition
  objectives are different.
- `ordinal/` is treated as its own family rather than a special case of
  classification, because ordinal cutpoints and boundary-aware acquisitions are
  central to its API.
- `hybrid/` contains wrappers and output specifications for heterogeneous
  multi-output models that combine regression, binary, multiclass, and ordinal
  outputs.
- `components/` contains reusable likelihoods, posterior wrappers, transforms,
  kernels, decomposition utilities, and small helper functions shared by model
  wrappers.
- `transforms/` contains input-transform builders used by the API layer, including
  Normalize and input perturbation composition.

---

## Directory convention inside each family

Where possible, each model family uses the same internal subdirectories:

| Directory | Meaning |
|---|---|
| `base/` | Standard model wrappers and core likelihood / posterior integration. |
| `deep/` | DeepGP or Deep Kernel GP variants. |
| `high_dim/` | PCA, REMBO, SAAS, or other high-dimensional wrappers. |
| `robust/` | Heteroscedastic, robust relevance pursuit, or noise-aware variants. |

Not every family is required to have a complete implementation in every
subcategory. If a file exists only as a placeholder, it should say so clearly in
its docstring.

---

## High-level API registry support

The default high-level registry in `bochan.api.model_registry` is lazy and maps
`input_type`, `task_type`, and `model_type` to model classes only when they are
requested.

### Normal-input registry

| `task_type` | Registered `model_type` values |
|---|---|
| `regression` | `base`, `deepgp`, `deepkernel`, `deepgpdeepkernel`, `saas`, `pca`, `rembo`, `rrp`, `hetero` |
| `multi_objective` | `base`, `deepgp`, `deepkernel`, `deepgpdeepkernel`, `saas`, `pca`, `rembo`, `rrp`, `hetero` |
| `binary` | `base`, `deepgp`, `deepkernel`, `deepgpdeepkernel`, `saas`, `pca`, `rembo`, `rrp`, `hetero` |
| `ordinal` | `base`, `deepgp`, `deepkernel`, `deepgpdeepkernel`, `saas`, `pca`, `rembo`, `rrp`, `hetero` |
| `multiclass` | `base`, `deepgp`, `deepkernel`, `saas`, `pca`, `rembo`, `rrp`, `hetero` |

### Mixed-input registry

The same task families are registered under `input_type="mixed"` when `cat_dims`
is supplied. Mixed regression and multi-objective models use BoTorch's
`MixedSingleTaskGP` for `model_type="base"`; mixed classification and ordinal
models use family-specific mixed wrappers.

| `task_type` | Registered `model_type` values |
|---|---|
| `regression` | `base`, `deepgp`, `deepkernel`, `deepgpdeepkernel`, `saas`, `pca`, `rembo`, `rrp`, `hetero` |
| `multi_objective` | `base`, `deepgp`, `deepkernel`, `deepgpdeepkernel`, `saas`, `pca`, `rembo`, `rrp`, `hetero` |
| `binary` | `base`, `deepgp`, `deepkernel`, `deepgpdeepkernel`, `saas`, `pca`, `rembo`, `rrp`, `hetero` |
| `ordinal` | `base`, `deepgp`, `deepkernel`, `deepgpdeepkernel`, `saas`, `pca`, `rembo`, `rrp`, `hetero` |
| `multiclass` | `base`, `deepgp`, `deepkernel`, `saas`, `pca`, `rembo`, `rrp`, `hetero` |

`deepgpdeepkernel` is currently registered for regression, multi-objective,
binary, and ordinal tasks. Multiclass has `deepgp` and `deepkernel` entries, but
not a separate `deepgpdeepkernel` registry key.

### Non-Gaussian registry status

Non-Gaussian models exist under `models/regression/non_gaussian/`, but they are
not currently part of the default high-level `ModelConfig(task_type=...)`
registry. Use them directly, or add a custom registry / `model_cls` /
`model_factory` path when wiring them through the API.

---

## Core API conventions

### `train_inputs` and `train_inputs_raw`

Use the following convention consistently:

```text
train_inputs      = inputs actually used by the internal latent / BoTorch model
train_inputs_raw  = original raw search-space inputs
```

For models without any transform or dimension reduction, these can contain the
same values. For transformed, mixed, high-dimensional, or input-perturbation
models, they are intentionally different.

This distinction matters because:

- fit helpers usually need the internal training input shape;
- visualization and candidate-update logic often need raw search-space inputs;
- high-dimensional wrappers may train on latent `Z` while accepting raw `X` at
  the public API boundary;
- mixed models must preserve categorical columns while transforming continuous
  columns.

### `posterior(X, ...)`

`posterior(X)` should return the prediction object expected by downstream
acquisition functions.

Examples:

| Family | `posterior(X)` should represent |
|---|---|
| Gaussian regression | predictive distribution of continuous response `y` |
| Binary classification | predictive Bernoulli probability / probability-scale posterior |
| Multiclass classification | class probability / simplex-like predictive representation |
| Ordinal regression | ordinal class probabilities or expected utility depending on wrapper design |
| Poisson non-Gaussian | response rate / count-scale posterior wrapper |
| Beta non-Gaussian | response mean / Beta observation-scale posterior wrapper |
| Gamma non-Gaussian | response mean / positive response posterior wrapper |
| Negative Binomial non-Gaussian | response mean / count-scale posterior wrapper |
| Hybrid multi-output | output collection or objective-space posterior assembled from submodels |

For acquisition functions that need the latent GP directly, provide
`latent_posterior(X)`.

### `latent_posterior(X, ...)`

Use `latent_posterior` when the model is trained through a latent GP but the
public `posterior` exposes a transformed response-scale distribution.

Typical examples:

- binary classification: latent `f` -> sigmoid probability;
- multiclass classification: class-wise latent GP -> class probabilities;
- ordinal regression: latent `f` -> cutpoint class probabilities;
- Poisson regression: latent `f` -> positive rate;
- Beta regression: latent `f` -> mean in `(0, 1)`;
- Gamma / Negative Binomial regression: latent `f` -> positive mean.

### `forward(X)`

For GP wrappers that are trained through GPyTorch MLL / ELBO, `forward(X)` should
return the latent GP distribution used by the likelihood during fitting. The
public prediction API should remain `posterior(X)`.

### `make_mll()`

Wrappers should expose a `make_mll()` method when there is a recommended training
objective.

Typical examples:

- exact Gaussian GP: `ExactMarginalLogLikelihood`;
- variational classification / ordinal / non-Gaussian GP: `VariationalELBO` or
  another approximate MLL;
- deep GP / deep kernel wrappers: family-specific MLL helpers when required.

This makes examples less ambiguous than manually constructing `mll` from nested
attributes.

### `condition_on_observations`

Where supported, `condition_on_observations(X, Y, ...)` should accept raw
search-space `X`, update the raw training set, apply target preparation, and
return a new wrapper instance with consistent transforms / likelihood state.

For non-Gaussian or classification models, observation noise passed through a
Gaussian-style `noise=` argument may be unsupported. In that case, the method
should raise `NotImplementedError` explicitly rather than silently ignoring it.

---

## Model families

## 1. Gaussian regression

Gaussian regression models are the standard continuous-output models. They are
used when the observation model is well approximated by a Gaussian likelihood.

Typical use cases:

- continuous response optimization;
- continuous response active learning;
- level-set estimation for scalar thresholds;
- multi-output regression through independent model lists or compatible
  multi-output wrappers.

Common variants:

- base exact GP wrappers;
- mixed continuous / categorical GP wrappers;
- DeepGP and Deep Kernel GP variants;
- high-dimensional wrappers such as PCA / REMBO / SAAS;
- robust relevance pursuit and heteroscedastic variants.

For standard Bayesian optimization, prefer BoTorch's existing acquisition
functions whenever the wrapper exposes a BoTorch-compatible posterior.

---

## 2. Non-Gaussian regression

Non-Gaussian regression models are used when the response distribution is not
well described by Gaussian observation noise.

Current families:

| Family | Target type | Typical link / response scale |
|---|---|---|
| Poisson | non-negative integer counts | latent `f` -> rate `lambda` |
| Beta | continuous values in `(0, 1)` | latent `f` -> mean `mu` |
| Gamma | positive continuous values | latent `f` -> positive mean |
| Negative Binomial | over-dispersed counts | latent `f` -> positive mean |

Layout:

```text
regression/non_gaussian/<family>/
├── base/
├── deep/
├── high_dim/
└── robust/
```

Design notes:

- The model-specific likelihood and posterior helpers live in
  `models/components/<family>.py`.
- The wrapper should expose `posterior(X)` on the response scale.
- The wrapper should expose `latent_posterior(X)` for acquisition functions that
  need latent GP uncertainty.
- `posterior.rsample()` should be interpreted as differentiable response-scale
  samples such as rate or mean samples, not necessarily raw observation samples.
- Raw observation samples such as Poisson counts may be non-reparameterized and
  should be provided through explicit helper methods when needed.

Acquisition notes:

- Custom non-Gaussian active-learning and level-set acquisitions live under
  `bochan.acquisition.non_gaussian`.
- Standard BO acquisitions such as qEI / qNEI / qUCB are not reimplemented if the
  response-scale posterior and objective are sufficient.

---

## 3. Binary classification

Binary classification models use a latent GP and a classification likelihood or
posterior wrapper to expose probability-scale predictions.

Typical use cases:

- feasibility modeling;
- binary constraints in Bayesian optimization;
- boundary exploration;
- active learning with BALD / entropy / margin uncertainty.

Important conventions:

- `forward(X)` returns latent GP quantities used for training.
- `latent_posterior(X)` returns the latent GP posterior.
- `posterior(X)` should expose probability-scale predictions when the wrapper is
  intended for acquisition functions.
- Mixed models should keep categorical columns unmodified by continuous input
  transforms.

Binary classification is kept under `classification/binary/` instead of
`classification/base/` because multiclass classification has different posterior
semantics and shape expectations.

---

## 4. Multiclass classification

Multiclass classification models extend the classification family to more than
two unordered classes.

Typical use cases:

- multiple discrete labels without ordinal ordering;
- class-probability objectives such as maximizing `p(target_class | x)`;
- multiclass constraints or feasibility categories;
- active learning with entropy, BALD, JointBALD, margin uncertainty, or
  probability variance;
- class-probability level-set estimation.

Important distinction from ordinal models:

- multiclass labels have no natural order;
- posterior values represent class probabilities;
- boundary logic is based on probability ambiguity or target-class thresholds,
  not ordered cutpoints.

Registry notes:

- `base`, `deepgp`, `deepkernel`, `saas`, `pca`, `rembo`, `rrp`, and `hetero` are
  registered for both normal and mixed multiclass inputs.
- `deepgpdeepkernel` is not currently a multiclass registry key.

---

## 5. Ordinal regression

Ordinal regression models are used when labels have an ordered structure, for
example class `0 < 1 < 2 < ...`.

Typical use cases:

- ordered quality levels;
- risk categories;
- thresholded material or process outcomes;
- ordinal constraints in Bayesian optimization;
- boundary-focused active learning and level-set estimation.

Important conventions:

- The model has latent GP values and ordered cutpoints / thresholds.
- `get_cutpoints()` or equivalent helper methods should expose the current
  cutpoints when possible.
- `posterior(X)` should expose ordinal class probabilities or a wrapper that can
  be converted to expected utility.
- Utility values may often be inferred as `0, 1, ..., K-1`, but explicit
  `utility_values` should be supported where acquisition functions need them.
- Boundary-aware acquisitions should support target boundary selection when
  applicable.

---

## 6. Hybrid multi-output models

Hybrid models combine outputs with different task semantics, for example:

```text
strength      -> regression
is_defect     -> binary
defect_type   -> multiclass
quality_rank  -> ordinal
```

`models/hybrid/` provides:

- output specifications that describe each output's task type and conversion
  metadata;
- prediction wrappers that preserve output names and task semantics;
- `HybridMultiOutputModel`, which delegates fitting / posterior calls to
  submodels and can expose objective-space predictions.

Typical use cases:

- multi-objective optimization over heterogeneous measurements;
- combining a continuous objective with binary feasibility;
- jointly proposing experiments for regression and ordered quality labels;
- serving a single model id that wraps several task-specific models.

Hybrid wrappers should not hide the submodel task semantics. Acquisition
objectives should use `OutputSpec` / `HybridObjectiveSpec` or output names rather
than relying on positional assumptions when names are available.

---

## 7. Components

`models/components/` contains reusable building blocks shared by wrappers.

Typical contents:

- likelihood classes;
- posterior wrapper classes;
- target-preparation utilities;
- input-transform helpers;
- mixed continuous / categorical helpers;
- decomposition / projection utilities;
- default covariance module builders;
- functions for aligning tensor shapes.

Component modules should not define high-level model-family policy. They should
provide small, reusable primitives that family wrappers compose.

---

## Input transforms and mixed variables

Input transforms should be passed as constructor arguments where possible.

`InputTransformConfig` in the API builds transforms through
`models/transforms/input.py`. The default behavior is `normalize=True` and
`perturbation=False`; input perturbation can be enabled independently.

For mixed continuous / categorical models:

- continuous columns may be normalized or transformed;
- categorical columns should remain unchanged;
- helper checks should raise an error if an input transform modifies categorical
  columns;
- `cat_dims` should be normalized and stored consistently.

For input perturbation:

- training transforms should not accidentally expand training data through
  perturbation samples;
- evaluation-time transforms may expand `q` to `q * n_w`;
- objective classes or acquisition helpers should aggregate expanded scores back
  to `q`;
- `ObjectiveConfig(n_w=...)` should match the perturbation count used by the
  model transform when risk aggregation is needed.

---

## High-dimensional wrappers

High-dimensional wrappers may use PCA, REMBO, SAAS, or related strategies.

Recommended state convention:

```text
train_inputs_raw          = raw X in the original search space
preproject_train_inputs   = transformed X before projection, when applicable
projected_train_inputs    = latent Z after projection, when applicable
train_inputs              = inputs actually used by the internal GP
```

Public `posterior(X)` should generally accept raw `X` unless the wrapper clearly
states that it expects latent projected inputs.

High-dimensional wrappers should also make feature-importance interpretation
explicit. For example, SAAS / ARD-style lengthscales should be interpreted in the
space actually used by the internal model, while PCA / REMBO importance may need
to be mapped back through the projection.

---

## Deep models

DeepGP and Deep Kernel GP wrappers should preserve the same public API:

- `forward(X)` for training-time latent distributions;
- `posterior(X)` for downstream predictive quantities;
- `latent_posterior(X)` when response-scale prediction is transformed through a
  likelihood;
- `input_transform` support before feature extraction where appropriate.

For mixed deep kernel models, continuous dimensions may be passed through the
feature extractor while categorical dimensions are handled by a categorical or
mixed kernel component.

---

## Robust and heteroscedastic models

Robust and heteroscedastic models should make the distinction between latent
uncertainty and observation noise explicit.

Typical use cases:

- avoiding noisy regions in candidate selection;
- modeling input-dependent observation variance;
- robust relevance pursuit or sparse feature selection;
- combining mean prediction with noise penalties in acquisition functions.

When a model has separate mean and noise components, wrappers should expose a
clear API so acquisition functions can access both without relying on fragile
private attributes.

Heteroscedastic classification and ordinal wrappers are expected to expose enough
noise or confidence information for `qHetero...` acquisitions to apply
`noise_mode` / `noise_combine` style behavior without knowing the model's private
implementation details.

---

## Minimal usage patterns

### Standard BoTorch-style training

```python
model = SomeGPModel(train_X=train_X, train_Y=train_Y, input_transform=input_tf)
mll = model.make_mll()
fit_func(mll)
posterior = model.posterior(test_X)
```

### Latent-posterior use

```python
latent_post = model.latent_posterior(test_X)
response_post = model.posterior(test_X)
```

Use this pattern for classification, multiclass, ordinal, and non-Gaussian models
where latent `f` and response-scale predictions have different meanings.

### Updating with new observations

```python
new_model = model.condition_on_observations(X=new_X, Y=new_Y)
```

The returned model should preserve model-family settings such as likelihood,
input transform, categorical dimensions, inducing point configuration, link
function settings, and output specifications.

---

## Implementation checklist for new models

When adding a new model wrapper, check the following:

- [ ] Does the wrapper expose `posterior(X)`?
- [ ] If there is a latent GP, does it expose `latent_posterior(X)`?
- [ ] Are `train_inputs` and `train_inputs_raw` consistent with the package
      convention?
- [ ] Does `forward(X)` return the correct training-time latent distribution?
- [ ] Is `make_mll()` implemented when there is a recommended MLL / ELBO?
- [ ] Does `condition_on_observations` preserve raw inputs and model settings?
- [ ] Are input transforms applied consistently at train and eval time?
- [ ] For mixed models, are categorical columns preserved?
- [ ] For high-dimensional models, is the raw-to-latent projection state stored?
- [ ] For hybrid models, are output names and task-specific output specs kept?
- [ ] Are tensor shapes compatible with q-batch acquisition functions?
- [ ] Are public class names aligned with the family naming policy?
- [ ] Are unsupported options rejected explicitly with `NotImplementedError` or
      `ValueError`?
- [ ] If the model should be reachable from the high-level API, is it registered
      in `bochan.api.model_registry`?

---

## Relationship with acquisition functions

Model wrappers should be designed so that acquisition functions do not need to
know implementation details. In practice:

- Gaussian regression should work with BoTorch's standard acquisitions whenever
  possible.
- Binary / multiclass / ordinal models often need probability or utility
  objectives.
- Non-Gaussian models should expose response-scale posterior quantities so
  standard BO acquisitions remain usable.
- Custom non-Gaussian active-learning and level-set acquisitions use
  `latent_posterior(X)` plus `model.likelihood(function_samples)` to estimate
  response-scale uncertainty.
- Heteroscedastic models should expose stable public access to noise-related
  predictions so `qHetero...` acquisitions do not depend on private attributes.
- Hybrid wrappers should expose an objective-space posterior for acquisitions
  that need all outputs on a common scalar / multi-objective scale.

Keeping this boundary clean makes it easier to add new acquisition functions
without rewriting model wrappers.
