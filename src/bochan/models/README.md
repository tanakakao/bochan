# models package

`bochan.models` contains BoTorch-style surrogate-model wrappers used by the
acquisition package and higher-level optimization / active-learning workflows.

The package is organized along two axes:

1. **model family / response semantics** — regression, classification, ordinal,
   hybrid, external, and related concrete model implementations;
2. **cross-cutting strategy** — multi-task, independent multi-output, and
   multi-fidelity mechanics that can be reused by several concrete families.

The canonical ownership rules are documented in `ARCHITECTURE.md`.

---

## Design goals

The main goals of this package are:

1. Keep a BoTorch-like public interface.
2. Separate raw search-space inputs from transformed / latent-model inputs.
3. Keep posterior semantics explicit across regression, classification, ordinal,
   and non-Gaussian response families.
4. Make advanced variants discoverable through regular family-owned paths.
5. Keep cross-family mechanics in dedicated strategy packages rather than
   duplicating them in each model family.
6. Avoid model-specific acquisition logic when a BoTorch-compatible posterior or
   objective is sufficient.
7. Keep high-level registry behavior explicit and lazy.

A model wrapper should generally provide:

- `posterior(X, ...)`;
- `latent_posterior(X, ...)` when the model has a latent GP layer;
- `condition_on_observations(X, Y, ...)` when supported;
- `make_mll()` for the recommended marginal log-likelihood / ELBO;
- `train_inputs`;
- `train_inputs_raw`;
- `train_targets`;
- `input_transform` support where applicable.

---

## Current layout

The current broad package layout is:

```text
models/
├── components/          # shared likelihood/posterior/decomposition primitives
├── transforms/          # shared input-transform builders
├── regression/
│   ├── gaussian/
│   ├── beta/
│   ├── gamma/
│   ├── count/
│   ├── external/
│   ├── foundation/
│   ├── neural/
│   └── multioutput.py
├── classification/
│   ├── binary/
│   ├── multiclass/
│   └── common/
├── ordinal/
├── hybrid/
├── multitask/
├── multioutput/
├── multifidelity/
└── external/
```

Important ownership rules:

- `regression/gaussian/` owns standard continuous Gaussian-response models.
- `regression/beta/` and `regression/gamma/` own continuous non-Gaussian
  response families.
- `regression/count/{poisson,negative_binomial}/` owns count-response families.
- `regression/external/`, `regression/foundation/`, and `regression/neural/` own
  regression-specific integrations such as external estimators, PFN-style
  foundation models, and deep ensembles.
- `classification/binary/` and `classification/multiclass/` remain separate
  because their likelihoods, posterior semantics, target labels, and
  acquisitions differ.
- `classification/common/` contains shared classification internals rather than
  a concrete public task family.
- `ordinal/` is a first-class family because cutpoints, ordered probabilities,
  and boundary-aware acquisitions are central to its semantics.
- `hybrid/` combines heterogeneous task families while preserving output task
  metadata.
- `multitask/` owns correlated task/output mechanics and task-feature adapters.
- `multioutput/` owns wrappers that aggregate independently fitted outputs.
- `multifidelity/` owns reusable fidelity-axis abstractions and adapters.
- concrete likelihood-specific multi-task / multi-fidelity implementations stay
  with their owning family when they are not genuinely cross-family.

This means **multi-output does not imply multi-task correlation**.

---

## Common family convention

GP-oriented response families commonly use subpackages such as:

| Directory | Meaning |
|---|---|
| `base/` | Standard wrappers and core likelihood / posterior integration. |
| `deep/` | DeepGP or Deep Kernel GP variants. |
| `high_dim/` | PCA, REMBO, SAAS, VAE, or related high-dimensional wrappers. |
| `robust/` | Heteroscedastic, robust relevance pursuit, or noise-aware variants. |

This convention is intentionally not forced onto external / foundation / neural
integrations when a different grouping is clearer.

---

## High-level API registry support

The canonical high-level model registry lives in
`bochan.api.registry.model`. It is lazy and is indexed by:

```text
input_type -> task_type -> model_type
```

The exact `model_type` set is **task-dependent** and changes as families are
added. The registry itself is the source of truth; this README intentionally does
not maintain a second exhaustive flat list.

Representative normal-input regression strategies currently include:

- GP families: `base`, `kronecker`, `multitask`, `multifidelity`, `deepgp`,
  `deepkernel`, `deepgpdeepkernel`, `saas`, `pca`, `rembo`, `vae`, `rrp`,
  `hetero`;
- external / neural / foundation families: `lightgbm`, `lightgbm_ensemble`,
  `ngboost`, `ngboost_ensemble`, `random_forest`, `deep_ensemble`, `pfn`,
  `tabpfn`;
- distribution-specific regression keys prefixed by `beta_`, `gamma_`,
  `poisson_`, and `negative_binomial_`.

Binary, multiclass, ordinal, multi-objective, mixed-input, multi-task, and
multi-fidelity registries expose the subset appropriate to their task contracts.
For example, a model key that exists for regression should not be assumed to be
available for multiclass or ordinal tasks unless it is registered there.

When `cat_dims` is supplied, the API may infer mixed-input handling where the
requested model family supports it.

---

## Core API conventions

### `train_inputs` and `train_inputs_raw`

Use the following convention consistently:

```text
train_inputs      = inputs actually used by the internal latent / BoTorch model
train_inputs_raw  = original raw search-space inputs
```

For models without a transform or dimension reduction, these may contain the
same values. For transformed, mixed, high-dimensional, or input-perturbation
models they are intentionally different.

This distinction matters because:

- fit helpers usually need the internal training-input shape;
- visualization and candidate-update logic often need raw search-space inputs;
- high-dimensional wrappers may train on latent `Z` while accepting raw `X` at
  the public boundary;
- mixed models must preserve categorical columns while transforming continuous
  columns.

### `posterior(X, ...)`

`posterior(X)` should return the prediction object expected by downstream
acquisition functions, but its semantics depend on the model family.

| Family | Main public posterior semantics |
|---|---|
| Gaussian regression | continuous response predictive distribution |
| Binary classification | probability-scale prediction / posterior |
| Multiclass classification | class-probability representation |
| Ordinal regression | latent score or task-specific ordinal posterior contract; class probabilities are exposed separately where required |
| Poisson / count regression | response rate / count-scale posterior wrapper |
| Beta regression | response mean / Beta observation-scale posterior wrapper |
| Gamma regression | positive response posterior wrapper |
| Hybrid multi-output | task-aware output collection or objective-space posterior |

Do not infer epistemic uncertainty merely from a common `[..., q, m]` shape.
Independent ModelList outputs, correlated multi-task outputs, and heterogeneous
transformed outputs have different covariance semantics.

### `latent_posterior(X, ...)`

Use `latent_posterior` when the model is trained through a latent GP but the
public prediction is transformed through a likelihood or link function.

Typical examples:

- binary classification: latent `f` -> sigmoid probability;
- multiclass classification: class-wise latent GP -> class probabilities;
- ordinal regression: latent `f` -> cutpoint probabilities;
- Poisson regression: latent `f` -> positive rate;
- Beta regression: latent `f` -> mean in `(0, 1)`;
- Gamma / Negative Binomial regression: latent `f` -> positive mean.

### `forward(X)`

For GPyTorch-trained wrappers, `forward(X)` should return the latent distribution
used by the likelihood during fitting. The public prediction API should remain
`posterior(X)`.

### `make_mll()`

Wrappers should expose `make_mll()` when there is a recommended training
objective.

Typical examples:

- exact Gaussian GP: `ExactMarginalLogLikelihood`;
- variational classification / ordinal / non-Gaussian GP: `VariationalELBO` or
  another family-appropriate approximate MLL;
- deep wrappers: family-specific MLL helpers where required.

### `condition_on_observations`

Where supported, `condition_on_observations(X, Y, ...)` should accept raw
search-space `X`, update the raw training set, apply target preparation, and
return a new wrapper with consistent transforms / likelihood state.

Unsupported Gaussian-style `noise=` arguments for non-Gaussian or classification
families should raise explicitly instead of being silently ignored.

---

## Model families

### Gaussian regression

Gaussian regression is the standard continuous-output family. Typical variants
include exact GP, mixed-input GP, DeepGP / Deep Kernel GP, PCA / REMBO / SAAS /
VAE high-dimensional wrappers, multi-task / multi-fidelity forms, robust
relevance pursuit, and heteroscedastic models.

Prefer BoTorch standard acquisitions whenever the model exposes a compatible
posterior.

### Non-Gaussian regression

Current non-Gaussian response ownership is:

| Family | Path | Target type |
|---|---|---|
| Beta | `regression/beta/` | continuous values in `(0, 1)` |
| Gamma | `regression/gamma/` | positive continuous values |
| Poisson | `regression/count/poisson/` | non-negative integer counts |
| Negative Binomial | `regression/count/negative_binomial/` | over-dispersed counts |

These families commonly use `base/`, `deep/`, `high_dim/`, and `robust/`
subpackages. Multi-task and Kronecker variants that are specific to a likelihood
remain under that likelihood's family-owned path.

Custom non-Gaussian active-learning and level-set acquisitions live under
`bochan.acquisition.non_gaussian`. Standard BO acquisitions are not reimplemented
when the response-scale posterior and objective are sufficient.

### Binary classification

Binary classification uses a latent GP plus a probability-scale prediction
contract. Typical uses include feasibility modeling, binary constraints,
boundary exploration, and active learning with entropy / BALD / margin criteria.

### Multiclass classification

Multiclass labels are unordered. Posterior semantics are class probabilities,
and target-class BO / LSE operates on selected class probabilities rather than
ordered cutpoints.

### Ordinal regression

Ordinal models represent ordered labels with latent scores and cutpoints.
Class-probability and expected-utility operations should preserve that ordering
rather than treating the task as unordered multiclass classification.

### Hybrid multi-output models

Hybrid models combine heterogeneous outputs, for example:

```text
strength      -> regression
is_defect     -> binary
defect_type   -> multiclass
quality_rank  -> ordinal
```

Hybrid wrappers should preserve output names and task semantics so objectives and
serving layers do not depend on fragile positional assumptions.

### Multi-task, multi-output, and multi-fidelity

These are deliberately separate concepts:

- `multitask/`: correlated tasks/outputs, task features, and shared covariance
  structure;
- `multioutput/`: independent output models collected behind one interface;
- `multifidelity/`: explicit fidelity dimensions and shared fidelity mechanics.

Use the package that matches the statistical relationship being modeled rather
than treating all vector-valued outputs as the same abstraction.

---

## Input transforms and mixed variables

`InputTransformConfig` builds shared transforms through
`models/transforms/input.py`.

For mixed continuous / categorical models:

- continuous columns may be normalized or transformed;
- categorical columns must remain valid category coordinates;
- helper checks should reject transforms that modify categorical columns;
- `cat_dims` should be normalized and stored consistently.

For input perturbation:

- training transforms must not accidentally expand training data through
  perturbation samples;
- evaluation-time transforms may expand `q` to `q * n_w`;
- objectives or acquisition helpers aggregate expanded scores back to `q`;
- `ObjectiveConfig(n_w=...)` should match the perturbation count when risk
  aggregation is required.

---

## High-dimensional wrappers

High-dimensional models may use PCA, REMBO, SAAS, VAE, or related strategies.
Recommended state terminology is:

```text
train_inputs_raw          = raw X in the original search space
preproject_train_inputs   = transformed X before projection, when applicable
projected_train_inputs    = latent Z after projection, when applicable
train_inputs              = inputs actually used by the internal model
```

Public `posterior(X)` should generally accept raw `X` unless a wrapper explicitly
states that it expects latent projected inputs.

Feature-importance interpretation must be tied to the space where the parameter
is defined. PCA loadings, REMBO projections, SAAS / ARD lengthscales, and deep
latent lengthscales are not automatically raw-feature importance scores.

---

## Robust and heteroscedastic models

Robust and heteroscedastic wrappers should distinguish latent epistemic
uncertainty from observation noise. If the model has separate mean and noise
components, downstream acquisitions should access them through stable public
contracts rather than fragile private attributes.

---

## Minimal usage patterns

### Standard BoTorch-style training

```python
model = SomeGPModel(train_X=train_X, train_Y=train_Y, input_transform=input_tf)
mll = model.make_mll()
fit_func(mll)
posterior = model.posterior(test_X)
```

### Latent / response posterior

```python
latent_post = model.latent_posterior(test_X)
response_post = model.posterior(test_X)
```

### Updating with new observations

```python
new_model = model.condition_on_observations(X=new_X, Y=new_Y)
```

The returned model should preserve family settings such as likelihood, input
transform, categorical dimensions, inducing-point configuration, link-function
settings, and output specifications.

---

## Implementation checklist for new models

- [ ] Does the wrapper expose `posterior(X)`?
- [ ] If there is a latent GP, does it expose `latent_posterior(X)`?
- [ ] Are `train_inputs` and `train_inputs_raw` consistent with the package convention?
- [ ] Does `forward(X)` return the correct training-time latent distribution?
- [ ] Is `make_mll()` implemented when there is a recommended MLL / ELBO?
- [ ] Does `condition_on_observations` preserve raw inputs and model settings?
- [ ] Are input transforms applied consistently at train and eval time?
- [ ] For mixed models, are categorical columns preserved?
- [ ] For projected models, is raw-to-latent state stored explicitly?
- [ ] For hybrid models, are output names and task-specific specs preserved?
- [ ] Are tensor shapes compatible with q-batch acquisitions?
- [ ] Are unsupported options rejected explicitly?
- [ ] If the model should be public, is it registered in `bochan.api.registry.model`?
- [ ] Does the model live in the family / strategy package that owns its semantics?

---

## Relationship with acquisition functions

Model wrappers should expose enough stable posterior semantics that acquisition
functions do not need model-private implementation details.

- Gaussian regression should use BoTorch standard acquisitions whenever possible.
- Binary / multiclass / ordinal models often require probability or utility
  objectives.
- Non-Gaussian models should expose response-scale posterior quantities so
  standard BO acquisitions remain usable when mathematically appropriate.
- Custom non-Gaussian active-learning / LSE acquisitions may use
  `latent_posterior(X)` plus the likelihood to estimate response uncertainty.
- Heteroscedastic models should expose stable public access to noise-related
  predictions.
- Hybrid wrappers should expose task-aware or objective-space predictions for
  downstream multi-output acquisitions.

Keeping this boundary clean makes it easier to add models and acquisitions
without rebuilding adapters or duplicating cross-family logic.
