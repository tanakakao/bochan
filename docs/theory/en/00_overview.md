# 00. Overview and Reading Guide

This directory is the theory reference for `bochan`.  It is organized as a
small textbook rather than as a list of unrelated feature notes.

The purpose is to connect four layers that must remain consistent in practical
Bayesian optimization software:

1. the mathematical problem;
2. the probabilistic model;
3. the decision criterion;
4. the concrete `bochan` implementation.

Every chapter therefore contains both theory and an implementation
correspondence section.  The mathematical notation explains what a component
means.  The source map explains which class, method, tensor, or directory
implements that object.

---

## 1. Scope

`bochan` covers three sequential-design goals:

### Bayesian optimization

Find inputs with high utility:

```math
x^*\in\arg\max_{x\in\mathcal X}u(x).
```

### Active Learning

Choose observations that improve a model, reduce uncertainty, or maximize
information gain:

```math
x_{t+1}\in\arg\max_{x\in\mathcal X}
I(\text{future observation};\text{learning target}\mid\mathcal D_t).
```

### Level-set Estimation

Identify a region or boundary such as

```math
L_h^+=\{x:f(x)\ge h\},
\qquad
B_h=\{x:f(x)=h\}.
```

The same surrogate model can support all three goals, but the acquisition
function and evaluation loss are different.  A model that is appropriate for
optimization is not automatically being used correctly for Active Learning or
LSE.

---

## 2. Sequential-design loop

At iteration `t`, the observed dataset is

```math
\mathcal D_t
=\{(x_i,y_i)\}_{i=1}^{n_t}.
```

A probabilistic model defines a posterior

```math
p(f\mid\mathcal D_t).
```

An objective or posterior transform converts model outputs into the quantity of
interest.  An acquisition function evaluates the value of collecting data at a
candidate batch `X`:

```math
\alpha_t(X;\mathcal D_t).
```

The next batch is selected by

```math
X_{t+1}\in\arg\max_{X\in\mathcal X^q}\alpha_t(X;\mathcal D_t).
```

After the experiment or simulator returns new observations, the data are
appended and the cycle repeats.

In implementation terms:

```text
training data
    -> model construction
    -> model fitting
    -> posterior / samples
    -> objective or posterior transform
    -> acquisition function
    -> acquisition optimizer
    -> candidate post-processing
    -> experiment
    -> updated training data
```

`bochan` separates these stages so that model assumptions, user preferences,
and optimizer constraints are not silently mixed.

---

## 3. Four distinct mathematical spaces

A major source of implementation errors is treating different spaces as if they
were interchangeable.

### 3.1 Input space

The original design variable is

```math
x\in\mathcal X\subseteq\mathbb R^d
```

or a mixed continuous/categorical space.

A model may internally use

```math
z=T(x)
```

where `T` is normalization, PCA, REMBO, a VAE encoder, or a neural feature map.
Candidate optimization still needs a clearly defined search space and an
inverse or wrapper relationship to the original input.

### 3.2 Latent-function space

A GP commonly models

```math
f(x).
```

For regression this may be close to the observed response.  For classification
and ordinal models it is only a latent score.

### 3.3 Observation space

The likelihood maps latent values to observations:

```math
y\sim p(y\mid f(x)).
```

Examples are Gaussian responses, Bernoulli labels, categorical labels, ordered
classes, counts, and positive continuous measurements.

### 3.4 Decision or objective space

The user values a quantity

```math
u(x)=T_{\mathrm{decision}}[p(y\mid x,\mathcal D)].
```

Examples include:

- a regression response;
- probability of success;
- probability of satisfying a constraint;
- expected ordinal utility;
- a vector of multiple objectives;
- a risk measure under input perturbation.

A threshold, `best_f`, reference point, or constraint must be expressed in the
same space as the acquisition value that consumes it.

---

## 4. Model taxonomy

The repository supports several response types.

| Response type | Typical likelihood | Main predictive object |
|---|---|---|
| Continuous Gaussian | Gaussian | mean, covariance, samples |
| Bounded continuous | Beta | bounded response distribution |
| Positive continuous | Gamma | positive response distribution |
| Count | Poisson / Negative Binomial | count distribution |
| Binary label | Bernoulli | class probability and latent score |
| Unordered multiclass label | Categorical / softmax | class-probability vector |
| Ordered class | ordered logit | class probabilities, cutpoints, utility |
| Multiple homogeneous outputs | shared or independent likelihoods | vector posterior |
| Heterogeneous outputs | task-specific likelihoods | transformed objective vector |

The same output type can be combined with:

- exact or variational inference;
- continuous or mixed inputs;
- homoscedastic or heteroscedastic noise;
- Deep Kernel or DeepGP structure;
- high-dimensional priors or projections;
- single-output, multi-output, or hybrid wrappers.

---

## 5. Decision-component taxonomy

`bochan` distinguishes the following roles.

### Model

Represents uncertainty about latent functions or predictive responses.

### Likelihood

Defines the observation distribution conditional on latent variables.

### Posterior transform

Changes the representation of a posterior, for example from latent logits to a
selected probability or utility.

### MC objective

Maps posterior samples to scalar or vector decision values.

### Acquisition function

Assigns value to collecting one or more observations.

### Acquisition optimizer

Maximizes the acquisition subject to search-space constraints.

### Candidate post-processing

Applies rounding, repair, sparsity, or domain-specific validity rules after or
during optimization.

These roles should not be collapsed.  For example, a linear constraint on `x`
is an optimizer constraint, whereas a probability of experimental feasibility
is a modeled outcome constraint.

---

## 6. Book structure

The chapters are intentionally divided so that each topic has one primary
home.

### Part I: foundations and sequential decisions

| Chapter | File | Primary responsibility |
|---|---|---|
| 00 | `00_overview.md` | terminology, architecture, and reading guide |
| 01 | `01_gaussian_process_models.md` | GP probability foundations, conditioning, inference, and posterior contracts |
| 02 | `02_bayesian_optimization.md` | optimization problem, regret, sequential loop, q-batch, and noisy decisions |
| 03 | `03_acquisition_functions.md` | BO acquisition mathematics and acquisition optimization |
| 04 | `04_active_learning.md` | information and uncertainty acquisition for model learning |
| 05 | `05_level_set_estimation.md` | LSE problem definition, loss functions, confidence sets, and evaluation |
| 06 | `06_classification_and_ordinal_bo.md` | decision objectives for binary, multiclass, and ordinal BO |
| 07 | `07_multi_objective_and_constraints.md` | Pareto optimization, hypervolume, scalarization, and constraints |
| 08 | `08_input_perturbation_and_risk.md` | robust objectives, chance constraints, VaR, CVaR, and perturbation sampling |
| 09 | `09_shape_conventions.md` | tensor-axis and interface contracts |

### Part II: model families and implementation details

| Chapter | File | Primary responsibility |
|---|---|---|
| 10 | `10_regression_models_and_likelihoods.md` | Gaussian and non-Gaussian regression models |
| 11 | `11_classification_models.md` | binary and multiclass model theory |
| 12 | `12_ordinal_models.md` | ordered-logit models, cutpoints, and ordinal uncertainty |
| 13 | `13_heteroscedastic_and_robust_models.md` | input-dependent noise, outliers, and robust likelihood interpretation |
| 14 | `14_deep_and_high_dimensional_models.md` | DKL, DeepGP, SAAS, PCA, REMBO, and VAE-GP |
| 15 | `15_heterogeneous_multi_output.md` | heterogeneous likelihoods and hybrid posterior construction |
| 16 | `16_level_set_mathematics_and_implementation.md` | task-specific LSE acquisition formulas and source correspondence |

The introductory chapters define concepts once.  Detailed chapters then refer
back to those definitions and focus on model- or implementation-specific
behavior.

---

## 7. Suggested reading paths

### First use of Gaussian-process Bayesian optimization

1. Chapter 00
2. Chapter 01
3. Chapter 10
4. Chapter 02
5. Chapter 03
6. Chapter 09

### Classification or ordinal optimization

1. Chapters 00 and 01
2. Chapter 11 or 12
3. Chapter 06
4. Chapter 03
5. Chapter 09

### Active Learning

1. Chapters 01 and 04
2. the relevant model chapter: 10, 11, or 12
3. Chapter 09
4. Chapter 13 when observation noise is input dependent

### Level-set Estimation

1. Chapter 05
2. the relevant model chapter
3. Chapter 16
4. Chapters 08 and 09 for robust or perturbed LSE

### Multi-objective and heterogeneous-output problems

1. Chapter 07
2. Chapter 15
3. Chapters 06 and 12 when probabilities or ordinal utilities are objectives
4. Chapter 09

### High-dimensional or deep models

1. Chapters 01 and 10
2. Chapter 14
3. Chapters 02 and 03
4. Chapter 09

---

## 8. Notation

| Symbol | Meaning |
|---|---|
| $n$ | number of observations |
| $d$ | input dimension |
| $q$ | number of candidates selected jointly |
| $m$ | number of outputs or objective dimensions |
| $K$ | number of classes |
| $n_w$ | number of input-perturbation samples per nominal candidate |
| $X$ | candidate tensor or design matrix |
| $\mathcal D_t$ | observed data at iteration `t` |
| $f$ | latent function |
| $y$ | observed response |
| $u$ | utility or objective value |
| $k$ | covariance kernel |
| $\mu,\Sigma$ | posterior mean and covariance |
| $\alpha$ | acquisition function; also a risk level when context is explicit |
| $h$ | level-set threshold |

When the same Greek symbol is common in two fields, the chapter states the
local meaning explicitly.  For example, `alpha` can denote an acquisition
function or a tail probability.

---

## 9. Implementation architecture

The main source directories mirror the mathematical separation.

```text
src/bochan/models/          probabilistic models and posterior wrappers
src/bochan/models/*/            family-owned observation likelihoods and model helpers
src/bochan/fit/             model-specific fitting procedures
src/bochan/acquisition/     BO, Active Learning, and LSE acquisitions
src/bochan/optim/           acquisition optimizers and candidate repair
src/bochan/api/             high-level configuration and registries
```

Representative high-level registries are:

```text
src/bochan/api/model_registry.py
src/bochan/api/acquisition_registry.py
```

The registry resolves user-facing model and acquisition names to concrete
classes.  The theoretical meaning is defined by the resolved class, not by the
alias alone.

---

## 10. Posterior-space contract

For every model family, these questions must be answered.

1. What random variable does `posterior(X)` represent?
2. Is it latent, predictive, probability, utility, or a proxy posterior?
3. Does the variance include observation noise?
4. Is `rsample()` differentiable with respect to `X`?
5. What is the final output axis?
6. Are extra model-batch, task, or DeepGP sample axes present?
7. Which input transform is applied during training and evaluation?
8. Can the model be conditioned or fantasized for look-ahead acquisitions?

Current important differences include:

- binary classification `posterior()` returns a probability-space posterior;
- multiclass classification `posterior()` returns class probabilities;
- the base ordinal `posterior()` returns a latent scalar posterior and
  `class_probs()` returns ordered-class probabilities;
- `HybridPosterior` stores marginal means and variances in a common output
  space but does not represent cross-output covariance.

These are implementation contracts, not incidental details.

---

## 11. Documentation convention for each chapter

Each chapter should contain:

1. a precise problem statement;
2. assumptions and notation;
3. core derivations or formulas;
4. interpretation of uncertainty;
5. failure modes and non-equivalent alternatives;
6. connection to BO, Active Learning, or LSE where relevant;
7. tensor-shape consequences;
8. an implementation correspondence table;
9. references for further study.

Code examples are used to explain interfaces, not to replace the mathematical
specification.

---

## 12. Status of the theory reference

This reference documents the current repository implementation.  Some
components are exact implementations of standard statistical models.  Others
are engineering approximations designed to provide a BoTorch-compatible
posterior or acquisition interface.

The chapters explicitly label important approximations, including:

- residual-based heteroscedastic noise fitting;
- independent normal proxy sampling in `HybridPosterior`;
- moment reduction of DeepGP sample dimensions;
- acquisition penalties used as practical diversity mechanisms;
- score-level robust aggregation that is not identical to a fully specified
  generative model.

When implementation and textbook terminology differ, the implemented formula
and posterior contract take precedence.