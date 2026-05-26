# 00. Overview

`bochan` is an experimental BoTorch-oriented library for practical Bayesian optimization, active learning, and level-set estimation. Its theoretical foundation is the same surrogate-model loop used in standard Bayesian optimization, but the implementation scope is broader than ordinary continuous regression.

The central idea is to treat different model families through a common decision-making interface:

- regression models estimate continuous responses,
- classification models estimate class probabilities or latent decision functions,
- ordinal models estimate ordered-label probabilities through latent functions and cutpoints,
- multi-output models expose several response channels,
- objectives convert model outputs into scalar or vector scores,
- acquisition functions decide which candidate points should be evaluated next.

This document gives the top-level taxonomy used by the repository.

---

## 1. Core problem

A generic sequential design problem can be written as follows. At iteration `t`, we have observed

```text
D_t = {(x_i, y_i)}_{i=1}^t.
```

We fit a probabilistic surrogate model to `D_t`, compute an acquisition function `alpha_t(x)`, and choose the next point by

```text
x_{t+1} = argmax_x alpha_t(x).
```

For q-batch selection, we choose a batch of points

```text
X_{t+1} = {x_{t+1,1}, ..., x_{t+1,q}}.
```

In BoTorch notation this is represented as a tensor `X` with shape

```text
batch_shape x q x d
```

where `d` is the input dimension and `q` is the number of jointly selected candidates.

---

## 2. Three related goals

This repository treats Bayesian optimization, active learning, and level-set estimation as related but distinct sequential design problems.

### Bayesian optimization

Bayesian optimization searches for high-value inputs:

```text
maximize f(x).
```

The acquisition function trades off predicted performance and uncertainty. Examples include EI, NEI, UCB, KG, EHVI, and NEHVI.

### Active learning

Active learning selects observations that improve the surrogate model or reduce uncertainty. The point does not need to have high predicted objective value.

Typical criteria include predictive entropy, posterior variance, BALD, and integrated posterior variance.

### Level-set estimation

Level-set estimation tries to identify a region such as

```text
{x | f(x) >= tau}
```

or a boundary such as

```text
{x | f(x) = tau}.
```

This naturally leads to boundary-oriented acquisition functions such as Straddle, ICU-style criteria, and boundary variance criteria.

---

## 3. Why a unified library is useful

BoTorch already provides strong primitives, but practical projects often require combinations that are not a single standard class:

- q-batch optimization,
- mixed continuous and categorical variables,
- multi-output models,
- model lists,
- noisy observations,
- constraints,
- input perturbation,
- risk measures,
- classification and ordinal outputs,
- active learning and level-set estimation.

The goal of `bochan` is to make these combinations explicit and consistent rather than hiding them in one-off scripts.

The repository therefore emphasizes:

1. BoTorch-compatible model interfaces,
2. reusable objective classes,
3. acquisition functions with consistent argument names,
4. explicit tensor shape conventions,
5. family-specific implementations that still share common design rules.

---

## 4. Model output versus decision score

One important design rule is to separate the model output from the decision score.

A model may return:

- continuous posterior mean and variance,
- latent classification function values,
- class probabilities,
- ordinal class probabilities,
- multiple independent outputs,
- posterior samples.

An objective converts those outputs into the quantity that an acquisition function should optimize or aggregate.

Examples:

- regression objective: use the posterior sample directly,
- binary classification objective: convert latent values to probabilities or scores,
- ordinal objective: convert class probabilities to expected utility,
- multi-output objective: scalarize or preserve multiple outputs,
- risk objective: aggregate perturbed evaluations by mean, VaR, or CVaR.

This separation is especially important for classification and ordinal models because the latent function is not the same object as the final predictive label distribution.

---

## 5. Common abstraction

A practical acquisition pipeline can be written as:

```text
X candidate
  -> model.posterior(X)
  -> samples or predictive moments
  -> objective(samples, X)
  -> acquisition value
```

For input perturbation, the pipeline becomes:

```text
X candidate
  -> expanded perturbed inputs X_tilde with q * n_w points
  -> model.posterior(X_tilde)
  -> objective values for perturbed points
  -> aggregation from q * n_w back to q
  -> acquisition value
```

This is why `bochan` treats objective classes as first-class components rather than small anonymous callables.

---

## 6. Practical design priorities

The project is still evolving, so the highest priority is not backward compatibility. The highest priorities are:

1. consistent APIs across regression, classification, and ordinal models,
2. predictable tensor shapes,
3. BoTorch-like behavior where possible,
4. explicit handling of latent versus predictive quantities,
5. clean separation of model, objective, acquisition, and optimizer logic,
6. minimal examples that demonstrate each feature in isolation.

---

## 7. Recommended documentation map

The theory documents should be read as a layered specification:

- Gaussian process models define the probabilistic surrogate.
- Bayesian optimization defines the sequential loop.
- Acquisition functions define the decision criterion.
- Active learning and level-set estimation define non-optimization design goals.
- Classification and ordinal documents define non-Gaussian output handling.
- Multi-objective and constraint documents define practical design-space structure.
- Input perturbation and risk documents define robustness.
- Shape conventions define the implementation contract.

Together, these documents are intended to make the repository understandable as an extensible Bayesian optimization framework rather than a collection of unrelated acquisition functions.
