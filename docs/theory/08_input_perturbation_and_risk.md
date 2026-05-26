# 08. Input Perturbation and Risk

Input perturbation and risk aggregation are important when the selected input cannot be executed exactly or when robustness matters more than nominal performance.

---

## 1. Motivation

Standard Bayesian optimization evaluates a nominal input:

```text
x -> f(x).
```

In real experiments, the executed condition may differ from the intended condition:

```text
x_tilde = x + w.
```

where `w` is an input perturbation.

A candidate that is optimal only at exactly `x` may be poor if small execution errors occur. Robust optimization therefore evaluates a distribution around `x`, not only the point itself.

---

## 2. Perturbed objective

Let `w` be sampled from a perturbation distribution. The perturbed response is:

```text
f(x + w).
```

For each nominal candidate `x`, we may sample `n_w` perturbations:

```text
x_tilde_1, ..., x_tilde_nw.
```

The objective then aggregates these values into one robust score.

---

## 3. Mean robustness

The simplest robust objective is the expected value under perturbation:

```text
R_mean(x) = E_w[f(x + w)].
```

This searches for conditions that perform well on average around the nominal input.

It does not specifically protect against bad-tail outcomes.

---

## 4. Worst-case robustness

A conservative robust objective is:

```text
R_worst(x) = min_w f(x + w)
```

for maximization.

This protects against poor local outcomes but can be too pessimistic, especially when the perturbation distribution contains rare extreme samples.

---

## 5. Value at Risk

For maximization, Value at Risk can be interpreted as a lower quantile of the perturbed outcome distribution.

For confidence level `alpha`, a practical definition is:

```text
VaR_alpha(x) = quantile_{1 - alpha}(f(x + w)).
```

This asks how bad the outcome can be at a specified tail probability.

The exact convention must be documented because VaR definitions differ between minimization and maximization contexts.

---

## 6. Conditional Value at Risk

Conditional Value at Risk aggregates the bad tail instead of using only a quantile:

```text
CVaR_alpha(x) = E[f(x + w) | f(x + w) is in the bad tail].
```

For maximization, CVaR focuses on the lower tail. It is often more stable and informative than VaR because it uses all tail samples rather than only the quantile boundary.

---

## 7. Shape expansion

Input perturbation changes the q-batch shape.

Without perturbation:

```text
X shape = batch_shape x q x d
```

With `n_w` perturbations per candidate:

```text
X_tilde shape = batch_shape x (q * n_w) x d
```

The objective must aggregate the perturbed dimension back to the original q-batch dimension:

```text
batch_shape x (q * n_w) -> batch_shape x q
```

This is one of the most important implementation contracts in `bochan`.

---

## 8. Classification and ordinal perturbation

For classification, perturbation may be applied before converting latent values to probabilities, or after obtaining predictive class scores depending on the model interface.

Useful robust scores include:

```text
E_w[P(y = 1 | x + w)]
VaR_w[P(y = 1 | x + w)]
CVaR_w[P(y = 1 | x + w)]
```

For ordinal models, robust expected utility can be written as:

```text
E_w[sum_k u_k P(y = k | x + w)].
```

Again, the objective should document whether it aggregates probabilities, utilities, latent values, or posterior samples.

---

## 9. Interaction with acquisition functions

Input perturbation can be introduced in different places:

1. transform candidate inputs before the model posterior,
2. use a posterior transform or objective to aggregate perturbations,
3. define a risk-aware objective used inside an acquisition function,
4. post-process selected candidates for robust feasibility.

The preferred `bochan` design is to keep risk aggregation in reusable objective classes when possible. This allows the same risk-aware objective to be reused across EI-like, UCB-like, active-learning, and multi-output acquisition functions.

---

## 10. Practical design guidance

A perturbation-aware objective or acquisition should document:

1. the perturbation distribution,
2. the number of perturbations `n_w`,
3. whether perturbations are applied to all dimensions or only selected dimensions,
4. how categorical dimensions are handled,
5. whether the aggregation is mean, worst-case, VaR, CVaR, or custom,
6. the maximization/minimization convention,
7. how `q * n_w` is reduced to `q`,
8. whether posterior samples or posterior means are aggregated,
9. whether the returned values are scalar or multi-output.

Robust optimization is powerful, but ambiguous conventions can easily produce the wrong candidate. The documentation should therefore be explicit even when the implementation seems straightforward.
