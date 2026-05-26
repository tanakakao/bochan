# 06. Classification and Ordinal Bayesian Optimization

Many practical optimization problems do not produce only continuous regression targets. Outputs may be binary labels, multiclass labels, ordered grades, or mixtures of continuous and discrete responses. This document explains the theoretical assumptions used when applying Bayesian optimization, active learning, and level-set estimation to classification and ordinal models.

---

## 1. Binary classification

In binary classification,

```text
y in {0, 1}.
```

A common GP classification model uses a latent function:

```text
f(x) ~ GP(m(x), k(x, x'))
```

and maps it to a class probability:

```text
p(y = 1 | x) = sigmoid(f(x)).
```

The latent value `f(x)` and the probability `p(y = 1 | x)` are different quantities.

This distinction matters because an acquisition function may use either:

- latent posterior uncertainty, or
- predictive class probability uncertainty.

---

## 2. Latent uncertainty versus probability uncertainty

A binary classifier can be uncertain for different reasons:

1. the latent function has high posterior variance,
2. the predicted probability is close to 0.5,
3. the model parameters or inducing representation are uncertain,
4. input perturbation changes the predicted class.

For example, a point can have high latent variance but still produce a saturated probability after a sigmoid transform. Conversely, a point near probability 0.5 may be a decision-boundary point even if latent variance is not maximal.

This is why API names and documentation should specify whether an acquisition uses latent or predictive quantities.

---

## 3. Classification objectives

For Bayesian optimization with classification outputs, the objective may be a score derived from class probabilities.

Examples:

```text
score(x) = p(y = 1 | x)
score(x) = logit probability
score(x) = class utility expectation
score(x) = probability of satisfying a constraint class
```

For binary classification used as a constraint, the class probability can be used as a feasibility probability. For classification used as an objective, the probability or utility should be clearly defined.

---

## 4. Multiclass classification

In multiclass classification,

```text
y in {0, 1, ..., K - 1}.
```

The model may produce class probabilities:

```text
p(y = k | x),  k = 0, ..., K - 1.
```

A decision score can be defined by class utilities:

```text
U(x) = sum_k u_k p(y = k | x).
```

where `u_k` is the utility of class `k`.

This utility view makes multiclass classification compatible with scalar Bayesian optimization.

---

## 5. Ordinal regression

Ordinal regression is different from ordinary multiclass classification because labels have an order:

```text
y in {0, 1, ..., K - 1},  0 < 1 < ... < K - 1.
```

A common latent-threshold formulation uses cutpoints:

```text
c_0 < c_1 < ... < c_{K-2}.
```

The class probability is:

```text
P(y = k | x) = P(c_{k-1} < f(x) <= c_k).
```

with implicit lower and upper bounds for the first and last classes.

---

## 6. Ordinal expected utility

Because ordinal classes are ordered, it is natural to assign utility values:

```text
u = [u_0, u_1, ..., u_{K-1}].
```

The expected utility is:

```text
EU(x) = sum_k u_k P(y = k | x).
```

If no custom utility is provided, a simple default is:

```text
u_k = k.
```

This makes ordinal predictions compatible with scalar objectives while preserving the ordered-label interpretation.

---

## 7. Ordinal level-set estimation

Ordinal level-set estimation often targets class boundaries. The boundaries are represented by cutpoints, not arbitrary class labels.

A boundary acquisition may target:

```text
f(x) ~= c_j
```

for a selected boundary index `j`.

This explains arguments such as:

```text
target_boundary_idx
boundary_reduction
utility_values
```

- `target_boundary_idx` chooses a specific cutpoint.
- `boundary_reduction` defines how to aggregate multiple boundaries.
- `utility_values` defines the score used when ordinal output is treated as an objective.

---

## 8. Hybrid model lists

Practical BO problems may combine:

- regression objective outputs,
- binary feasibility outputs,
- ordinal quality grades,
- multiclass category predictions.

A model list can represent these outputs as separate surrogate models. A common objective layer can then convert them into comparable scores.

Example:

```text
output 1: regression value to maximize
output 2: binary probability of feasibility
output 3: ordinal expected utility
```

The acquisition function should not need to know the internal likelihood of each model if the objective layer exposes the intended decision values.

---

## 9. Design guidance

Classification and ordinal BO components should document:

1. whether `posterior(X)` returns latent values or predictive probabilities,
2. whether sigmoid, softmax, or cutpoint conversion is applied internally,
3. how class utilities are defined,
4. how multi-output predictions are stacked,
5. whether the acquisition expects latent or probability values,
6. whether q-batch sampling is supported,
7. how input perturbation is aggregated,
8. how constraints should use classification probabilities.

Without these conventions, it is easy to accidentally optimize a latent score when the intended quantity was a probability or utility.
