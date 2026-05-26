# 05. Level-set Estimation

Level-set estimation is the task of identifying where a function is above, below, or near a threshold. It is closely related to active learning and Bayesian optimization, but the objective is different.

---

## 1. Problem definition

Given an unknown function `f`, a threshold `tau`, and a design space `X`, a level set can be written as:

```text
L_tau = {x in X | f(x) >= tau}.
```

The corresponding boundary is:

```text
B_tau = {x in X | f(x) = tau}.
```

The goal is not necessarily to find the maximum of `f`. The goal is to classify the design space relative to the threshold or identify the boundary accurately.

---

## 2. Why standard BO is not enough

Bayesian optimization tends to focus observations near high objective values. That may be inefficient for level-set estimation.

Example:

- If the goal is to find all safe process conditions, we need the safe/unsafe boundary.
- If the goal is to identify defect transition regions, we need samples around the transition.
- If the goal is to characterize a feasible region, sampling only the optimum is not enough.

Therefore, level-set estimation requires boundary-aware acquisition functions.

---

## 3. Regression level-set estimation

For a regression GP posterior, we use:

```text
mu(x) = E[f(x) | D]
sigma(x) = Std[f(x) | D].
```

A classic boundary-oriented criterion is Straddle:

```text
alpha(x) = beta * sigma(x) - |mu(x) - tau|.
```

This favors points where:

1. the model is uncertain, and
2. the predicted mean is close to the threshold.

The threshold term avoids spending observations in regions that are uncertain but far from the boundary.

---

## 4. Classification boundary estimation

For binary classification, the decision boundary often corresponds to:

```text
p(y = 1 | x) = 0.5
```

or, for a latent GP classifier,

```text
f(x) = 0.
```

These are related but not always equivalent in implementation because some wrappers return latent posteriors while others return probabilities.

Boundary criteria must document whether they use:

- latent mean and variance,
- class probability,
- predictive entropy,
- BALD or mutual information,
- margin uncertainty.

---

## 5. Ordinal boundary estimation

Ordinal regression has ordered categories and cutpoints. If the latent function is `f(x)` and cutpoints are

```text
c_0 < c_1 < ... < c_{K-2},
```

then each cutpoint is a class boundary.

A boundary-oriented ordinal acquisition may target:

- one specific cutpoint,
- all cutpoints evenly,
- boundaries around high-utility classes,
- class-probability ambiguity.

This is why arguments such as `target_boundary_idx` and `boundary_reduction` are useful. They make explicit whether the acquisition is searching one boundary or aggregating across several boundaries.

---

## 6. Multi-output level-set estimation

For multiple outputs,

```text
f(x) = (f_1(x), ..., f_m(x)),
```

we may define multiple thresholds:

```text
f_j(x) >= tau_j,  j = 1, ..., m.
```

A multi-output level-set criterion must decide how to reduce output-wise boundary scores.

Possible reductions:

- weighted mean across outputs,
- maximum boundary uncertainty,
- minimum confidence over outputs,
- product of feasibility probabilities,
- custom objective-based reduction.

The reduction should be explicit because it changes the search behavior substantially.

---

## 7. Relationship to constraints

Level-set estimation and constraints are closely related but not identical.

A constraint in BO asks whether a candidate is feasible:

```text
g(x) <= 0.
```

A level-set problem may ask to learn the entire feasible region boundary. In a constrained BO problem, the same model may be used for both feasibility estimation and acquisition weighting, but the acquisition objective is still optimization.

In `bochan`, it is useful to keep these roles separate:

- constraint models estimate feasibility,
- level-set acquisitions estimate boundaries,
- optimization acquisitions search for good feasible candidates.

---

## 8. q-batch level-set estimation

For q-batch level-set estimation, diversity is important. If all candidates lie on the same boundary location, the batch may waste evaluations.

Useful mechanisms include:

- same-batch distance penalties,
- pending-point penalties,
- observed-point penalties,
- log-determinant diversity terms,
- sequential greedy candidate generation,
- boundary-region balancing.

The distance space should match the candidate representation used by the acquisition. Mixing raw and transformed spaces can produce misleading penalties.

---

## 9. Design guidance

A level-set acquisition should document:

1. the target threshold or cutpoint,
2. whether it uses latent, predictive, probability, or utility values,
3. whether the criterion is global or boundary-specific,
4. how output dimensions are reduced,
5. how q-batch diversity is handled,
6. whether `X_pending` is supported,
7. whether input perturbation is supported,
8. whether the score is intended to be maximized.

This information is essential for comparing regression, classification, and ordinal level-set acquisition functions under one API.
