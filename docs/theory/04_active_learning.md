# 04. Active Learning

Active learning is a sequential design strategy for improving a model efficiently. It is related to Bayesian optimization but has a different objective.

---

## 1. Difference from Bayesian optimization

Bayesian optimization asks:

```text
Which point is likely to be best?
```

Active learning asks:

```text
Which point is most useful to observe?
```

The best observation for improving a model may not have a high objective value.

| Aspect | Bayesian optimization | Active learning |
|---|---|---|
| Goal | find high-value points | improve model knowledge |
| Typical score | improvement or utility | uncertainty or information |
| Good candidate | high predicted value and/or high improvement | uncertain, informative, or representative |
| Examples | EI, NEI, UCB, KG | entropy, BALD, variance, NIPV |

---

## 2. Regression active learning

For regression, a simple active-learning criterion is posterior variance:

```text
alpha(x) = Var[f(x) | D].
```

This chooses points where the surrogate model is uncertain.

However, pure variance sampling may oversample irrelevant regions. Practical variants may include:

- region weights,
- boundary weights,
- distance penalties,
- diversity penalties,
- integrated posterior variance over a target region,
- constraints or feasibility weighting.

---

## 3. Classification active learning

For binary classification, uncertainty is often high near the decision boundary:

```text
p(y = 1 | x) ~= 0.5.
```

Common criteria:

- predictive entropy,
- margin uncertainty,
- BALD,
- latent boundary uncertainty,
- expected model change.

Predictive entropy for binary classification is high when the predictive probability is close to 0.5.

A key distinction is whether the acquisition uses:

1. latent GP uncertainty, or
2. final class probability uncertainty.

Both can be useful, but they answer different questions.

---

## 4. Ordinal active learning

Ordinal regression has ordered classes:

```text
y in {0, 1, ..., K - 1}.
```

Uncertainty may occur because:

- several adjacent classes are plausible,
- the latent value is close to a cutpoint,
- the model is uncertain about the latent function,
- the utility of the predicted class is ambiguous.

Active-learning criteria for ordinal models should make clear whether they target:

- all class boundaries,
- a specific cutpoint,
- maximum predictive entropy,
- expected utility uncertainty,
- variance around a level-set threshold.

---

## 5. Multi-output active learning

For multi-output models, the acquisition function must reduce output-wise uncertainty.

Possible reductions:

- mean over outputs,
- weighted mean over outputs,
- maximum uncertainty over outputs,
- minimum uncertainty over outputs,
- product or sum of output-wise criteria,
- custom objective-based aggregation.

A consistent API should expose both:

```text
reduction over q-batch candidates
reduction over output dimensions
```

These are not the same operation and should not be conflated.

---

## 6. Diversity and q-batch active learning

For q-batch active learning, selecting the top `q` individual uncertainty points can produce duplicates or clustered points.

A practical q-batch active-learning acquisition may require:

- joint posterior sampling,
- log-determinant diversity terms,
- distance penalties between candidates,
- `X_pending` penalties,
- hard duplicate filtering,
- sequential greedy selection.

In this repository, penalties such as `same_batch_penalty_weight`, `pending_penalty_weight`, and `observed_penalty_weight` are practical tools for making q-batch candidates more useful.

---

## 7. Active learning under input perturbation

If the actual executed input is uncertain,

```text
x_tilde = x + w,
```

then active learning can target uncertainty around the perturbed neighborhood rather than only the nominal point.

This changes the acquisition meaning:

- nominal uncertainty: how uncertain is the model exactly at `x`?
- robust uncertainty: how uncertain is the model around possible executed inputs near `x`?
- risk-aware uncertainty: how uncertain are bad-tail outcomes under perturbation?

The objective layer is the right place to aggregate `q * n_w` perturbed evaluations back to `q` nominal candidates.

---

## 8. Design guidance

An active-learning acquisition should document:

1. which uncertainty it measures,
2. whether it uses latent or predictive outputs,
3. whether it is boundary-oriented or global,
4. how it handles output dimensions,
5. how it handles q-batch diversity,
6. how it handles pending and observed points,
7. whether input perturbation is supported,
8. whether its score is meant to be maximized.

This makes active-learning functions easier to compare with Bayesian optimization and level-set functions.
