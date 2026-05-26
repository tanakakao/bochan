# 03. Acquisition Functions

Acquisition functions convert a probabilistic model posterior into a decision value. They are the main bridge between surrogate modeling and candidate selection.

This document classifies the acquisition functions relevant to `bochan`.

---

## 1. General definition

Given a model posterior and a candidate batch `X`, an acquisition function returns a scalar score:

```text
alpha(X) -> acquisition value
```

For q-batch acquisition functions, `X` has shape:

```text
batch_shape x q x d
```

and the acquisition value usually has shape:

```text
batch_shape
```

The optimizer maximizes this value.

---

## 2. Improvement-based acquisition functions

Improvement-based acquisition functions measure how much better a candidate may be than the current best value.

### Expected Improvement

For maximization, improvement is

```text
I(x) = max(f(x) - f_best, 0).
```

Expected improvement is

```text
EI(x) = E[I(x)].
```

EI is intuitive and effective for noiseless or low-noise single-objective regression.

### Noisy Expected Improvement

When observations are noisy, the best observed value is uncertain. NEI integrates over uncertainty in baseline latent values rather than treating the best observation as fixed.

Practical rule:

- use EI when objective observations are reliable,
- use NEI when experimental noise is non-negligible.

### LogEI / LogNEI

Log variants are numerically more stable when improvement values are extremely small. In modern BoTorch usage, log-improvement variants are often preferred when available.

---

## 3. Confidence-bound acquisition functions

Upper Confidence Bound uses:

```text
UCB(x) = mu(x) + beta^{1/2} sigma(x)
```

for maximization.

The parameter `beta` controls the exploitation-exploration balance:

- small `beta`: more exploitative,
- large `beta`: more exploratory.

UCB is easy to interpret and useful as a baseline. It is also a useful conceptual model for many custom uncertainty-aware criteria.

---

## 4. Information-theoretic acquisition functions

Information-theoretic acquisition functions choose points that are expected to provide information about the optimum, optimal value, model parameters, or decision boundary.

Examples:

| Acquisition | Information target |
|---|---|
| KG | value of information for future decisions |
| PES | information about the optimizer |
| MES | information about the maximum value |
| JES | information about both optimizer and optimal value |
| BALD | mutual information between prediction and model / latent uncertainty |

These methods are often more expensive but can be attractive when each observation is costly and look-ahead value matters.

---

## 5. Active-learning acquisition functions

Active learning does not necessarily seek large objective values. It seeks useful observations.

Common criteria include:

- predictive variance,
- predictive entropy,
- BALD,
- negative integrated posterior variance,
- uncertainty sampling,
- margin uncertainty.

For classification, uncertainty is often highest near a probability of 0.5 in binary problems. For ordinal models, uncertainty can concentrate near cutpoints or ambiguous adjacent classes.

---

## 6. Level-set acquisition functions

Level-set estimation focuses on a threshold `tau` rather than the maximum.

A common regression criterion is Straddle:

```text
alpha_straddle(x) = beta * sigma(x) - |mu(x) - tau|.
```

This favors points that are both uncertain and close to the target boundary.

For classification and ordinal models, boundary search can be expressed in related ways:

- binary classification: search near probability 0.5 or latent decision boundary,
- ordinal regression: search near a chosen cutpoint,
- multi-output level-set: combine boundary criteria across outputs.

---

## 7. Multi-objective acquisition functions

For vector-valued objectives, acquisition functions should reason about Pareto improvement rather than scalar improvement.

Common approaches:

- EHVI: expected hypervolume improvement,
- NEHVI: noisy expected hypervolume improvement,
- NParEGO: random scalarization followed by scalar acquisition,
- constrained hypervolume improvement.

Key implementation concepts:

- `ref_point` defines the reference point for hypervolume,
- `X_baseline` defines existing evaluated points for noisy or baseline-aware acquisition functions,
- partitioning defines dominated and non-dominated regions,
- objectives must provide comparable output dimensions.

---

## 8. Objective-aware acquisition functions

Many BoTorch acquisition functions operate on posterior samples. An objective maps posterior samples to values used by the acquisition function.

In `bochan`, objective-aware acquisition design is important because outputs may represent:

- regression values,
- binary probabilities,
- multiclass scores,
- ordinal expected utility,
- risk-aggregated perturbed values,
- multi-output vectors.

A useful rule is:

```text
model posterior describes uncertainty;
objective describes what the user values;
acquisition function describes how to select observations.
```

Keeping these roles separate makes it easier to reuse the same acquisition logic across model families.

---

## 9. Pending and duplicate candidates

In q-batch and asynchronous optimization, some candidates may already be selected but not yet observed. These are usually represented as `X_pending`.

Ignoring pending points can produce duplicate or near-duplicate candidates. Practical strategies include:

- conditioning on pending points where the acquisition supports it,
- adding a soft distance penalty,
- using hard duplicate penalties,
- using sequential q-batch optimization,
- post-processing candidates after optimization.

If distances are used, the distance space must be consistent. Raw inputs and transformed inputs should not be mixed silently.

---

## 10. Design guidance for this repository

When adding a new acquisition function, document:

1. whether it is for optimization, active learning, or level-set estimation,
2. whether it expects latent values, probabilities, utilities, or posterior samples,
3. whether it supports q-batch input,
4. whether it supports `X_pending`,
5. whether it supports an `objective`,
6. whether it is single-output, multi-output, or multi-objective,
7. how it reduces output dimensions,
8. how it handles input perturbation or risk aggregation,
9. what tensor shapes it returns.

This information is often more important for users than the formula alone.
