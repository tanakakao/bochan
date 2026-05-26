# 09. Tensor Shape Conventions

Tensor shape consistency is one of the most important implementation contracts in `bochan`. Many acquisition-function errors are not caused by incorrect theory but by ambiguous tensor shapes.

This document defines the recommended conventions.

---

## 1. Basic symbols

| Symbol | Meaning |
|---|---|
| `n` | number of training observations |
| `d` | input dimension |
| `q` | number of candidates in a q-batch |
| `m` | number of model outputs or objective outputs |
| `K` | number of classes |
| `n_w` | number of input perturbation samples per candidate |
| `batch_shape` | leading batch dimensions used by BoTorch |

---

## 2. Training data

Recommended shapes:

```text
train_X: n x d
train_Y: n x 1          single-output regression
train_Y: n x m          multi-output regression or objective values
train_Y: n              labels may be accepted internally but should be normalized by wrappers
```

Wrappers should document whether labels are stored as:

```text
n
n x 1
n x m
```

Classification and ordinal wrappers should be explicit about whether `train_Y` contains labels, one-hot values, utilities, or transformed targets.

---

## 3. Candidate input

BoTorch acquisition functions generally expect:

```text
X: batch_shape x q x d
```

For a single q-batch without extra batch dimensions:

```text
X: q x d
```

Some utilities may temporarily add a batch dimension:

```text
1 x q x d
```

Acquisition functions should not silently assume only two dimensions unless they are documented as non-batched utilities.

---

## 4. Posterior mean and variance

A standard single-output model often returns:

```text
posterior.mean: batch_shape x q x 1
posterior.variance: batch_shape x q x 1
```

A multi-output model often returns:

```text
posterior.mean: batch_shape x q x m
posterior.variance: batch_shape x q x m
```

Some GPyTorch or fully Bayesian models may introduce additional leading batch dimensions. Acquisition functions should either support those dimensions or reduce them explicitly with documented behavior.

---

## 5. Posterior samples

Monte Carlo acquisition functions often draw samples with shape:

```text
sample_shape x batch_shape x q x m
```

The objective typically maps samples to:

```text
sample_shape x batch_shape x q
```

or for multi-objective acquisition functions:

```text
sample_shape x batch_shape x q x m_obj
```

where `m_obj` is the number of objective dimensions after objective conversion.

---

## 6. Acquisition output

For q-batch acquisition functions:

```text
acq_value: batch_shape
```

For a single candidate batch with no extra batch dimensions, this may be a scalar tensor.

The acquisition value should normally not retain the `q` dimension unless it is a pointwise diagnostic rather than an acquisition function intended for `optimize_acqf`.

---

## 7. Objective output

A scalar objective should reduce model outputs to:

```text
sample_shape x batch_shape x q
```

A multi-objective objective should return:

```text
sample_shape x batch_shape x q x m_obj
```

An objective should not accidentally collapse the q dimension unless the acquisition function explicitly expects that.

---

## 8. Input perturbation shapes

Without perturbation:

```text
X: batch_shape x q x d
```

With perturbation:

```text
X_tilde: batch_shape x (q * n_w) x d
```

The model posterior is evaluated on the expanded input. The objective then aggregates:

```text
batch_shape x (q * n_w) -> batch_shape x q
```

For posterior samples:

```text
sample_shape x batch_shape x (q * n_w) -> sample_shape x batch_shape x q
```

If the objective cannot infer `q` and `n_w`, these should be passed explicitly.

---

## 9. Classification shapes

Binary classification may use one output dimension:

```text
probability: batch_shape x q x 1
```

or a squeezed form:

```text
probability: batch_shape x q
```

The preferred convention for wrappers is to keep an explicit output dimension when possible:

```text
batch_shape x q x 1
```

Acquisition functions can squeeze only when they intentionally convert to a scalar score.

Multiclass classification should make the class dimension explicit:

```text
class_probs: batch_shape x q x K
```

---

## 10. Ordinal shapes

Ordinal models may expose:

```text
latent mean: batch_shape x q x 1
class_probs: batch_shape x q x K
expected_utility: batch_shape x q x 1 or batch_shape x q
```

The model wrapper should document what `posterior(X)` returns.

A useful convention is:

```text
latent_posterior(X): latent function posterior
posterior(X): predictive class probability or expected-utility-compatible posterior
```

When an objective computes expected utility, the objective output should normally be:

```text
sample_shape x batch_shape x q
```

or

```text
batch_shape x q
```

for deterministic posterior-moment objectives.

---

## 11. ModelList and hybrid outputs

A model list may combine different output types. The objective layer should convert each model output into a compatible value.

Example final objective tensor:

```text
sample_shape x batch_shape x q x m_obj
```

where `m_obj` may contain:

1. regression value,
2. binary feasibility probability,
3. ordinal expected utility.

Raw model outputs should not be concatenated blindly if they have different meanings.

---

## 12. Common shape errors

Common errors include:

1. collapsing `q` and output dimensions accidentally,
2. treating `q * n_w` as `q`,
3. mixing latent and probability dimensions,
4. forgetting the class dimension in multiclass or ordinal models,
5. using `ModelList` outputs without objective conversion,
6. ignoring fully Bayesian leading batch dimensions,
7. computing pending penalties in raw space while acquisition values use transformed space,
8. returning `batch_shape x q` from an acquisition function that should return `batch_shape`.

When debugging, print shapes at these points:

```text
X
posterior.mean
samples
objective(samples)
acq_value
```

---

## 13. Minimal contract for new components

Every new model, objective, or acquisition function should document:

```text
Input X shape:
Posterior mean shape:
Sample shape:
Objective output shape:
Acquisition output shape:
Input perturbation behavior:
Multi-output behavior:
Classification / ordinal behavior:
```

This small amount of documentation prevents many downstream errors and makes the library easier to extend.
