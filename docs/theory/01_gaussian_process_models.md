# 01. Gaussian Process Models

Gaussian processes are the primary probabilistic surrogate models used in Bayesian optimization. This document summarizes the GP concepts that are assumed throughout `bochan` and explains how they map to BoTorch-like model wrappers.

---

## 1. Regression GP

A Gaussian process prior is written as

```text
f(x) ~ GP(m(x), k(x, x'))
```

where `m(x)` is a mean function and `k(x, x')` is a covariance kernel.

For standard Gaussian regression, observations are modeled as

```text
y = f(x) + eps,
eps ~ Normal(0, sigma^2).
```

Given observed data `(X, y)`, the model returns a posterior distribution over function values at new inputs `X_*`:

```text
p(f_* | X, y, X_*).
```

In implementation, this posterior is the object from which acquisition functions obtain predictive means, variances, and samples.

---

## 2. Predictive posterior

The most common quantities used by acquisition functions are:

```text
mu(x)     = E[f(x) | D]
sigma^2(x) = Var[f(x) | D]
```

where `D` is the observed dataset.

Exploitation-oriented criteria use `mu(x)`. Exploration-oriented criteria use `sigma(x)` or entropy-like quantities. Most practical acquisition functions combine both.

Examples:

- UCB uses mean plus uncertainty.
- EI uses the distribution of improvement over the current best value.
- Straddle uses uncertainty and distance from a level-set threshold.
- Entropy and BALD use predictive uncertainty or information gain.

---

## 3. Latent function and observation model

For Gaussian regression, the latent function and the observed response are close:

```text
latent f(x) -> observed y through Gaussian noise.
```

For classification and ordinal regression, this is no longer true. The GP usually models a latent function, and the likelihood converts that latent value into a probability distribution over labels.

Binary classification example:

```text
f(x) ~ GP(...)
p(y = 1 | x) = sigmoid(f(x)).
```

Ordinal regression example:

```text
f(x) ~ GP(...)
p(y = k | x) = P(c_{k-1} < f(x) <= c_k).
```

This distinction is central in `bochan` because some acquisition functions should operate on latent uncertainty, while others should operate on predictive probabilities or expected utility.

---

## 4. Model families in this repository

The same probabilistic pattern appears in several model families:

| Family | Typical output | Important distinction |
|---|---|---|
| Gaussian regression | continuous posterior | latent and response are close |
| Non-Gaussian regression | count / positive / bounded response | likelihood transforms latent values |
| Binary classification | class probability or latent score | latent value differs from probability |
| Multiclass classification | class probability vector | class dimension must be explicit |
| Ordinal regression | ordered class probabilities | cutpoints define class boundaries |
| Heteroscedastic models | mean and input-dependent noise | predictive variance has multiple sources |
| DeepGP / DeepKernel GP | transformed latent representation | feature extractor affects posterior geometry |
| SAAS / sparse models | relevance-weighted inputs | high-dimensional irrelevant features are suppressed |

The implementation should make these differences explicit rather than forcing all models to behave as ordinary regression.

---

## 5. BoTorch-style interface

A BoTorch-compatible model is expected to provide a method like:

```python
posterior = model.posterior(X)
```

where `X` is usually shaped as:

```text
batch_shape x q x d
```

The returned posterior should expose at least:

```text
posterior.mean
posterior.variance
posterior.rsample(...)
```

when relevant.

`bochan` wrappers should follow this convention as much as possible. For models with latent and predictive quantities, it is often useful to expose both:

```python
model.latent_posterior(X)
model.posterior(X)
```

A recommended convention is:

- `latent_posterior(X)` returns the GP latent function posterior,
- `posterior(X)` returns the final predictive posterior used by ordinary users,
- acquisition functions explicitly choose which one they need.

---

## 6. Input transforms

BoTorch models often support `input_transform`. In this repository, input transforms should be treated as part of model construction and prediction behavior, not as an external preprocessing detail.

Important rules:

1. The model should apply the same transform consistently during training and prediction.
2. Categorical dimensions should not be normalized as continuous variables.
3. Mixed models should clearly separate continuous and categorical dimensions.
4. If dimensionality reduction is used, raw and transformed training inputs should be traceable.

A useful implementation convention is:

```text
train_inputs_raw     original input space
train_inputs         model's actual training input space
```

For PCA or REMBO wrappers, `train_inputs_raw` should represent the original high-dimensional input, while internal GP training may happen in a latent lower-dimensional space.

---

## 7. Noise and uncertainty

Predictive uncertainty can include different components:

- epistemic uncertainty from limited data,
- observation noise,
- input perturbation uncertainty,
- model approximation uncertainty,
- heteroscedastic noise.

Acquisition functions must be explicit about which uncertainty they use.

For example:

- standard level-set search often uses latent function uncertainty,
- noisy BO often uses noise-aware improvement criteria such as NEI,
- heteroscedastic active learning may penalize pure observation noise if the goal is model improvement,
- input perturbation may aggregate the distribution over perturbed inputs rather than only the posterior at the nominal point.

---

## 8. Practical implementation contract

For a model wrapper to work well with the rest of the library, it should document:

1. whether `posterior(X)` returns latent values, probabilities, utilities, or observations,
2. the shape of `posterior.mean`,
3. whether output dimension is the final dimension,
4. whether `rsample` is supported,
5. how `input_transform` is applied,
6. how categorical dimensions are represented,
7. how training inputs are stored,
8. whether the model supports `condition_on_observations`.

Most shape and compatibility problems in acquisition functions come from one of these assumptions being implicit. The purpose of this library is to make those assumptions explicit.
