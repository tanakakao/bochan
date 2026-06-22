# 11. Classification Models

Gaussian-process classification models discrete labels through one or more
latent Gaussian processes and a non-Gaussian likelihood.  This chapter focuses
on model construction, likelihoods, variational inference, probability
prediction, calibration, and the current `bochan` posterior contracts.

Decision criteria are treated elsewhere:

- Chapter 04: Active Learning, entropy, BALD, and margin sampling;
- Chapter 06: classification outputs as BO objectives or constraints;
- Chapter 16: classification Level-set Estimation formulas.

The central distinction is:

> latent-function uncertainty, uncertainty in a class-probability function, and
> randomness of a future class label are different mathematical objects.

---

## 1. Binary classification

Let

$$
y_i\in\{0,1\}.
$$

A scalar latent function follows

$$
f\sim\mathcal{GP}(m,k).
$$

The likelihood maps the latent value to class probability:

$$
P(Y=1\mid f)=\pi(f),
\qquad
P(Y=0\mid f)=1-\pi(f).
$$

Common inverse links are:

### Logistic link

$$
\pi(f)
=
\sigma(f)
=
\frac{1}{1+e^{-f}}.
$$

### Probit link

$$
\pi(f)=\Phi(f).
$$

GPyTorch's standard `BernoulliLikelihood` uses a probit-style construction.
Code should therefore not assume that every binary posterior was produced by a
logistic sigmoid.

---

## 2. Likelihood and latent scale

For observations

$$
\mathbf y=(y_1,\ldots,y_n),
$$

the likelihood is

$$
p(\mathbf y\mid\mathbf f)
=
\prod_{i=1}^{n}
\pi(f_i)^{y_i}
[1-\pi(f_i)]^{1-y_i}.
$$

The latent scale is not directly observed.  Its interpretation depends on the
link:

### Logistic odds

$$
\log
\frac{P(Y=1\mid f)}{P(Y=0\mid f)}
=f.
$$

### Probit latent-noise interpretation

Introduce

$$
z=f+\epsilon,
\qquad
\epsilon\sim\mathcal N(0,1),
$$

and set

$$
Y=\mathbf1[z>0].
$$

Then

$$
P(Y=1\mid f)=\Phi(f).
$$

Latent values from different links are not directly comparable even when their
probabilities are similar.

---

## 3. Non-conjugate posterior

Bayes' theorem gives

$$
p(\mathbf f\mid\mathbf y)
\propto
p(\mathbf y\mid\mathbf f)
\mathcal N(\mathbf f;\mathbf m,K).
$$

The Bernoulli likelihood is not Gaussian, so this posterior is not analytically
Gaussian and the evidence

$$
p(\mathbf y)
=
\int
p(\mathbf y\mid\mathbf f)p(\mathbf f)d\mathbf f
$$

is not available in closed form.

Approximation methods include:

- Laplace approximation;
- expectation propagation;
- variational inference;
- MCMC.

The current base classification models in `bochan` use sparse variational GP
inference.

---

## 4. Sparse variational binary GP

Choose inducing inputs

$$
Z=(z_1,\ldots,z_M)
$$

and inducing variables

$$
\mathbf u=f(Z).
$$

Let

$$
q(\mathbf u)
=
\mathcal N(\mathbf m_u,S_u).
$$

The induced variational latent distribution is

$$
q(\mathbf f)
=
\int
p(\mathbf f\mid\mathbf u)
q(\mathbf u)d\mathbf u.
$$

Training maximizes

$$
\mathcal L_{\mathrm{ELBO}}
=
\sum_{i=1}^{n}
\mathbb E_{q(f_i)}
[
\log p(y_i\mid f_i)
]
-
\operatorname{KL}
[q(\mathbf u)\|p(\mathbf u)].
$$

The expected log likelihood is evaluated through GPyTorch likelihood
quadrature.  Both GP and variational parameters are optimized.

Important practical parameters are:

- number and initial position of inducing points;
- whether inducing locations are learned;
- kernel and ARD dimensions;
- optimizer learning rate;
- number of epochs;
- minibatch size;
- class imbalance treatment.

---

## 5. Predictive class probability

At a test input, the variational latent posterior is approximately Gaussian:

$$
q(f_*)
=
\mathcal N(\mu_f,\sigma_f^2).
$$

The predictive class probability is

$$
p_*(x)
=
P(Y=1\mid x,\mathcal D)
=
\int
\pi(f_*)q(f_*)df_*.
$$

In general,

$$
p_*(x)
\ne
\pi(\mu_f).
$$

The plug-in value ignores latent uncertainty.

For a probit link and Gaussian latent posterior, the integral has the identity

$$
\int
\Phi(f)
\mathcal N(f;\mu,\sigma^2)df
=
\Phi\left(
\frac{\mu}{\sqrt{1+\sigma^2}}
\right).
$$

The probability moves toward `0.5` as latent uncertainty increases.

---

## 6. Three binary uncertainty quantities

### 6.1 Latent posterior variance

$$
V_f(x)
=
\operatorname{Var}[f(x)\mid\mathcal D].
$$

This measures uncertainty in the latent decision function.

### 6.2 Probability-function variance

If latent posterior samples generate

$$
p^{(s)}(x)=\pi(f^{(s)}(x)),
$$

then

$$
V_p(x)
=
\operatorname{Var}_s[p^{(s)}(x)].
$$

This measures posterior uncertainty in the class probability.

### 6.3 Bernoulli observation variance

For fixed predictive probability `p`,

$$
V_Y(x)=p(x)[1-p(x)].
$$

This is randomness of a future binary label.  It is maximal at `p=0.5` even if
`p` is known exactly.

These quantities support different decisions and must not be given the same
name simply as "classification variance."

---

## 7. Binary posterior contract in `bochan`

The main continuous-input model is

```text
BinaryClassificationGPModel
```

and the mixed-input model is

```text
BinaryClassificationMixedGPModel
```

in

```text
src/bochan/models/classification/binary/base/models.py
```

The public accessors are:

```python
probability_posterior = model.posterior(X)
latent_posterior = model.latent_posterior(X)
```

### `posterior(X)`

1. transforms the input;
2. evaluates the latent variational GP;
3. applies the Bernoulli likelihood;
4. constructs `SimpleBernoulliPosterior`.

Its mean is probability of class `1`:

$$
\operatorname{mean}=P(Y=1\mid x,\mathcal D).
$$

Its variance is based on the predictive Bernoulli distribution, with optional
engineering noise addition where configured.

### `latent_posterior(X)`

Bypasses the likelihood and returns a `GPyTorchPosterior` over `f(x)`.

This distinction is relied on by latent-boundary LSE and probability-space BO.

---

## 8. `SimpleBernoulliPosterior`

The custom binary posterior provides a BoTorch-style interface:

```text
mean:     batch_shape x q x 1
variance: batch_shape x q x 1
```

and sampling behavior expected by acquisitions.

A Bernoulli random variable is discrete, while many BoTorch MC acquisitions
expect differentiable reparameterized samples.  Any continuous probability or
normal-proxy sampling path should be interpreted according to the posterior
implementation rather than as exact discrete label sampling.

The relevant source is:

```text
src/bochan/posteriors/bernoulli.py
```

---

## 9. Binary decision boundary

For a symmetric monotone link,

$$
f(x)=0
$$

corresponds to

$$
P(Y=1\mid f)=0.5.
$$

For a probability threshold `tau_p`, the link-level threshold is

$$
tau_f=\pi^{-1}(tau_p).
$$

However, the marginal predictive probability

$$
P(Y=1\mid x,\mathcal D)
=
\int\pi(f)q(f)df
$$

also depends on latent variance.  Consequently, setting the posterior latent
mean equal to `pi^{-1}(tau_p)` is not always exactly equivalent to a marginal
probability contour at `tau_p`.

---

## 10. Class imbalance

If class proportions are highly unequal, maximum likelihood can produce a model
with good overall accuracy but poor minority-class behavior.

Possible approaches include:

- stratified initial data;
- weighted likelihood or resampling;
- calibrated decision thresholds;
- utility-sensitive objectives;
- targeted Active Learning;
- class-specific evaluation metrics.

Changing the class threshold does not retrain the probability model.  Weighted
training can change probability calibration and should be evaluated explicitly.

---

## 11. Calibration

A calibrated binary model satisfies approximately

$$
P(Y=1\mid p(X)=r)=r.
$$

Useful metrics include:

### Brier score

$$
\operatorname{BS}
=
\frac1n
\sum_i(p_i-y_i)^2.
$$

### Log loss

$$
-rac1n
\sum_i
[y_i\log p_i+(1-y_i)\log(1-p_i)].
$$

### Reliability diagram

Bin predictions and compare mean probability with observed class frequency.

### Expected calibration error

Weighted average of binwise probability-frequency gaps.

BO and adaptive sampling alter the input distribution, so calibration on random
held-out data may not transfer perfectly to acquisition-selected regions.

---

## 12. Multiclass classification

Let

$$
y\in\{0,1,\ldots,K-1\}.
$$

Introduce class-wise latent functions

$$
f_k(x),
\qquad k=0,\ldots,K-1.
$$

Let

$$
\mathbf f(x)
=[f_0(x),\ldots,f_{K-1}(x)].
$$

A categorical likelihood uses class probabilities

$$
p_k(x)
=P(Y=k\mid\mathbf f(x)).
$$

---

## 13. Softmax likelihood

The softmax model is

$$
p_k
=
\frac{\exp(f_k/T)}
{\sum_{j=0}^{K-1}\exp(f_j/T)},
$$

where `T>0` is temperature.

### Shift invariance

For any scalar `a`,

$$
\operatorname{softmax}(\mathbf f+a\mathbf1)
=
\operatorname{softmax}(\mathbf f).
$$

Only relative logits are identifiable.  Absolute class-wise latent means do not
have independent interpretation.

### Temperature

- `T<1`: sharper probabilities;
- `T>1`: flatter probabilities.

Post-hoc temperature scaling can improve calibration but should be fitted on
separate calibration data.

---

## 14. Class-wise variational GP

The current multiclass base model uses one class-batched latent SVGP.
Conceptually,

$$
f_k\sim\mathcal{GP}(m_k,k_k).
$$

The inducing-point and kernel batch shape is

```text
[num_classes]
```

and the variational objective is

$$
\mathcal L
=
\sum_i
\mathbb E_{q(\mathbf f_i)}
[
\log P(y_i\mid\mathbf f_i)
]
-
\sum_k
\operatorname{KL}[q(\mathbf u_k)\|p(\mathbf u_k)].
$$

With the current independent class-batch kernel construction, dependence among
class probabilities arises through the softmax normalization, while latent GP
processes are represented in a class-wise batched form.

---

## 15. Multiclass implementation shapes

For acquisition input

```text
batch_shape x q x d
```

the class-batched variational GP needs a class axis.  The wrapper inserts

```text
batch_shape x 1 x q x d
```

which broadcasts internally to class batch

```text
batch_shape x K x q x d
```

The probability posterior exposes

```text
batch_shape x q x K
```

This class batch is model structure and must not be averaged as if it were a
DeepGP or fully Bayesian sample axis.

---

## 16. Multiclass posterior contract in `bochan`

The main classes are

```text
MulticlassClassificationGPModel
MulticlassClassificationMixedGPModel
```

in

```text
src/bochan/models/classification/multiclass/base/models.py
```

Public accessors:

```python
latent = model.latent_posterior(X)
probability = model.posterior(X)
probs = model.class_probs(X)
predicted_class = model.predict_class(X)
```

### `latent_posterior(X)`

Returns a class-batched `GPyTorchPosterior` over latent logits.

### `posterior(X)`

Wraps the latent posterior in `MulticlassProbsPosterior`.

### `class_probs(X)`

Returns

```text
batch_shape x q x K
```

probability means.

The posterior object and shared helpers are in

```text
src/bochan/models/components/multiclass.py
```

---

## 17. Multiclass predictive probabilities

The predictive probability requires integration over latent logits:

$$
p_k(x)
=
\mathbb E_{q(\mathbf f(x))}
\left[
rac{e^{f_k/T}}
{\sum_je^{f_j/T}}
\right].
$$

This is not generally equal to

$$
rac{e^{\mathbb E[f_k]/T}}
{\sum_je^{\mathbb E[f_j]/T}}.
$$

`MulticlassProbsPosterior` uses the latent posterior to provide probability
moments or sampling behavior suitable for current acquisitions.  Its exact
approximation should be interpreted from the implementation.

---

## 18. Multiclass uncertainty

### Predictive entropy

$$
H(Y\mid x,\mathcal D)
=-\sum_kp_k\log p_k.
$$

### Top-two margin

Let

$$
p_{(1)}\ge p_{(2)}.
$$

Margin is

$$
p_{(1)}-p_{(2)}.
$$

### Probability covariance

Posterior samples of the probability vector yield

$$
\operatorname{Cov}[\mathbf p(x)].
$$

Because probabilities sum to one, their covariance is singular in the full
`K`-dimensional space.

### Categorical observation covariance

For one-hot label vector `e_Y`,

$$
\operatorname{Cov}(e_Y\mid\mathbf p)
=
\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^\top.
$$

This is future-label randomness, not posterior uncertainty in `p`.

---

## 19. Target class and class set

A target class probability is

$$
p_{k^*}(x).
$$

For acceptable class set `A`, union probability is

$$
P(Y\in A\mid x)
=
\sum_{k\in A}p_k(x).
$$

This sum is probabilistically meaningful because classes are mutually
exclusive.  A mean, maximum, or minimum over selected class probabilities is a
score with different meaning.

The target selection belongs in an objective, posterior transform, or
acquisition configuration rather than changing the underlying multiclass
likelihood.

---

## 20. Mixed-input classification

For continuous dimensions `C` and categorical dimensions `G`, current mixed
kernels follow a pattern such as

$$
k(x,x')
=
k_C+k_G+k_C'k_G'.
$$

The implementation:

- normalizes continuous columns only;
- uses `CategoricalKernel` for category dimensions;
- checks that `input_transform` does not alter categories;
- uses class-batched kernels for multiclass output;
- preserves categorical values when `InputPerturbation` expands q.

Category counts and valid fixed-feature combinations must be consistent with
candidate optimization.

---

## 21. Condition-on-observations behavior

Exact Gaussian conditioning is not available for variational classification in
the same closed form as exact regression.

The binary and multiclass wrappers can implement

```python
condition_on_observations(X, Y)
```

by reconstructing a new model on old plus new data and copying learned
parameters.  This is an approximate workflow and may not perform full
variational refitting unless explicitly invoked.

Consequences:

- look-ahead fantasies are approximate;
- fantasy sample dimensions may be unsupported;
- acquisition classes requiring exact conditioning should be validated
  separately.

---

## 22. Heteroscedastic classification interpretation

A binary or multiclass likelihood is already stochastic.  A second noise model
requires a precise generative meaning, such as:

### Label corruption

$$
P(Y_{\mathrm{obs}}=1)
=[1-\rho(x)]p(x)+\rho(x)[1-p(x)].
$$

### Input-dependent temperature

$$
p_k
=
\operatorname{softmax}
\left(
\frac{f_k}{T(x)}
\right).
$$

### External probability-estimation uncertainty

Targets may be estimated class proportions with measurement variance rather
than individual labels.

The current robust wrappers can add or combine an auxiliary noise prediction
with posterior variance or acquisition scores.  This should be called a
noise-aware engineering convention unless the likelihood itself includes the
noise process.  Chapter 13 gives the full distinction.

---

## 23. Deep and high-dimensional variants

Classification families include:

- DeepGP;
- Deep Kernel GP;
- Deep Kernel DeepGP;
- SAAS;
- decomposition or embedding variants where implemented;
- heteroscedastic and robust variants.

The likelihood and posterior-space distinctions in this chapter remain valid.
The deep or high-dimensional model changes latent representation and inference,
not the semantic meaning of the class label.

See Chapter 14.

---

## 24. Model evaluation

### Binary

- log loss;
- Brier score;
- ROC-AUC and precision-recall AUC;
- sensitivity/specificity at operational threshold;
- calibration;
- class-specific error;
- posterior probability error in simulations.

### Multiclass

- categorical log loss;
- macro and weighted F1;
- classwise recall;
- confusion matrix;
- multiclass Brier score;
- classwise and aggregate calibration.

### Sequential decision evaluation

Predictive metrics should be accompanied by:

- BO regret or target success;
- Active Learning loss versus labels;
- LSE boundary or set error;
- robustness under adaptive sampling.

---

## 25. `bochan` source map

| Component | Source |
|---|---|
| Binary base models | `src/bochan/models/classification/binary/base/models.py` |
| Binary custom posterior | `src/bochan/posteriors/bernoulli.py` |
| Binary fitting | `src/bochan/fit/` classification fitting helpers |
| Binary robust models | `src/bochan/models/classification/binary/robust/` |
| Binary deep models | `src/bochan/models/classification/binary/deep/` |
| Binary high-dimensional models | `src/bochan/models/classification/binary/high_dim/` |
| Multiclass base models | `src/bochan/models/classification/multiclass/base/models.py` |
| Multiclass posterior helpers | `src/bochan/models/components/multiclass.py` |
| Multiclass robust models | `src/bochan/models/classification/multiclass/robust/` |
| Multiclass deep models | `src/bochan/models/classification/multiclass/deep/` |
| Multiclass high-dimensional models | `src/bochan/models/classification/multiclass/high_dim/` |
| Classification posterior transforms | `src/bochan/models/transforms/posterior/classification.py` |
| High-level model resolution | `src/bochan/api/model_registry.py` |

---

## 26. Model checklist

1. Binary or multiclass?
2. Link and likelihood?
3. Latent GP kernel?
4. Inducing-point count and initialization?
5. Class imbalance treatment?
6. `posterior()` latent or probability?
7. Probability marginalization or plug-in approximation?
8. Variance meaning?
9. Calibration procedure?
10. Mixed-input category handling?
11. Conditioning/fantasy support?
12. Deep/high-dimensional representation?
13. Evaluation metrics?
14. Decision objective or constraint transformation?

---

## 27. References

- Rasmussen and Williams, *Gaussian Processes for Machine Learning*, 2006.
- Nickisch and Rasmussen, *Approximations for Binary Gaussian Process Classification*, 2008.
- Titsias, *Variational Learning of Inducing Variables in Sparse Gaussian Processes*, 2009.
- Guo et al., *On Calibration of Modern Neural Networks*, 2017, for temperature-scaling concepts.
