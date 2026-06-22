# 10. Regression Models and Likelihoods

This chapter gives the detailed mathematical contract for regression models in
`bochan`.  The important distinction is not only which covariance kernel is
used, but also which random variable is modeled, which likelihood connects the
latent function to observations, and which space is returned by `posterior()`.

---

## 1. Common latent-function formulation

Let

\[
\mathcal D = \{(x_i, y_i)\}_{i=1}^n,
\qquad x_i \in \mathbb R^d.
\]

A latent function is assigned a Gaussian-process prior

\[
f \sim \mathcal{GP}(m, k),
\]

so that, at the training inputs,

\[
\mathbf f = [f(x_1),\ldots,f(x_n)]^\top
\sim \mathcal N(\mathbf m, K_{XX}).
\]

The likelihood determines how the observed response is generated:

\[
p(\mathbf y\mid \mathbf f,\theta_{\mathrm{lik}})
= \prod_{i=1}^n p(y_i\mid f_i,\theta_{\mathrm{lik}}).
\]

For Gaussian likelihoods this model is conjugate.  For Beta, Gamma, Poisson,
Negative-Binomial, classification, and ordinal likelihoods it is not, and the
posterior must be approximated.

---

## 2. Gaussian regression

### 2.1 Homoscedastic observation model

The standard model is

\[
y_i = f(x_i) + \varepsilon_i,
\qquad
\varepsilon_i \sim \mathcal N(0,\sigma_n^2).
\]

Equivalently,

\[
\mathbf y\mid \mathbf f
\sim \mathcal N(\mathbf f,\sigma_n^2 I).
\]

For test inputs `X_*`, define

\[
K = K_{XX},\qquad
K_* = K_{X X_*},\qquad
K_{**}=K_{X_*X_*}.
\]

The latent posterior is

\[
\mu_*
= m_* + K_*^\top(K+\sigma_n^2I)^{-1}(\mathbf y-\mathbf m),
\]

\[
\Sigma_*
= K_{**}-K_*^\top(K+\sigma_n^2I)^{-1}K_*.
\]

The predictive distribution for a future noisy observation adds observation
noise:

\[
p(\mathbf y_*\mid\mathcal D)
= \mathcal N(\mu_*,\Sigma_*+\sigma_n^2I).
\]

This distinction is represented in BoTorch by the `observation_noise` argument.
An acquisition using latent function uncertainty should normally use
`observation_noise=False`.  A predictive interval for an actually observed
measurement may need `observation_noise=True`.

### 2.2 Exact marginal likelihood

For exact Gaussian regression, hyperparameters are commonly estimated by
maximizing

\[
\log p(\mathbf y\mid X,\theta)
= -\frac12(\mathbf y-\mathbf m)^\top K_y^{-1}(\mathbf y-\mathbf m)
  -\frac12\log|K_y|
  -\frac n2\log(2\pi),
\]

where

\[
K_y = K_{XX}+\sigma_n^2 I.
\]

The three terms respectively express data fit, model complexity, and a
normalizing constant.  A larger marginal likelihood is not equivalent to a
smaller training RMSE; it rewards calibrated covariance structure as well as
mean fit.

### 2.3 Implementation mapping

The high-level model registry maps ordinary Gaussian regression to BoTorch
models:

| API selection | Implementation |
|---|---|
| `task_type="regression", model_type="base"` | `botorch.models.SingleTaskGP` |
| mixed input + `model_type="base"` | `botorch.models.MixedSingleTaskGP` |
| mixed input + `model_type="kronecker"` | `bochan.models.regression.gaussian.MixedKroneckerMultiTaskGP` |

The standard contract is:

```python
posterior = model.posterior(X, observation_noise=False)
mean = posterior.mean
variance = posterior.variance
samples = posterior.rsample(sample_shape)
```

For standard regression, `posterior.mean` and `posterior.variance` are in the
continuous response space after any `outcome_transform` is undone by the model.

---

## 3. Known observation variances

When replicate experiments or measurement models provide a known variance
\(s_i^2\), use

\[
y_i\mid f_i \sim \mathcal N(f_i,s_i^2).
\]

Then

\[
K_y = K_{XX}+\operatorname{diag}(s_1^2,\ldots,s_n^2).
\]

This is different from learning an unknown input-dependent noise function.
Known `train_Yvar` values are data; a heteroscedastic noise GP is a second
statistical model.

Implementation rules:

- `train_Yvar` should have shape `N x 1` for ordinary single-output regression;
- values are variances, not standard deviations;
- variances must be expressed in the same transformed outcome space used during
  fitting;
- adding `train_Yvar` to classification probability variance is only an
  engineering convention and is not the Bernoulli likelihood itself.

---

## 4. Sparse variational GP inference

Classification, ordinal, DeepGP, and several non-Gaussian models in `bochan`
use inducing points.  Let

\[
\mathbf u = f(Z),\qquad Z\in\mathbb R^{M\times d},\qquad M\ll n.
\]

A variational distribution

\[
q(\mathbf u)=\mathcal N(\mathbf m_u,S_u)
\]

induces

\[
q(\mathbf f)=\int p(\mathbf f\mid\mathbf u)q(\mathbf u)d\mathbf u.
\]

Training maximizes the evidence lower bound

\[
\mathcal L_{\mathrm{ELBO}}
=
\mathbb E_{q(\mathbf f)}[\log p(\mathbf y\mid\mathbf f)]
-
\operatorname{KL}[q(\mathbf u)\|p(\mathbf u)].
\]

The expectation is evaluated analytically when possible and otherwise by
quadrature or Monte Carlo.  The KL term prevents the variational posterior from
moving arbitrarily far from the GP prior.

`BinaryClassificationGPModel`, `MulticlassClassificationGPModel`, and
`OrdinalGPModel` are wrappers around variational latent GPs.  Their fit helpers
therefore optimize a `VariationalELBO`, not an exact marginal likelihood.

---

## 5. Non-Gaussian regression

### 5.1 Generalized GP model

For non-Gaussian responses, define a response parameter through a link function

\[
\eta(x)=f(x),\qquad \vartheta(x)=g^{-1}(\eta(x)).
\]

The observation distribution is then

\[
y\mid x \sim p(y\mid\vartheta(x)).
\]

The latent GP posterior, response mean, and predictive observation distribution
are different objects.  An acquisition must state which one it consumes.

### 5.2 Beta regression

For a bounded response \(y\in(0,1)\), a common parameterization is

\[
y\sim\operatorname{Beta}(\alpha,\beta),
\qquad
\mu=\frac{\alpha}{\alpha+\beta},
\qquad
\phi=\alpha+\beta.
\]

Using

\[
\mu(x)=\sigma(f_\mu(x)),
\qquad
\phi(x)=\operatorname{softplus}(f_\phi(x))+\epsilon,
\]

we obtain

\[
\alpha(x)=\mu(x)\phi(x),
\qquad
\beta(x)=(1-\mu(x))\phi(x).
\]

The conditional variance is

\[
\operatorname{Var}(Y\mid x)
=
\frac{\mu(x)(1-\mu(x))}{\phi(x)+1}.
\]

A model with only a GP for \(\mu\) and a constant \(\phi\) is not fully
heteroscedastic.  A model with a second latent process for \(\phi(x)\) can
represent input-dependent dispersion.

Implementation family:

```text
src/bochan/models/regression/non_gaussian/beta/
```

### 5.3 Gamma regression

For positive continuous responses,

\[
y\sim\operatorname{Gamma}(a,b),
\]

where `b` may denote rate.  Then

\[
\mathbb E[Y]=\frac{a}{b},
\qquad
\operatorname{Var}(Y)=\frac{a}{b^2}.
\]

Positive parameters should be generated by a positive link such as
`softplus`.  Documents and implementations must always state whether the second
parameter is a rate or a scale because the variance formula changes.

Implementation family:

```text
src/bochan/models/regression/non_gaussian/gamma/
```

### 5.4 Poisson regression

For count data,

\[
y\sim\operatorname{Poisson}(\lambda(x)),
\qquad
\lambda(x)=\exp(f(x))
\]

or a numerically safer positive link.

\[
\mathbb E[Y\mid x]=\lambda(x),
\qquad
\operatorname{Var}(Y\mid x)=\lambda(x).
\]

The equality of mean and variance is restrictive.  When observed variance is
substantially larger than the mean, Negative-Binomial regression is usually
more appropriate.

Implementation family:

```text
src/bochan/models/regression/non_gaussian/poisson/
```

### 5.5 Negative-Binomial regression

One common mean-dispersion parameterization is

\[
\mathbb E[Y\mid x]=\mu(x),
\qquad
\operatorname{Var}(Y\mid x)=\mu(x)+\frac{\mu(x)^2}{r(x)}.
\]

The dispersion \(r\) controls overdispersion.  As \(r\to\infty\), the variance
approaches the Poisson variance.

Implementation family:

```text
src/bochan/models/regression/non_gaussian/negative_binomial/
```

---

## 6. Kernels and ARD

For a Matérn-5/2 kernel with automatic relevance determination,

\[
r(x,x')=
\sqrt{\sum_{j=1}^d\frac{(x_j-x'_j)^2}{\ell_j^2}},
\]

\[
k(x,x')
=
\sigma_f^2
\left(1+\sqrt5 r+\frac53r^2\right)
\exp(-\sqrt5 r).
\]

A small length scale \(\ell_j\) means the posterior may change rapidly along
input dimension `j`.  It is not automatically a causal feature importance.
Length scales are affected by input scaling, collinearity, priors, and finite
sample uncertainty.

For mixed continuous and categorical inputs, `bochan` follows the general
pattern

\[
k(x,x')
=
k_c(x_c,x'_c)+k_g(x_g,x'_g)
+k_c'(x_c,x'_c)k_g'(x_g,x'_g),
\]

where `c` and `g` denote continuous and categorical components.  Input
transforms must not continuously normalize categorical codes.

---

## 7. Multi-output Gaussian regression

### 7.1 Independent outputs

For `m` outputs, fitting independent models gives

\[
p(\mathbf f_1,\ldots,\mathbf f_m\mid\mathcal D)
=
\prod_{j=1}^m p(\mathbf f_j\mid\mathcal D_j).
\]

This is flexible and robust when outputs have different noise or missingness,
but it cannot transfer information between outputs.

### 7.2 Correlated tasks

A separable multitask covariance has the form

\[
\operatorname{Cov}[f_a(x),f_b(x')]
=
k_X(x,x')B_{ab},
\]

where `B` is the task covariance matrix.  In Kronecker models, the full
covariance can be written approximately as

\[
K_{\mathrm{full}}=B\otimes K_X.
\]

This representation is efficient when all tasks are observed on a shared input
grid.  The learned task covariance is a model parameter conditioned on the
kernel and noise model; it is not identical to the empirical Pearson
correlation of observed targets.

---

## 8. Posterior-space checklist

Before attaching an acquisition function, document the following.

| Question | Why it matters |
|---|---|
| Does `posterior()` return latent `f`, response mean, or observation distribution? | Determines the threshold and objective scale. |
| Is observation noise included? | Separates reducible epistemic uncertainty from irreducible measurement noise. |
| Is `rsample()` differentiable with respect to `X`? | Required by MC acquisitions optimized with gradients. |
| Is the output shape `... x q x m`? | Required by BoTorch objective and reduction conventions. |
| Is an `outcome_transform` automatically undone? | Determines the scale of `best_f`, thresholds, and constraints. |
| Is fitting exact MLL or variational ELBO? | Changes the interpretation and numerical behavior of training. |

---

## 9. Source map

| Theory component | Main implementation locations |
|---|---|
| Exact Gaussian regression | BoTorch `SingleTaskGP`, `MixedSingleTaskGP` through `src/bochan/api/model_registry.py` |
| Gaussian robust models | `src/bochan/models/regression/gaussian/robust/` |
| Gaussian DeepGP / Deep Kernel | `src/bochan/models/regression/gaussian/deep/` |
| Gaussian high-dimensional models | `src/bochan/models/regression/gaussian/high_dim/` |
| Beta regression | `src/bochan/models/regression/non_gaussian/beta/` |
| Gamma regression | `src/bochan/models/regression/non_gaussian/gamma/` |
| Poisson regression | `src/bochan/models/regression/non_gaussian/poisson/` |
| Negative-Binomial regression | `src/bochan/models/regression/non_gaussian/negative_binomial/` |
| Heteroscedastic shared helpers | `src/bochan/models/components/heteroscedastic.py` |

---

## 10. References

- Rasmussen and Williams, *Gaussian Processes for Machine Learning*, 2006.
- Titsias, *Variational Learning of Inducing Variables in Sparse Gaussian Processes*, 2009.
- Wilson et al., *Deep Kernel Learning*, 2016.
- Damianou and Lawrence, *Deep Gaussian Processes*, 2013.
