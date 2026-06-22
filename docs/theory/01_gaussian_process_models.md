# 01. Gaussian-process Foundations

This chapter introduces the probability and linear-algebra foundations shared
by the model families in `bochan`.  It defines Gaussian processes, posterior
conditioning, kernels, marginal likelihood, variational inference, and the
posterior interface expected by BoTorch-style acquisition functions.

Detailed response models are treated later:

- Chapter 10: Gaussian and non-Gaussian regression;
- Chapter 11: binary and multiclass classification;
- Chapter 12: ordinal models;
- Chapter 13: heteroscedastic and robust models;
- Chapter 14: deep and high-dimensional models.

---

## 1. Random variables, vectors, and functions

A scalar Gaussian random variable is written

$$
Z\sim\mathcal N(\mu,\sigma^2).
$$

A Gaussian random vector is

$$
\mathbf z\sim\mathcal N(\boldsymbol\mu,\Sigma),
$$

where `Sigma` is symmetric positive semidefinite.  For every vector `a`,

$$
\mathbf a^\top\Sigma\mathbf a\ge 0.
$$

A Gaussian process extends this idea from a finite vector to a random function.
A function `f` follows a GP if every finite collection of evaluations is jointly
Gaussian:

$$
f\sim\mathcal{GP}(m,k)
$$

means that, for arbitrary inputs

$$
X=(x_1,\ldots,x_n),
$$

$$
\mathbf f_X
=
[f(x_1),\ldots,f(x_n)]^\top
\sim
\mathcal N(\mathbf m_X,K_{XX}),
$$

with

$$
[\mathbf m_X]_i=m(x_i),
\qquad
[K_{XX}]_{ij}=k(x_i,x_j).
$$

A GP is therefore a distribution over functions, not a single fitted curve.
Posterior means, variances, and samples are summaries of that function
distribution after observing data.

---

## 2. Mean and covariance functions

### 2.1 Mean function

The mean function is

$$
m(x)=\mathbb E[f(x)].
$$

Common choices are:

- zero mean;
- constant mean;
- linear mean;
- a parametric trend estimated jointly with the kernel.

A zero mean prior does not mean the posterior prediction is zero.  It means
that deviations from zero are explained through observed data and the kernel.
When the input leaves the region supported by data, the posterior tends toward
the prior mean.

### 2.2 Covariance function

The covariance function is

$$
k(x,x')
=
\operatorname{Cov}[f(x),f(x')].
$$

It determines:

- smoothness;
- amplitude;
- characteristic length scales;
- periodicity;
- additive or interaction structure;
- task correlation;
- similarity between categories.

A valid kernel must produce a positive-semidefinite Gram matrix for every input
set.

---

## 3. Common kernels

### 3.1 Squared-exponential kernel

The isotropic radial-basis-function kernel is

$$
k(x,x')
=
\sigma_f^2
\exp\left(
-\frac{\|x-x'\|^2}{2\ell^2}
\right).
$$

It implies infinitely differentiable sample paths.  This can be smoother than
many physical response surfaces.

### 3.2 Matérn kernel

The Matérn family is

$$
k_\nu(r)
=
\sigma_f^2
\frac{2^{1-\nu}}{\Gamma(\nu)}
\left(\sqrt{2\nu}r\right)^\nu
K_\nu\left(\sqrt{2\nu}r\right),
$$

where

$$
r^2
=
\sum_{j=1}^d
\frac{(x_j-x_j')^2}{\ell_j^2}.
$$

For `nu=5/2`,

$$
k(r)
=
\sigma_f^2
\left(1+\sqrt5r+\frac53r^2\right)e^{-\sqrt5r}.
$$

Matérn-5/2 is a common default in Bayesian optimization because it allows less
smooth functions than the RBF kernel while remaining differentiable enough for
gradient-based acquisition optimization.

### 3.3 Automatic relevance determination

ARD assigns one length scale per input dimension:

$$
r^2
=
\sum_j
\frac{(x_j-x'_j)^2}{\ell_j^2}.
$$

A large `ell_j` makes the kernel insensitive to dimension `j`; a small `ell_j`
allows rapid variation along that dimension.

Length scales are not causal feature importances.  They depend on:

- input normalization;
- collinearity;
- priors;
- kernel choice;
- observation noise;
- finite data;
- likelihood approximation.

### 3.4 Additive and product kernels

If

$$
f(x)=f_1(x_{S_1})+f_2(x_{S_2}),
$$

an additive kernel can be used:

$$
k(x,x')=k_1(x_{S_1},x'_{S_1})+k_2(x_{S_2},x'_{S_2}).
$$

A product kernel represents interactions:

$$
k(x,x')=k_1(x,x')k_2(x,x').
$$

Mixed continuous/categorical models in `bochan` frequently use a combination of
continuous, categorical, and interaction kernels.

### 3.5 Kernel amplitude and noise

A scaled kernel is

$$
k_{\mathrm{scaled}}(x,x')
=
\sigma_f^2k_0(x,x').
$$

The signal variance `sigma_f^2` describes function variation.  It must not be
confused with observation-noise variance `sigma_n^2`.

---

## 4. Joint Gaussian conditioning

Let the training latent values and test latent values have joint distribution

$$
\begin{bmatrix}
\mathbf f\\
\mathbf f_*
\end{bmatrix}
\sim
\mathcal N\left(
\begin{bmatrix}
\mathbf m\\
\mathbf m_*
\end{bmatrix},
\begin{bmatrix}
K_{XX} & K_{X X_*}\\
K_{X_*X} & K_{X_*X_*}
\end{bmatrix}
\right).
$$

For Gaussian observations

$$
\mathbf y=\mathbf f+\boldsymbol\varepsilon,
\qquad
\boldsymbol\varepsilon\sim\mathcal N(0,\Sigma_n),
$$

we obtain

$$
\mathbf y
\sim
\mathcal N(\mathbf m,K_{XX}+\Sigma_n).
$$

Conditioning gives

$$
\boldsymbol\mu_*
=
\mathbf m_*
+
K_{X_*X}
(K_{XX}+\Sigma_n)^{-1}
(\mathbf y-\mathbf m),
$$

$$
\Sigma_*
=
K_{X_*X_*}
-
K_{X_*X}
(K_{XX}+\Sigma_n)^{-1}
K_{XX_*}.
$$

The posterior mean is a kernel-weighted interpolation or smoothing of observed
residuals.  The posterior covariance depends on input geometry and noise, not
on the observed target values for a fixed set of hyperparameters.

In numerical implementations, the inverse is not formed explicitly.  A
Cholesky factorization or linear solve is used:

$$
K_y=LL^\top.
$$

---

## 5. Latent and predictive distributions

The latent posterior is

$$
p(f_*\mid\mathcal D).
$$

The predictive observation distribution is

$$
p(y_*\mid\mathcal D)
=
\int p(y_*\mid f_*)p(f_*\mid\mathcal D)df_*.
$$

For Gaussian noise,

$$
\operatorname{Var}(y_*\mid\mathcal D)
=
\operatorname{Var}(f_*\mid\mathcal D)
+\sigma_n^2.
$$

This distinction matters for sequential decisions:

- learning the underlying function often uses latent variance;
- predicting a future measurement uses observation variance too;
- NEI accounts for uncertainty in latent baseline values;
- heteroscedastic reliability may require total predictive variance.

In BoTorch-style models, `observation_noise=False` generally requests a latent
posterior, while `observation_noise=True` asks the model to include its
likelihood noise when supported.

Classification and ordinal likelihoods are non-Gaussian, so the predictive
response distribution is not obtained by simply adding a variance term.

---

## 6. Marginal likelihood

For Gaussian regression,

$$
p(\mathbf y\mid X,\theta)
=
\mathcal N(\mathbf y;\mathbf m,K_y),
$$

where

$$
K_y=K_{XX}+\Sigma_n.
$$

The log marginal likelihood is

$$
\log p(\mathbf y\mid X,\theta)
=
-\frac12
(\mathbf y-\mathbf m)^\top K_y^{-1}(\mathbf y-\mathbf m)
-
\frac12\log|K_y|
-
\frac n2\log(2\pi).
$$

The terms are:

1. data fit;
2. complexity penalty through covariance volume;
3. normalization.

This is not equivalent to minimizing training error.  Very short length scales
may fit the data but incur a complexity penalty.  Very large noise may explain
residuals but reduce predictive sharpness.

Hyperparameters include:

- length scales;
- signal variance;
- noise variance;
- mean parameters;
- task covariance parameters;
- likelihood parameters.

The exact marginal likelihood is used only when the likelihood and model allow
analytic Gaussian integration.

---

## 7. Non-Gaussian likelihoods

For a general likelihood,

$$
p(\mathbf y\mid\mathbf f)
=
\prod_i p(y_i\mid f_i),
$$

Bayes' theorem gives

$$
p(\mathbf f\mid\mathbf y)
\propto
p(\mathbf y\mid\mathbf f)p(\mathbf f).
$$

For Bernoulli, categorical, ordered-logit, Poisson, and many other likelihoods,
this posterior is not Gaussian and the evidence integral is not closed form.

Approximation choices include:

- Laplace approximation;
- expectation propagation;
- variational inference;
- Markov-chain Monte Carlo;
- quadrature for one-dimensional likelihood expectations.

`bochan` classification and ordinal models primarily use sparse variational GP
inference.

---

## 8. Sparse variational Gaussian processes

Choose inducing inputs

$$
Z=(z_1,\ldots,z_M),
\qquad M\ll n,
$$

and inducing variables

$$
\mathbf u=f(Z).
$$

Introduce a variational distribution

$$
q(\mathbf u)=\mathcal N(\mathbf m_u,S_u).
$$

The GP conditional induces

$$
q(\mathbf f)
=
\int p(\mathbf f\mid\mathbf u)q(\mathbf u)d\mathbf u.
$$

The evidence lower bound is

$$
\mathcal L_{\mathrm{ELBO}}
=
\mathbb E_{q(\mathbf f)}
[\log p(\mathbf y\mid\mathbf f)]
-
\operatorname{KL}
[q(\mathbf u)\|p(\mathbf u)].
$$

The first term rewards predictive fit under the likelihood.  The KL term
regularizes the approximate posterior toward the GP prior.

Inducing-point choices affect both accuracy and computational cost.  With `M`
inducing points, common costs are approximately cubic in `M` rather than in
`n`, though exact complexity depends on batching and structure.

Important implementation parameters include:

- number of inducing points;
- initial inducing locations;
- whether inducing locations are learned;
- variational distribution type;
- minibatch size;
- optimization learning rate and epochs.

---

## 9. Fully Bayesian Gaussian processes

Maximum-likelihood fitting estimates one hyperparameter vector

$$
\hat\theta.
$$

A fully Bayesian model integrates hyperparameters:

$$
p(f_*\mid\mathcal D)
=
\int
p(f_*\mid\mathcal D,\theta)
\,p(\theta\mid\mathcal D)
\,d\theta.
$$

In practice, posterior samples

$$
\theta^{(s)}\sim p(\theta\mid\mathcal D)
$$

create an additional model-batch dimension.  Acquisition functions must either
average over this dimension or preserve it according to BoTorch ensemble
semantics.

SAAS models are an important example.  Their shrinkage prior is detailed in
Chapter 14.

---

## 10. Multi-output covariance

For vector-valued latent function

$$
\mathbf f(x)
=
[f_1(x),\ldots,f_m(x)]^\top,
$$

a separable covariance can be written

$$
\operatorname{Cov}[f_a(x),f_b(x')]
=
B_{ab}k_X(x,x').
$$

The matrix `B` is a task covariance.  On a common input grid, the full
covariance may have Kronecker form

$$
K_{\mathrm{full}}
=B\otimes K_X.
$$

An independent ModelList instead assumes

$$
\operatorname{Cov}[f_a(x),f_b(x')]=0
\quad\text{for }a\ne b.
$$

The learned task covariance is a property of the probabilistic model.  It is not
identical to the empirical correlation of observed targets because it is
conditioned on kernels, transforms, noise, and priors.

Chapter 15 treats heterogeneous outputs with different likelihoods.

---

## 11. Input and outcome transforms

### 11.1 Input normalization

For lower and upper bounds `l_j,u_j`, normalization is

$$
\tilde x_j
=
\frac{x_j-l_j}{u_j-l_j}.
$$

Length scales are then measured in normalized coordinates.  Without consistent
normalization, ARD values and optimizer step sizes are difficult to interpret.

### 11.2 Outcome standardization

For output mean `bar y` and standard deviation `s_y`,

$$
\tilde y
=
\frac{y-\bar y}{s_y}.
$$

A model may fit in standardized space and untransform posterior means and
variances when returning predictions.  Thresholds, `best_f`, and constraints
must be expressed in the same space expected by the acquisition.

### 11.3 Mixed inputs

Categorical identifiers are not continuous quantities.  A category code of `2`
is not twice category `1`.  Mixed models should:

- transform continuous columns only;
- preserve categorical values;
- use categorical kernels or fixed-feature enumeration;
- verify that an input transform does not modify categorical columns.

### 11.4 Expanded transforms

`InputPerturbation` may change

```text
batch_shape x q x d
```

to

```text
batch_shape x (q * n_w) x d.
```

Training-time transforms must not expand the number of training inputs unless
the targets are expanded consistently.  `bochan` wrappers generally allow such
expansion at acquisition evaluation time only.

---

## 12. Posterior samples and reparameterization

A differentiable Monte Carlo acquisition uses samples

$$
f^{(s)}
=
\mu+L\epsilon^{(s)},
\qquad
\epsilon^{(s)}\sim\mathcal N(0,I),
$$

where

$$
LL^\top=\Sigma.
$$

This reparameterization moves randomness into fixed base samples, allowing
gradients to flow through `mu`, `L`, and ultimately candidate input `X`.

A BoTorch-compatible posterior should implement or support:

```python
posterior.mean
posterior.variance
posterior.rsample(sample_shape)
posterior.rsample_from_base_samples(...)
```

Not every custom posterior represents an exact multivariate Gaussian.  For
example, probability and hybrid posteriors may provide moment-matched or proxy
sampling.  The chapter for each model states the approximation.

---

## 13. Numerical stability

### 13.1 Jitter

A small diagonal term is often added:

$$
K\leftarrow K+\epsilon I.
$$

Jitter stabilizes factorization.  It is not a scientific observation-noise
parameter.

### 13.2 Conditioning

A large condition number can result from:

- duplicate or near-duplicate inputs;
- extremely long or short length scales;
- very small noise;
- badly scaled features;
- redundant task covariance structure.

### 13.3 Cholesky failures

Remedies include:

- increasing jitter;
- using double precision;
- normalizing inputs;
- removing exact duplicates;
- constraining or regularizing hyperparameters;
- examining whether the model is overparameterized.

### 13.4 Small predictive variance

Posterior variance near zero at training points may be valid for nearly
noise-free exact GPs.  A zero variance everywhere, constant posterior mean, or
identical predictions across inputs usually indicates a shape, transform,
conditioning, or model-fitting problem.

---

## 14. Model evaluation

### Predictive mean accuracy

Use held-out RMSE, MAE, or task-appropriate loss.

### Probabilistic accuracy

Use negative log predictive density, calibration, and interval coverage.

### Decision accuracy

For BO, evaluate regret.  For Active Learning, evaluate uncertainty reduction or
predictive loss versus sample count.  For LSE, evaluate set or boundary error.

### Posterior diagnostics

Inspect:

- learned length scales and output scales;
- noise estimates;
- inducing-point coverage;
- posterior covariance eigenvalues;
- calibration by region;
- sensitivity across random initializations or posterior hyperparameter samples.

A high marginal likelihood does not guarantee good sequential decisions.

---

## 15. `bochan` implementation correspondence

### 15.1 Core BoTorch and GPyTorch concepts

| Mathematical object | Implementation concept |
|---|---|
| GP prior | GPyTorch model `forward()` returning `MultivariateNormal` |
| Gaussian likelihood | `GaussianLikelihood`, fixed-noise model, or BoTorch model likelihood |
| Exact marginal likelihood | `ExactMarginalLogLikelihood` |
| Variational posterior | `VariationalStrategy` and `CholeskyVariationalDistribution` |
| Variational objective | `VariationalELBO` |
| Deep variational objective | `DeepApproximateMLL` wrapping an ELBO |
| Predictive posterior | `GPyTorchPosterior` or a task-specific custom posterior |
| Input normalization | BoTorch `Normalize` input transform |
| Outcome standardization | BoTorch `Standardize` outcome transform |

### 15.2 Repository locations

```text
src/bochan/models/
src/bochan/models/components/
src/bochan/models/transforms/
src/bochan/likelihoods/
src/bochan/fit/
```

Representative wrappers include:

- `BinaryClassificationGPModel`;
- `MulticlassClassificationGPModel`;
- `OrdinalGPModel`;
- `DeepGPModel`;
- `HybridMultiOutputModel`.

### 15.3 Posterior conventions in the current codebase

| Model family | Main accessors |
|---|---|
| Standard regression | `posterior(X, observation_noise=...)` |
| Binary classification | `posterior(X)` for probability, `latent_posterior(X)` for latent GP |
| Multiclass classification | `posterior(X)` for probabilities, `latent_posterior(X)` for class-wise latent GP |
| Ordinal | `posterior(X)` for latent GP, `class_probs(X)` for probabilities |
| Hybrid | `posterior(X, output_mode=...)` returning `HybridPosterior` |

Generic acquisition code must inspect these contracts rather than infer them
from the broad task name.

---

## 16. Extension checklist

A new model should document and test:

1. training-data shapes;
2. exact versus variational fitting objective;
3. latent random variable;
4. likelihood and response space;
5. `posterior()` meaning;
6. observation-noise behavior;
7. sample reparameterization;
8. output dimension and extra batch dimensions;
9. input- and outcome-transform behavior;
10. mixed-input handling;
11. `condition_on_observations` or fantasy support;
12. compatibility with analytic and MC acquisitions.

These properties form the implementation-level definition of a probabilistic
surrogate in `bochan`.

---

## 17. References

- Rasmussen and Williams, *Gaussian Processes for Machine Learning*, 2006.
- Titsias, *Variational Learning of Inducing Variables in Sparse Gaussian Processes*, 2009.
- GPyTorch and BoTorch model-interface documentation and source code.
