# 14. Deep and High-dimensional Models

This chapter separates model families that are often grouped together because
they all transform the input space.  Their assumptions are different:

- Deep Kernel Learning uses a deterministic neural feature map inside a GP
  kernel.
- Deep Gaussian Processes compose stochastic GP layers.
- PCA uses a fixed data-dependent linear projection.
- REMBO uses a fixed random low-dimensional embedding.
- SAAS keeps the original axes but places a sparsity prior on inverse length
  scales.
- VAE-GP learns a nonlinear latent representation with reconstruction and GP
  objectives.

These methods should not be selected only because the raw dimension is large.
The correct choice depends on the expected geometry of the effective function.

---

## 1. Deep Kernel Learning

### 1.1 Model

Let a neural network define a deterministic feature map

```math
\phi_\theta:\mathbb R^d\rightarrow\mathbb R^p.
```

A GP is placed on the transformed representation:

```math
f(x)=g(\phi_\theta(x)),
\qquad
 g\sim\mathcal{GP}(m,k).
```

The effective kernel in raw input space is

```math
k_{\mathrm{DKL}}(x,x')
=
k(\phi_\theta(x),\phi_\theta(x')).
```

The neural network is not itself probabilistic in the usual DKL
implementation.  Conditional on learned parameters `theta`, uncertainty comes
from the GP layer.

### 1.2 Training objective

For Gaussian regression, network and GP parameters can be trained jointly by
maximizing the exact marginal likelihood

```math
\log p(\mathbf y\mid \phi_\theta(X),\theta_{\mathrm{GP}}).
```

Equivalently, minimize the negative marginal log likelihood.  Gradients flow
through the kernel matrix and the feature extractor.

### 1.3 Benefits and risks

DKL can represent nonstationary behavior in the raw input space because local
distances are learned by `phi_theta`.  However:

- a flexible feature extractor can overfit small datasets;
- the deterministic network's parameter uncertainty is not represented;
- marginal-likelihood optimization may produce collapsed representations;
- posterior variance can be overconfident outside the training manifold;
- feature scaling and network initialization strongly affect acquisition
  optimization.

Therefore DKL should be compared with an ordinary ARD GP using calibration and
sequential decision metrics, not only training RMSE.

### 1.4 Implementation mapping

Gaussian DKL models are implemented under

```text
src/bochan/models/regression/gaussian/deep/deepkernel.py
```

The wrapper stores raw training inputs, applies `input_transform`, sends the
transformed inputs through a `DeepKernel` or `DeepKernelMixed` feature map, and
uses an inner exact GP.  `forward()` returns the latent GP distribution and
`posterior()` returns a BoTorch-compatible posterior.

The high-level registry keys are

```text
model_type="deepkernel"
```

for regression, binary, multiclass, and ordinal families where registered.

---

## 2. Deep Gaussian Processes

### 2.1 Stochastic composition

A Deep GP composes random functions:

```math
h_1(x)\sim\mathcal{GP}(m_1,k_1),
```

```math
h_2(h_1)\sim\mathcal{GP}(m_2,k_2),
```

```math
\cdots
```

```math
f(x)=h_L(h_{L-1}(\cdots h_1(x))).
```

Unlike DKL, hidden representations are random variables.  Integrating over
intermediate layers produces a non-Gaussian predictive distribution even when
each conditional layer is Gaussian.

### 2.2 Variational training

Each layer commonly uses inducing variables.  A schematic variational
objective is

```math
\mathcal L
=
\mathbb E_{q(f_L)}[\log p(\mathbf y\mid f_L)]
-
\sum_{l=1}^L
\mathrm{KL}[q(u_l)\|p(u_l)].
```

The expected log likelihood is estimated using samples propagated through the
layers.  GPyTorch uses `DeepApproximateMLL` together with a variational ELBO.

### 2.3 Predictive moments

DeepGP prediction often contains an extra sample dimension:

```math
f_*^{(s)}\sim q(f_*),
\qquad s=1,\ldots,S.
```

A moment-matched approximation uses

```math
\hat\mu
=
\frac1S\sum_s\mu_s,
```

```math
\hat\Sigma
=
\frac1S\sum_s
\left[
\Sigma_s+\mu_s\mu_s^\top
\right]
-
\hat\mu\hat\mu^\top.
```

Simply averaging variances omits between-sample variation.  Implementations
that collapse extra DeepGP sample dimensions should document whether they use
full moment matching or a simpler approximation.

### 2.4 Implementation mapping

Gaussian DeepGP models are in

```text
src/bochan/models/regression/gaussian/deep/deepgp.py
```

The implementation contract states:

- `forward()` returns the latent DeepGP distribution for training;
- `posterior()` returns a `GPyTorchPosterior` for acquisitions;
- raw training inputs remain traceable;
- input transforms are applied inside `forward` or `posterior`;
- mixed transforms must not change categorical columns;
- `InputPerturbation` may expand `q` only at evaluation time.

Corresponding classification and ordinal implementations are under

```text
src/bochan/models/classification/*/deep/
src/bochan/models/ordinal/deep/
```

### 2.5 Deep Kernel DeepGP

The combined model first applies a deterministic neural representation and then
passes it through stochastic GP layers:

```math
x\xrightarrow{\phi_\theta}z
\xrightarrow{h_1}\cdots\xrightarrow{h_L}f.
```

This is more expressive but introduces two sources of representation learning
and a difficult optimization problem.  It should be treated as an experimental
model requiring stronger validation than either DKL or a shallow GP.

---

## 3. High-dimensional Bayesian optimization

High-dimensional BO is difficult for at least three reasons:

1. distances concentrate and kernels become weakly informative;
2. acquisition optimization becomes harder;
3. the sample size is usually small relative to dimension.

The methods below solve different versions of this problem.

---

## 4. ARD and SAAS

### 4.1 ARD kernel

An ARD kernel uses one length scale per dimension:

```math
r^2(x,x')
=
\sum_{j=1}^d
\frac{(x_j-x_j')^2}{\ell_j^2}.
```

Large `ell_j` makes the function insensitive to dimension `j`.  Maximum
marginal-likelihood ARD can be unstable when `d` is large and `n` is small.

### 4.2 SAAS prior

SAAS assumes that only a small number of coordinate axes are important.  In a
simplified form, inverse length scales obey a global-local shrinkage prior:

```math
\tau\sim\mathrm{HalfCauchy}(\beta),
```

```math
\rho_j=\ell_j^{-1}\sim\mathrm{HalfCauchy}(\tau).
```

A small global scale shrinks most inverse length scales toward zero, equivalent
to large length scales and low relevance.  A few local scales may escape the
shrinkage.

SAAS is appropriate when the effective subspace is axis aligned.  It is less
appropriate when the response depends on dense linear combinations such as

```math
f(x)=g(a^\top x)
```

with many nonzero components in `a`.

Implementation:

```text
src/bochan/models/regression/gaussian/high_dim/saas.py
src/bochan/models/classification/*/high_dim/saas.py
src/bochan/models/ordinal/high_dim/saas.py
```

The registry key is

```text
model_type="saas"
```

where supported.

---

## 5. PCA-based GP

PCA computes a linear projection from the empirical covariance of inputs.  Let
centered data be

```math
X_c=X-\mathbf 1\bar x^\top.
```

If

```math
X_c=U\Sigma V^\top,
```

the first `p` principal components define

```math
z=V_p^\top(x-\bar x).
```

A GP is fitted in `z` space.

PCA preserves directions of high input variance, not directions predictive of
`Y`.  A low-variance direction can be crucial to the objective and discarded.
PCA is therefore an unsupervised dimensionality-reduction baseline, not a
feature-selection method optimized for BO.

Implementation:

```text
src/bochan/models/regression/gaussian/high_dim/decomposition.py
```

with task-specific variants in corresponding high-dimensional folders.

---

## 6. REMBO

### 6.1 Random embedding

REMBO assumes the function has low effective dimension `p` embedded in a high
ambient dimension `d`.  A random matrix

```math
A\in\mathbb R^{d\times p},
\qquad p\ll d
```

maps a latent candidate `z` to raw space:

```math
x=Az.
```

With box constraints, a projection or clipping map is applied:

```math
x=\Pi_{\mathcal X}(Az).
```

The GP is fitted in the low-dimensional latent coordinates.

### 6.2 Assumptions and failure modes

REMBO can work when the objective varies in an unknown low-dimensional linear
subspace.  Problems include:

- many latent points mapping to the same clipped raw point;
- distorted distances near box boundaries;
- incompatibility with arbitrary nonlinear or compositional constraints;
- sensitivity to random embedding and latent bounds;
- difficulty representing categorical dimensions.

Multiple random embeddings or adaptive embedding methods can reduce dependence
on a single random draw.

Implementation:

```text
src/bochan/models/regression/gaussian/high_dim/decomposition.py
src/bochan/models/ordinal/high_dim/decomposition.py
```

and corresponding task wrappers.  The registry key is `rembo`.

---

## 7. VAE-GP

### 7.1 Latent-variable model

A VAE encoder defines

```math
q_\phi(z\mid x)
=
\mathcal N(\mu_\phi(x),
\mathrm{diag}(\sigma_\phi^2(x))).
```

Using reparameterization,

```math
z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
```

A decoder models

```math
p_\psi(x\mid z),
```

and a GP predicts the target from a latent representation.

The `bochan` VAE-GP loss combines

```math
\mathcal J
=
\lambda_{\mathrm{GP}}\mathcal J_{\mathrm{GP}}
+
\lambda_{\mathrm{rec}}\|x-\hat x\|^2
+
\lambda_{\mathrm{KL}}
\mathrm{KL}[q_\phi(z\mid x)\|p(z)].
```

The GP and acquisition use the deterministic encoder mean

```math
z=\mu_\phi(x)
```

rather than a random sample.  This keeps acquisition evaluation deterministic.
The decoder reconstruction term uses reparameterized latent samples.

### 7.2 Current search space

The current implementation optimizes acquisitions in raw `X` space:

```math
x\rightarrow\mu_\phi(x)\rightarrow\text{GP posterior}.
```

It does not directly optimize an arbitrary latent `z` and decode it to a raw
candidate.  This avoids invalid decoded points but means the raw-space
acquisition problem remains high dimensional.

### 7.3 Mixed inputs

For mixed inputs, only continuous columns are encoded.  Categorical columns are
preserved and concatenated to the latent continuous representation:

```math
[\mu_\phi(x_{\mathrm{cont}}),x_{\mathrm{cat}}].
```

The decoder does not generate categories.  Categorical values must be supplied
when decoding a mixed latent representation.

Implementation and detailed operational documentation:

```text
src/bochan/models/regression/gaussian/high_dim/vae.py
src/bochan/models/regression/gaussian/high_dim/vae_mixed.py
src/bochan/models/regression/gaussian/high_dim/README.md
```

---

## 8. Choosing a model family

| Expected structure | Preferred starting model |
|---|---|
| Smooth function in original coordinates | Standard ARD GP |
| Few relevant original variables | SAAS |
| Low-dimensional linear subspace not axis aligned | REMBO or related embedding |
| High redundancy in observed input distribution | PCA baseline |
| Nonlinear manifold with enough training data | VAE-GP or DKL |
| Strong nonstationarity and enough data | DKL |
| Hierarchical stochastic function composition | DeepGP |
| Mixed categories plus continuous variables | Mixed GP; add deep/high-dimensional methods only after validating category handling |

The standard GP should remain the baseline.  More expressive models increase
optimization and calibration risk.

---

## 9. Validation

Deep and high-dimensional models require more than predictive RMSE.

### 9.1 Predictive quality

- RMSE or MAE;
- negative log predictive density;
- interval coverage;
- calibration by distance from training data.

### 9.2 Sequential quality

- simple regret versus number of observations;
- probability of reaching a target;
- acquisition optimization failures;
- duplicate candidate rate;
- robustness across random seeds.

### 9.3 Representation diagnostics

- latent dimension and reconstruction error for VAE/PCA;
- relevant-dimension posterior for SAAS;
- sensitivity across REMBO embedding seeds;
- pairwise distances before and after DKL transformation;
- DeepGP sample variance decomposition.

### 9.4 Ablations

For learned representations, compare:

1. fixed random network plus GP;
2. pretrained feature extractor plus GP;
3. jointly trained feature extractor plus GP;
4. ordinary ARD GP.

This separates the value of representation learning from the value of the GP.

---

## 10. Source map

| Model | Implementation |
|---|---|
| Gaussian DeepGP | `src/bochan/models/regression/gaussian/deep/deepgp.py` |
| Gaussian DKL | `src/bochan/models/regression/gaussian/deep/deepkernel.py` |
| Deep Kernel DeepGP | `src/bochan/models/regression/gaussian/deep/deepkerneldeepgp.py` |
| Shared deep layers | `src/bochan/models/components/layers.py` |
| Classification deep models | `src/bochan/models/classification/*/deep/` |
| Ordinal deep models | `src/bochan/models/ordinal/deep/` |
| PCA / REMBO | task-specific `high_dim/decomposition.py` modules |
| SAAS | task-specific `high_dim/saas.py` modules |
| VAE-GP | `src/bochan/models/regression/gaussian/high_dim/vae*.py` |
| High-level registry | `src/bochan/api/model_registry.py` |

---

## 11. References

- Damianou and Lawrence, *Deep Gaussian Processes*, 2013.
- Wilson et al., *Deep Kernel Learning*, 2016.
- Wang et al., *Bayesian Optimization in a Billion Dimensions via Random Embeddings*, 2013.
- Eriksson and Jankowiak, *High-dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces*, 2021.
- Kingma and Welling, *Auto-Encoding Variational Bayes*, 2014.
