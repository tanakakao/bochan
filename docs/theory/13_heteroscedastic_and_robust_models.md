# 13. Heteroscedastic and Robust Models

This chapter separates several concepts that are often grouped under the word
"noise": known measurement variance, learned input-dependent observation noise,
outliers, label ambiguity, input perturbation, and model uncertainty.

They require different probabilistic models and should not be combined only by
adding variances unless the resulting generative model is explicitly defined.

---

## 1. Decomposition of uncertainty

For a continuous observation

$$
Y=f(X)+\varepsilon,
$$

the predictive variance may be decomposed conceptually as

$$
\operatorname{Var}(Y_*\mid\mathcal D)
=
\underbrace{\operatorname{Var}(f_*\mid\mathcal D)}_{\text{epistemic}}
+
\underbrace{\mathbb E[\sigma_n^2(X_*)\mid\mathcal D]}_{\text{aleatoric}}
+
\text{additional approximation terms}.
$$

Epistemic uncertainty may decrease after informative observations.  Aleatoric
noise is inherent in repeated measurements at the same input unless the data
collection mechanism is improved.

Input perturbation is separate.  If the executed input is random,

$$
\widetilde X=X+\Delta,
$$

then variability in \(f(\widetilde X)\) is induced by uncertain inputs even if
the observation likelihood is noiseless.

---

## 2. Known heteroscedastic observation variance

When each experiment has a known measurement variance \(s_i^2\), use

$$
y_i\mid f_i\sim\mathcal N(f_i,s_i^2).
$$

The covariance of the observations is

$$
K_y=K_f+\operatorname{diag}(s_1^2,\ldots,s_n^2).
$$

This is appropriate when variances come from:

- repeated measurements;
- instrument calibration;
- a known propagation-of-error model;
- an external estimator with justified uncertainty.

Known variance is passed as `train_Yvar`.  It should not be re-estimated by a
second GP unless the goal is to smooth or extrapolate those variance estimates.

---

## 3. Learned heteroscedastic Gaussian regression

### 3.1 Two-latent-function model

A standard formulation is

$$
y_i=f(x_i)+\varepsilon_i,
\qquad
\varepsilon_i\sim\mathcal N(0,\sigma^2(x_i)),
$$

with

$$
f\sim\mathcal{GP}(m_f,k_f),
$$

$$
g\sim\mathcal{GP}(m_g,k_g),
\qquad
\sigma^2(x)=\exp(g(x))
$$

or

$$
\sigma^2(x)=\operatorname{softplus}(g(x))+\epsilon.
$$

The full posterior is

$$
p(f,g\mid\mathbf y)
\propto
p(\mathbf y\mid f,g)p(f)p(g),
$$

which is non-conjugate because the covariance depends on the latent process
`g`.

### 3.2 Residual-based approximation

A common practical approximation is:

1. fit a mean GP \(\hat f\);
2. compute residuals
$$
   r_i=y_i-\hat f(x_i);
$$
3. construct log-variance targets
$$
   z_i=\log(r_i^2+\epsilon);
$$
4. fit a second GP to \((x_i,z_i)\);
5. predict
$$
   \hat\sigma^2(x)=\exp(\mu_g(x)).
$$

This is not joint Bayesian inference.  The residual targets depend on the
estimated mean, and uncertainty in the mean GP is not fully propagated into the
noise model.  It is nevertheless useful as an engineering approximation.

`bochan` provides shared helpers for this pattern in

```text
src/bochan/models/components/heteroscedastic.py
```

including functions for residual log variance, fitting a noise model, and
mapping log-noise predictions back to variance.

### 3.3 Prediction

For a test point,

$$
\mu_Y(x)=\mu_f(x),
$$

$$
\operatorname{Var}(Y\mid x,\mathcal D)
\approx
\operatorname{Var}(f(x)\mid\mathcal D)
+
\hat\sigma^2(x).
$$

This addition is valid for Gaussian observations when the noise is conditionally
independent of the latent mean and the noise estimate is treated as fixed.

The latent variance alone should be used for tasks that value reducible model
uncertainty, such as learning the underlying process function.  The total
predictive variance is appropriate for prediction intervals of future
measurements.

---

## 4. Heteroscedastic acquisition design

There is no universally correct way to use the learned noise.

### 4.1 Avoiding irreducibly noisy points

A noise-penalized score can be written as

$$
\alpha_{\mathrm{pen}}(x)
=
\alpha_0(x)w(\sigma_n^2(x)),
$$

with, for example,

$$
w_{\mathrm{linear}}(v)=\frac{1}{1+\lambda v},
$$

or

$$
w_{\mathrm{exp}}(v)=\exp(-\lambda v).
$$

This is reasonable when the objective is to learn a reproducible process and
measurements in high-noise regions have low value.

### 4.2 Learning the noise function

If noise itself is scientifically important, high-noise or uncertain-noise
regions may deserve more observations.  A criterion may combine mean-model and
noise-model information:

$$
\alpha(x)
=
\alpha_f(x)+\lambda\alpha_g(x).
$$

Penalizing high predicted noise would work against this goal.

### 4.3 Repeated measurements

When an existing point is noisy, a replicate can reduce uncertainty about its
mean.  A candidate-diversity penalty that forbids duplicate points may therefore
be inappropriate for heteroscedastic experiments.  Duplicate suppression and
replicate design should be configured separately.

---

## 5. Heteroscedastic classification

Classification already has a non-Gaussian likelihood.  For binary data,

$$
Y\mid f\sim\operatorname{Bernoulli}(\pi(f)).
$$

The conditional observation variance is

$$
\operatorname{Var}(Y\mid f)=\pi(f)[1-\pi(f)].
$$

An additional `noise_model` needs a precise interpretation.

### 5.1 Label-flip model

Let \(\rho(x)\in[0,0.5)\) be the probability that a label is flipped.  Then

$$
P(Y_{\mathrm{obs}}=1\mid f,x)
=
[1-\rho(x)]\pi(f)+\rho(x)[1-\pi(f)].
$$

This shrinks probabilities toward `0.5` in unreliable regions.

### 5.2 Input-dependent temperature

For logit classification,

$$
P(Y=1\mid f,x)
=
\sigma\left(\frac{f(x)}{s(x)}\right),
\qquad s(x)>0.
$$

A large scale `s(x)` makes the class transition diffuse.

### 5.3 Probability-estimate noise

If training targets are estimated probabilities rather than binary labels, an
additional variance can describe uncertainty in those estimates.  This is not a
standard Bernoulli-label model.

### 5.4 Current implementation interpretation

The current binary classification wrapper can add a predicted or supplied
variance to `SimpleBernoulliPosterior.variance`.  This creates a useful
noise-aware posterior interface, but it should not be described as an exact
label-flip or temperature likelihood unless that likelihood is actually used in
training.

Relevant code:

```text
src/bochan/models/classification/binary/robust/heteroscedastic.py
src/bochan/models/classification/multiclass/robust/heteroscedastic.py
src/bochan/models/components/heteroscedastic.py
```

---

## 6. Heteroscedastic ordinal regression

A principled input-dependent ordered-logit scale is

$$
P(Y\le j\mid x)
=
\sigma\left(\frac{c_j-f(x)}{s(x)}\right),
\qquad s(x)>0.
$$

Class probabilities are

$$
P(Y=j\mid x)
=
\sigma\left(\frac{c_j-f(x)}{s(x)}\right)
-
\sigma\left(\frac{c_{j-1}-f(x)}{s(x)}\right).
$$

A larger scale produces more overlap between adjacent grades.  This model is
different from adding a noise variance after computing ordinal probabilities.

The current ordinal heteroscedastic wrappers and acquisitions should be
interpreted according to their actual implementation:

```text
src/bochan/models/ordinal/robust/heteroscedastic.py
src/bochan/acquisition/ordinal/active_learning/hetero_single_output.py
src/bochan/acquisition/ordinal/active_learning/hetero_multi_output.py
src/bochan/acquisition/ordinal/levelset_estimation/hetero_single_output.py
src/bochan/acquisition/ordinal/levelset_estimation/hetero_multi_output.py
```

When the noise model only reweights an acquisition, the correct phrase is
"noise-aware ordinal acquisition", not necessarily "fully heteroscedastic
ordered-logit inference".

---

## 7. Robust Relevance Pursuit

Robust Relevance Pursuit (RRP) and outlier relevance models aim to identify a
small subset of observations or components that need additional flexibility.
A generic robust observation model is

$$
y_i=f(x_i)+o_i+\varepsilon_i,
$$

where the outlier effects \(o_i\) are sparse.

A sparsity-inducing prior or iterative pursuit mechanism prefers

$$
\|\mathbf o\|_0\ll n
$$

or a continuous relaxation such as an \(\ell_1\)-type penalty.

This is different from ARD or SAAS:

| Method | Sparse object |
|---|---|
| RRP / outlier pursuit | observations, likelihood terms, or local corrections |
| ARD | input sensitivity through length scales |
| SAAS | dimensions through a shrinkage prior on inverse length scales |
| k-sparse candidate repair | nonzero components of a proposed design point |

The terms should not be used interchangeably.

Main implementation families:

```text
src/bochan/models/regression/gaussian/robust/relevance_pursuit.py
src/bochan/models/classification/binary/robust/
src/bochan/models/classification/multiclass/robust/
src/bochan/models/ordinal/robust/
src/bochan/fit/robust/
```

---

## 8. Outliers versus heavy-tailed noise

Sparse outlier correction is one robustness strategy.  Another is a heavy-tailed
likelihood, for example

$$
y_i\mid f_i\sim\operatorname{StudentT}(\nu,f_i,\sigma).
$$

A Student-t likelihood downweights large residuals continuously, whereas sparse
outlier pursuit attempts to isolate a small number of exceptional points.

The correct choice depends on the data-generating mechanism:

- occasional sensor failures: sparse outlier model;
- consistently heavy-tailed fluctuations: Student-t likelihood;
- variance changing with process conditions: heteroscedastic Gaussian model;
- wrong class labels: label-noise classification model.

---

## 9. Input perturbation is not observation noise

Let a nominal candidate be `x`, but the executed condition be

$$
\widetilde x=x+\delta,
\qquad
\delta\sim p(\delta).
$$

A robust objective may be

$$
\rho[f(\widetilde x)]
$$

where `rho` is mean, VaR, CVaR, or another risk functional.

This uncertainty appears before the function evaluation, not after it.  The
implementation expands each nominal candidate into `n_w` perturbed inputs:

```text
batch_shape x q x d
    -> batch_shape x (q * n_w) x d
```

The objective then reduces each block of `n_w` values back to one nominal
candidate.  Heteroscedastic observation noise and input perturbation can both be
present, but their effects should be modeled in different stages.

---

## 10. Shape and transform rules

The shared heteroscedastic helpers enforce several implementation contracts:

1. `train_Yvar` is converted to shape `N x 1` and clamped to a positive floor.
2. A noise model should generally use only the normalization part of an input
   transform, not an `InputPerturbation` transform that expands `q`.
3. For mixed models, normalization must exclude categorical dimensions.
4. Predicted noise tensors must be aligned with posterior mean/variance shapes.
5. If a nominal `q` is expanded to `q * n_w`, raw candidate points may need
   `repeat_interleave` only for distance or categorical consistency checks.

Implementation:

```text
src/bochan/models/components/heteroscedastic.py
```

---

## 11. Evaluation criteria

A heteroscedastic model should not be evaluated only by RMSE.  Recommended
metrics are:

### Mean prediction

$$
\operatorname{RMSE}
=
\sqrt{\frac1n\sum_i(y_i-\hat\mu_i)^2}.
$$

### Predictive log likelihood

$$
\operatorname{NLPD}
=-\frac1n\sum_i\log p(y_i\mid x_i,\mathcal D).
$$

### Interval coverage

For a nominal coverage `1-alpha`, compare

$$
\frac1n\sum_i
\mathbf 1[y_i\in C_{1-\alpha}(x_i)]
$$

with `1-alpha`.

### Noise-model quality

When replicate-based variance estimates are available, compare predicted and
empirical variances on a log scale and assess calibration by noise strata.

### Decision quality

For robust optimization, evaluate the true repeated-experiment mean, variance,
quantiles, and constraint violation rate at the recommended candidate.

---

## 12. Source map

| Component | Implementation |
|---|---|
| Shared heteroscedastic utilities | `src/bochan/models/components/heteroscedastic.py` |
| Gaussian heteroscedastic models | `src/bochan/models/regression/gaussian/robust/heteroscedastic.py` |
| Non-Gaussian heteroscedastic models | `src/bochan/models/regression/non_gaussian/*/robust/` |
| Binary heteroscedastic models | `src/bochan/models/classification/binary/robust/heteroscedastic.py` |
| Multiclass heteroscedastic models | `src/bochan/models/classification/multiclass/robust/heteroscedastic.py` |
| Ordinal heteroscedastic models | `src/bochan/models/ordinal/robust/heteroscedastic.py` |
| Noise-aware ordinal acquisitions | `src/bochan/acquisition/ordinal/active_learning/hetero_*.py`, `src/bochan/acquisition/ordinal/levelset_estimation/hetero_*.py` |
| RRP fitting | `src/bochan/fit/robust/` |

---

## 13. References

- Goldberg, Williams, and Bishop, *Regression with Input-dependent Noise: A Gaussian Process Treatment*, 1998.
- Rasmussen and Williams, *Gaussian Processes for Machine Learning*, 2006.
- Jankowiak, Pleiss, and Gardner, work on robust Gaussian-process inference and relevance pursuit as implemented in the GPyTorch ecosystem.
