# 12. Ordinal Models

Ordinal regression models ordered categories such as quality grades, severity
levels, ratings, or process states.  It is not equivalent to ordinary
multiclass classification because the likelihood explicitly encodes the order
between classes.

This chapter describes the ordered-logit model implemented in `bochan`, its
posterior contract, expected utility, boundary probabilities, and its relation
to Bayesian optimization and Level-set Estimation.

---

## 1. Ordered latent-variable model

Let

\[
y\in\{0,1,\ldots,K-1\},
\qquad K\ge 3.
\]

A scalar latent quality function is assigned a GP prior:

\[
f\sim\mathcal{GP}(m,k).
\]

Let the ordered cutpoints be

\[
c_0<c_1<\cdots<c_{K-2}.
\]

Introduce the conventions

\[
c_{-1}=-\infty,
\qquad
c_{K-1}=+\infty.
\]

For an ordered-logit model,

\[
P(Y\le k\mid f)
=
\sigma(c_k-f),
\qquad k=0,\ldots,K-2.
\]

The class probability is a difference of cumulative probabilities:

\[
P(Y=k\mid f)
=
\sigma(c_k-f)-\sigma(c_{k-1}-f).
\]

At the two ends,

\[
P(Y=0\mid f)=\sigma(c_0-f),
\]

\[
P(Y=K-1\mid f)=1-\sigma(c_{K-2}-f).
\]

As `f` increases, probability mass moves toward higher ordered classes.

---

## 2. Cutpoint parameterization

Directly optimizing unconstrained cutpoints can violate their order.  `bochan`
uses positive gaps:

\[
\Delta_j=\operatorname{softplus}(r_j)+\epsilon,
\qquad \Delta_j>0.
\]

When the first cutpoint is fixed,

\[
c_0=0,
\qquad
c_j=\sum_{l=1}^{j}\Delta_l.
\]

When all cutpoints are free, cumulative positive gaps are centered:

\[
\tilde c_j=\sum_{l=0}^{j}\Delta_l,
\qquad
c_j=\tilde c_j-\frac1{K-1}\sum_{r=0}^{K-2}\tilde c_r.
\]

The centering or fixed-first-cutpoint convention addresses a location
non-identifiability.  Shifting the latent function and every cutpoint by the
same constant leaves the class probabilities unchanged:

\[
(f,c_0,\ldots,c_{K-2})
\mapsto
(f+a,c_0+a,\ldots,c_{K-2}+a).
\]

Implementation:

```text
src/bochan/likelihoods/ordinal.py
OrdinalLogitLikelihood.cutpoints
```

---

## 3. Marginal class probabilities

The likelihood defines \(P(Y=k\mid f)\), but prediction requires integration
over the latent posterior:

\[
p_k(x)
=
P(Y=k\mid x,\mathcal D)
=
\int P(Y=k\mid f)p(f\mid x,\mathcal D)df.
\]

`OrdinalLogitLikelihood.marginal_class_probs` performs one-dimensional Gaussian
quadrature for each class.  Therefore

```python
model.class_probs(X)
```

uses the latent posterior uncertainty, not only the posterior mean.

A plug-in probability

\[
P(Y=k\mid f=\mu_f(x))
\]

is cheaper but is not generally equal to the marginal probability.

---

## 4. Implementation posterior contract

The public base model is

```text
OrdinalGPModel
OrdinalMixedGPModel
```

with implementation in

```text
src/bochan/models/ordinal/base/
```

Unlike binary and multiclass wrappers, the ordinal model currently defines

```python
latent_posterior = model.posterior(X)
```

as the posterior of the scalar latent function.  It does **not** return class
probabilities.

The probability-space API is

```python
probs = model.class_probs(X)
predicted_class = model.predict_class(X)
expected_utility = model.expected_utility(X, utilities)
```

This difference must be respected by acquisition functions:

| Model family | `posterior(X)` in current implementation |
|---|---|
| Binary classification | probability posterior |
| Multiclass classification | probability posterior |
| Ordinal | latent scalar posterior |

For generic code, test model capabilities rather than assuming all
classification-like models return probabilities from `posterior()`.

---

## 5. Variational inference

The ordered likelihood is non-Gaussian, so the model uses a sparse variational
latent GP.  With inducing variables \(u=f(Z)\), optimize

\[
\mathcal L
=
\sum_{i=1}^n
\mathbb E_{q(f_i)}
  [\log P(y_i\mid f_i,c)]
-
\operatorname{KL}[q(u)\|p(u)].
\]

Both GP parameters and cutpoint gaps are learned.  The cutpoints should be
included in the optimizer through the likelihood parameters.

The latent model classes include

```text
_OrdinalLatentGP
_MixedOrdinalLatentGP
```

and use inducing-point variational distributions.

---

## 6. Expected ordinal utility

Assign a utility to each ordered class:

\[
\mathbf u=(u_0,\ldots,u_{K-1}).
\]

The posterior expected utility is

\[
U(x)
=
\mathbb E[u_Y\mid x,\mathcal D]
=
\sum_{k=0}^{K-1}u_kp_k(x).
\]

Typical utility choices include:

### Class index

\[
u_k=k.
\]

This assumes equal spacing between adjacent grades.

### Normalized index

\[
u_k=\frac{k}{K-1}.
\]

This maps utility to `[0, 1]` but still assumes equal spacing.

### Domain-specific utility

\[
\mathbf u=(0,0.1,0.7,1.0).
\]

This is appropriate when improvement from class 1 to 2 is more valuable than
improvement from class 0 to 1.

Expected utility provides a scalar objective for BO, but it does not remove the
uncertainty in the class distribution.  Two candidates can have the same
expected utility and very different risk profiles.

Implementation:

```text
src/bochan/acquisition/objective/ordinal.py
src/bochan/acquisition/ordinal/bayesian_optimization/utility_acquisitions.py
```

---

## 7. Boundary probabilities

Each cutpoint defines a boundary between adjacent ordered groups.

For boundary `j`, define the cumulative upper-group probability

\[
q_j(x)
=
P(Y\ge j+1\mid x,\mathcal D)
=
\sum_{k=j+1}^{K-1}p_k(x).
\]

In the deterministic latent limit,

\[
q_j(x)=P(f>c_j\mid x,\mathcal D)
\]

under the corresponding threshold interpretation.

A natural boundary-ambiguity score is

\[
A_j(x)=4q_j(x)[1-q_j(x)].
\]

Properties:

- `A_j=1` at `q_j=0.5`;
- `A_j=0` at `q_j=0` or `1`;
- it is symmetric around the boundary probability.

The implementation uses this form in

```text
ordinal_boundary_uncertainty
```

inside

```text
src/bochan/acquisition/ordinal/levelset_estimation/single_output.py
```

---

## 8. Latent cutpoint distance

When the acquisition operates directly in latent space, boundary `j` is

\[
f(x)=c_j.
\]

Given posterior mean and standard deviation

\[
\mu_f(x),\qquad \sigma_f(x),
\]

a Straddle-like score for boundary `j` is

\[
\alpha_j(x)
=
\beta\sigma_f(x)-|\mu_f(x)-c_j|.
\]

For a set of boundaries \(J\), reduce the boundary-wise scores using

\[
\operatorname{mean}_{j\in J}\alpha_j,
\quad
\sum_{j\in J}\alpha_j,
\quad
\max_{j\in J}\alpha_j,
\quad
\min_{j\in J}\alpha_j.
\]

These reductions have different meanings:

- `max`: sample near any one uncertain boundary;
- `mean`: balance all boundaries;
- `sum`: similar to mean but scale depends on boundary count;
- `min`: favor points useful for every selected boundary, often restrictive.

The API argument `target_boundary_idx` selects a specific cutpoint.  The index
refers to cutpoint `c_j`, not class `j` itself.

---

## 9. Ordinal Bayesian optimization

### 9.1 Expected-utility improvement

Let

\[
U(x)=\sum_k u_kp_k(x).
\]

A deterministic improvement score is

\[
I_U(x)=\max(U(x)-U_{\mathrm{best}},0).
\]

A Monte Carlo expected-improvement variant samples latent functions, converts
each sample to class probabilities or utilities, and averages the positive
improvement:

\[
\operatorname{EI}_U(x)
=
\mathbb E\left[
\max(U^{(s)}(x)-U_{\mathrm{best}},0)
\right].
\]

The `best_f` reference must be expressed in the same utility scale.

### 9.2 Target class or grade probability

For a required minimum grade `g`, optimize

\[
P(Y\ge g\mid x)
=
\sum_{k=g}^{K-1}p_k(x).
\]

This is often more interpretable than maximizing class-index utility.  It can
also be used as a feasibility probability.

### 9.3 Risk-sensitive ordinal objective

For class utility `u_Y`, possible risk summaries include:

\[
\mathbb E[u_Y],
\qquad
\operatorname{VaR}_{\alpha}(u_Y),
\qquad
\operatorname{CVaR}_{\alpha}(u_Y).
\]

Because ordinal utility is discrete, VaR can jump abruptly when class
probability crosses a quantile boundary.  CVaR is usually smoother but still
requires a clear convention for whether the lower or upper tail is undesirable.

---

## 10. Ordinal Active Learning

Possible information targets are different:

1. the scalar latent function `f`;
2. cutpoint locations `c_j`;
3. the complete class-probability vector;
4. one selected boundary probability `q_j`;
5. expected utility;
6. the predicted class label.

An acquisition should state which target it learns.

### Predictive entropy

\[
H(Y\mid x,\mathcal D)
=-\sum_{k=0}^{K-1}p_k(x)\log p_k(x).
\]

This selects ambiguous grade predictions.

### Ordinal BALD

\[
I(Y;f\mid x,\mathcal D)
=
H\!\left(\mathbb E_f[p(Y\mid f)]\right)
-
\mathbb E_f[H(p(Y\mid f))].
\]

This distinguishes latent epistemic uncertainty from unavoidable class overlap
induced by the ordered likelihood.

### Boundary-specific entropy

For boundary `j`, treat

\[
Z_j=\mathbf 1[Y\ge j+1]
\]

as a binary event and use

\[
H(Z_j)
=-q_j\log q_j-(1-q_j)\log(1-q_j).
\]

This is useful when only one engineering grade transition matters.

---

## 11. Multi-output ordinal models

For `m` ordinal outputs, independent models imply

\[
p(\mathbf y\mid x,\mathcal D)
=
\prod_{r=1}^m p(y_r\mid x,\mathcal D_r).
\]

This is the practical ModelList-style interpretation used by many `bochan`
wrappers.  It supports different class counts and cutpoints per output, but it
does not model cross-output dependence.

A correlated ordinal multitask model instead introduces vector-valued latent
functions

\[
\mathbf f(x)=(f_1(x),\ldots,f_m(x))
\]

with

\[
\operatorname{Cov}[f_r(x),f_s(x')]
=k_X(x,x')B_{rs}.
\]

Each output can retain its own cutpoints and likelihood.  The task covariance
`B` describes latent association, not observed-label Pearson correlation.

Implementation families include ModelList-style wrappers, explicit multitask
models, and Kronecker variants under

```text
src/bochan/models/ordinal/base/
```

---

## 12. Heteroscedastic ordinal caveat

An ordered-logit likelihood already contains stochastic class overlap.  A
second input-dependent noise model must have a specified role, for example:

- a scale parameter in `sigma((c_j-f)/s(x))`;
- uncertainty in externally supplied grade labels;
- annotator reliability;
- an acquisition penalty unrelated to the generative likelihood.

These alternatives lead to different probabilities.  A principled
input-dependent scale model is

\[
P(Y\le j\mid x)
=
\sigma\left(
\frac{c_j-f(x)}{s(x)}
\right),
\qquad s(x)>0.
\]

Larger `s(x)` makes class transitions more diffuse.  If the current wrapper only
predicts a separate noise score and combines it with an acquisition, describe
it as noise-aware acquisition weighting rather than a full heteroscedastic
ordered-logit likelihood.

Relevant code:

```text
src/bochan/models/ordinal/robust/heteroscedastic.py
src/bochan/acquisition/ordinal/active_learning/hetero_single_output.py
src/bochan/acquisition/ordinal/active_learning/hetero_multi_output.py
src/bochan/acquisition/ordinal/levelset_estimation/hetero_single_output.py
src/bochan/acquisition/ordinal/levelset_estimation/hetero_multi_output.py
```

---

## 13. Source map

| Theory component | Implementation |
|---|---|
| Ordered-logit likelihood and cutpoints | `src/bochan/likelihoods/ordinal.py` |
| Base continuous and mixed ordinal GP | `src/bochan/models/ordinal/base/` |
| Class probabilities and expected utility | methods on `OrdinalGPModel` and `OrdinalLogitLikelihood` |
| Posterior transforms | `src/bochan/models/transforms/posterior/ordinal.py` |
| Utility objectives | `src/bochan/acquisition/objective/ordinal.py` |
| BO acquisitions | `src/bochan/acquisition/ordinal/bayesian_optimization/` |
| Active Learning | `src/bochan/acquisition/ordinal/active_learning/` |
| Level-set Estimation | `src/bochan/acquisition/ordinal/levelset_estimation/` |
| Heteroscedastic support | `src/bochan/models/ordinal/robust/` and ordinal `hetero_*.py` acquisition modules |
| Deep ordinal models | `src/bochan/models/ordinal/deep/` |
| High-dimensional ordinal models | `src/bochan/models/ordinal/high_dim/` |

---

## 14. References

- Chu and Ghahramani, *Gaussian Processes for Ordinal Regression*, JMLR, 2005.
- Titsias, *Variational Learning of Inducing Variables in Sparse Gaussian Processes*, 2009.
- Houlsby et al., *Bayesian Active Learning for Classification and Preference Learning*, 2011.
