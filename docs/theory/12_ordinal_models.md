# 12. Ordinal Models

Ordinal regression models ordered labels such as quality grades, severity
levels, ratings, or process states.  It differs from ordinary multiclass
classification because one scalar latent ordering and a sequence of cutpoints
encode class order.

This chapter focuses on the ordered-logit model, identifiability, variational
inference, marginal class probabilities, utility summaries, and the current
`bochan` implementation contract.

Decision criteria are treated separately:

- Chapter 04: ordinal Active Learning;
- Chapter 06: ordinal outputs in Bayesian optimization and constraints;
- Chapter 16: ordinal Level-set Estimation formulas.

---

## 1. Ordinal response

Let

$$
Y\in\{0,1,\ldots,K-1\},
\qquad K\ge3,
$$

with order

$$
0<1<\cdots<K-1.
$$

The labels indicate ranking, but the numeric spacing is not necessarily equal.
For example, the difference between grades 0 and 1 need not equal the
difference between grades 2 and 3.

Treating labels as Gaussian regression targets incorrectly assumes metric
spacing and Gaussian residuals.  Treating them as unordered multiclass labels
ignores the known ordering.

---

## 2. Latent-threshold construction

Introduce a scalar latent function

$$
f\sim\mathcal{GP}(m,k)
$$

and ordered cutpoints

$$
c_0<c_1<\cdots<c_{K-2}.
$$

Use the conventions

$$
c_{-1}=-\infty,
\qquad
c_{K-1}=+\infty.
$$

The class is determined by the interval containing an unobserved continuous
latent variable.

In a deterministic threshold representation,

$$
Y=k
\quad\Longleftrightarrow\quad
c_{k-1}<f(x)\le c_k.
$$

A probabilistic ordinal likelihood adds link noise around these thresholds.

---

## 3. Ordered-logit likelihood

The cumulative probability is

$$
P(Y\le k\mid f)
=
\sigma(c_k-f),
\qquad k=0,\ldots,K-2.
$$

Equivalently,

$$
P(Y>k\mid f)
=
\sigma(f-c_k).
$$

Class probabilities are differences of adjacent cumulative probabilities:

$$
P(Y=k\mid f)
=
\sigma(c_k-f)
-
\sigma(c_{k-1}-f).
$$

At the endpoints:

$$
P(Y=0\mid f)
=
\sigma(c_0-f),
$$

$$
P(Y=K-1\mid f)
=
1-\sigma(c_{K-2}-f).
$$

As `f` increases, probability mass shifts toward higher classes.

---

## 4. Latent-noise interpretation

Ordered logit can be written using latent continuous response

$$
Z=f(x)+\epsilon,
$$

where `epsilon` follows a standard logistic distribution.  The observed class
is

$$
Y=k
\quad\Longleftrightarrow\quad
c_{k-1}<Z\le c_k.
$$

Then

$$
P(Y\le k\mid f)
=P(Z\le c_k\mid f)
=\sigma(c_k-f).
$$

An ordered-probit model instead uses Gaussian latent noise and cumulative normal
probabilities.  Logit and probit latent scales are not numerically identical.

---

## 5. Cutpoint ordering

Directly optimizing unrestricted cutpoints can violate

$$
c_0<c_1<\cdots<c_{K-2}.
$$

`bochan` parameterizes positive gaps:

$$
\Delta_j
=
\operatorname{softplus}(r_j)+\epsilon,
\qquad
\Delta_j>0.
$$

### Fixed first cutpoint

When `fix_first_cutpoint=True`,

$$
c_0=0,
$$

and remaining cutpoints are cumulative sums:

$$
c_j
=
\sum_{l=1}^{j}\Delta_l.
$$

### Free centered cutpoints

When the first cutpoint is not fixed, construct cumulative values

$$
\tilde c_j
=
\sum_{l=0}^{j}\Delta_l
$$

and center them:

$$
c_j
=
\tilde c_j
-
\frac1{K-1}
\sum_{r=0}^{K-2}\tilde c_r.
$$

Both constructions preserve strict order.

---

## 6. Location identifiability

The ordered likelihood is invariant to a common shift:

$$
f(x)\mapsto f(x)+a,
$$

$$
c_j\mapsto c_j+a.
$$

Because

$$
(c_j+a)-(f+a)=c_j-f,
$$

class probabilities do not change.

Therefore the model needs a location convention, such as:

- fixing the first cutpoint;
- centering all cutpoints;
- constraining the latent mean.

Without a convention, the latent level and cutpoint locations are not separately
identified.

---

## 7. Scale interpretation

In a standard ordered-logit model, logistic noise scale is fixed.  This fixes
the relative scale of latent function and cutpoint gaps.

If an additional scale `s` is introduced,

$$
P(Y\le k\mid f)
=
\sigma\left(
\frac{c_k-f}{s}
\right),
$$

then multiplying `f`, cutpoints, and `s` by a common factor creates another
identifiability issue unless one component is constrained.

The current base `OrdinalLogitLikelihood` uses the fixed logistic scale and
learned cutpoint gaps.

---

## 8. Likelihood for observed labels

For training labels `y_i`, the likelihood is

$$
p(\mathbf y\mid\mathbf f,\mathbf c)
=
\prod_{i=1}^{n}
P(Y_i=y_i\mid f_i,\mathbf c).
$$

The log likelihood is

$$
\log p(\mathbf y\mid\mathbf f,\mathbf c)
=
\sum_i
\log P(Y_i=y_i\mid f_i,\mathbf c).
$$

Very small class probabilities can create numerical instability.  The
implementation clamps probabilities by an `eps` floor and renormalizes them.

---

## 9. Sparse variational inference

The ordered likelihood is non-Gaussian.  The model uses inducing variables

$$
\mathbf u=f(Z)
$$

and variational distribution

$$
q(\mathbf u)
=
\mathcal N(\mathbf m_u,S_u).
$$

The induced latent posterior is

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
\sum_i
\mathbb E_{q(f_i)}
[
\log P(y_i\mid f_i,\mathbf c)
]
-
\operatorname{KL}
[q(\mathbf u)\|p(\mathbf u)].
$$

The optimization variables include:

- variational mean and covariance;
- kernel parameters;
- mean-function parameters;
- inducing locations when learnable;
- raw cutpoint gaps.

Cutpoints are likelihood parameters and must be included in the optimizer.

---

## 10. Quadrature for likelihood expectations

For one-dimensional latent `f_i`, expectations such as

$$
\mathbb E_{q(f_i)}
[
\log P(y_i\mid f_i)
]
$$

and marginal class probabilities can be evaluated by Gaussian quadrature.

If

$$
f_i\sim\mathcal N(\mu_i,\sigma_i^2),
$$

then

$$
\mathbb E[g(f_i)]
=
\int g(f)
\mathcal N(f;\mu_i,\sigma_i^2)df.
$$

Gauss-Hermite quadrature approximates this integral as a weighted sum at
transformed nodes.  It is practical because the ordinal latent dimension per
observation is scalar.

---

## 11. Marginal class probabilities

The conditional likelihood probability is

$$
P(Y=k\mid f).
$$

The predictive probability integrates latent uncertainty:

$$
p_k(x)
=
P(Y=k\mid x,\mathcal D)
=
\int
P(Y=k\mid f)
q(f\mid x,\mathcal D)df.
$$

This is not generally equal to the plug-in probability

$$
P(Y=k\mid f=\mu_f(x)).
$$

`OrdinalLogitLikelihood.marginal_class_probs` applies quadrature separately for
each class and renormalizes the result.

---

## 12. Cumulative probabilities

Ordered models are naturally interpreted through cumulative events.

### Probability below or equal to grade `k`

$$
P(Y\le k\mid x)
=
\sum_{r=0}^{k}p_r(x).
$$

### Probability at or above grade `g`

$$
P(Y\ge g\mid x)
=
\sum_{r=g}^{K-1}p_r(x).
$$

For boundary `j` between classes `j` and `j+1`, define

$$
g_j(x)
=
P(Y\ge j+1\mid x).
$$

These cumulative probabilities are useful for calibration, constraints, and
boundary interpretation.

---

## 13. Class prediction

The maximum-probability class is

$$
\hat y_{\mathrm{mode}}(x)
=
\arg\max_kp_k(x).
$$

For an ordered absolute-error loss, the Bayes-optimal class is a posterior
median, not necessarily the mode.

A posterior median class `k*` satisfies

$$
P(Y\le k^*\mid x)
\ge\frac12
$$

and

$$
P(Y\ge k^*\mid x)
\ge\frac12.
$$

For squared error on numeric class scores, the posterior mean class index may be
optimal, but this assumes equal numeric spacing.

The current `predict_class(X)` returns the probability mode through `argmax`.

---

## 14. Expected utility

Assign class utilities

$$
\mathbf u=(u_0,\ldots,u_{K-1}).
$$

Expected utility is

$$
U(x)
=
\sum_{k=0}^{K-1}u_kp_k(x).
$$

The likelihood encodes order; utility encodes user preference.  They are not the
same.

### Equal class index

$$
u_k=k
$$

assumes equal spacing.

### Domain-specific utility

$$
\mathbf u=(0,0.1,0.7,1.0)
$$

can encode a large value increase when a specification grade is reached.

`model.expected_utility(X, utilities)` uses marginalized class probabilities,
not only the latent mean.

---

## 15. Utility variance

For realized class utility `U_Y=u_Y`, conditional variance is

$$
\operatorname{Var}(U_Y\mid x)
=
\sum_kp_k(x)
[u_k-U(x)]^2.
$$

This measures uncertainty of the next realized class utility given the
predictive class probabilities.

It is not automatically posterior epistemic variance of the expected-utility
function.  The latter requires posterior samples of class probabilities or
latent functions:

$$
\operatorname{Var}_{f\mid\mathcal D}
\left[
\sum_ku_kP(Y=k\mid f)
\right].
$$

---

## 16. Boundary interpretation

Cutpoint `c_j` separates lower classes

$$
\{0,\ldots,j\}
$$

from upper classes

$$
\{j+1,\ldots,K-1\}.
$$

At fixed latent value,

$$
P(Y\ge j+1\mid f)
=
\sigma(f-c_j).
$$

At

$$
f=c_j,
$$

this cumulative probability is `0.5` under the logistic likelihood.

For a posterior over `f`, the marginalized cumulative probability also depends
on latent variance.  A contour of posterior mean `mu_f=c_j` is therefore a
latent boundary approximation, while

$$
P(Y\ge j+1\mid x)=0.5
$$

is a probability-space boundary.

---

## 17. Boundary ambiguity

For cumulative upper probability

$$
g_j(x)=P(Y\ge j+1\mid x),
$$

a normalized ambiguity score is

$$
A_j(x)
=4g_j(x)[1-g_j(x)].
$$

Properties:

$$
A_j=1
\quad\text{at}\quad g_j=0.5,
$$

$$
A_j=0
\quad\text{at}\quad g_j\in\{0,1\}.
$$

This score is used by ordinal LSE ICU-style acquisitions described in Chapter
16.  It is predictive boundary ambiguity, not latent posterior variance.

---

## 18. Ordinal posterior contract in `bochan`

The main models are

```text
OrdinalGPModel
OrdinalMixedGPModel
```

under

```text
src/bochan/models/ordinal/base/
```

The current contract is:

```python
latent = model.posterior(X)
probs = model.class_probs(X)
predicted_class = model.predict_class(X)
utility = model.expected_utility(X, utilities)
```

### `posterior(X)`

Returns a `GPyTorchPosterior` over scalar latent `f(x)`.

It does not return class probabilities.

### `class_probs(X)`

Uses the latent posterior distribution and ordinal likelihood quadrature to
return

```text
batch_shape x q x K
```

probabilities.

### `expected_utility(X, utilities)`

Returns

```text
batch_shape x q
```

expected utilities.

This differs from binary and multiclass wrappers, whose base `posterior()`
methods return probability-space posteriors.

---

## 19. Training-input contract

The base ordinal wrapper explicitly tracks two spaces.

### `train_inputs_raw`

Original search-space input before `input_transform`.

### `train_inputs`

Transformed inputs used by the latent GP.

The wrapper disables BoTorch's automatic transformed-input update because the
stored training input is already transformed.  Reapplying an evaluation-mode
`InputPerturbation` could expand the q dimension and make it inconsistent with
training labels.

`set_train_data()` expects raw-space input and applies the training transform
internally.

---

## 20. Inducing points

The ordinal wrapper stores:

```text
inducing_points_raw
inducing_points
```

where the latter is in transformed space.

If inducing points are not supplied, a subset of training inputs is selected.
The number used is

$$
M=\min(M_{\mathrm{requested}},n).
$$

Learning inducing locations can improve the approximation but may move them
away from valid categorical codes in an improperly constructed mixed model.
Mixed-input kernels and transforms must preserve categorical semantics.

---

## 21. Default covariance

The public `OrdinalGPModel` uses an ARD Matérn-5/2 covariance by default:

$$
k(x,x')
=
\sigma_f^2
\left(1+\sqrt5r+rac53r^2\right)e^{-\sqrt5r},
$$

$$
r^2
=
\sum_j
\frac{(x_j-x'_j)^2}{\ell_j^2}.
$$

The legacy internal latent class can default to RBF when no covariance is
provided, but the public wrapper supplies Matérn-5/2.  Documentation should
refer to the public model behavior.

---

## 22. Mixed ordinal inputs

For continuous dimensions `C` and category dimensions `G`, the mixed ordinal
kernel follows the pattern

$$
k(x,x')
=
k_C(x_C,x_C')
+k_G(x_G,x_G')
+k_C'(x_C,x_C')k_G'(x_G,x_G').
$$

Implementation rules:

- categorical dimensions are normalized as indices only, not numerically
  standardized;
- continuous kernels use continuous active dimensions;
- categorical kernels use `CategoricalKernel`;
- `input_transform` must not modify category columns;
- raw inputs may be repeated only to align with q expansion for validation.

The main mixed model is

```text
OrdinalMixedGPModel
```

under the ordinal base package.

---

## 23. Class-count inference

When `num_classes` is omitted, the base implementation infers it from unique
training labels.

Requirements are:

- at least three classes;
- labels are consecutive integers;
- labels start at zero.

For example,

```text
[0, 1, 2]
```

is valid, while

```text
[1, 2, 3]
```

or

```text
[0, 2, 3]
```

must be remapped or supplied with an explicitly supported configuration.

The inferred class count only reflects classes observed in training data.  If a
valid class is absent from the initial sample, explicitly setting class count is
safer.

---

## 24. Conditioning behavior

`condition_on_observations()` for a variational ordinal model cannot perform the
same analytic rank update as an exact Gaussian GP.

The wrapper reconstructs a model with old plus new data, copies learned state,
and can run configurable conditioning optimization steps.

Relevant parameters include:

```text
conditioning_steps
conditioning_lr
conditioning_batch_size
```

This is approximate variational conditioning.  Fantasy dimensions and exact
look-ahead semantics require careful testing.

---

## 25. Calibration for ordinal models

### Classwise calibration

For each class `k`, compare predicted `p_k` with observed frequency.

### Cumulative calibration

For each boundary `j`, calibrate

$$
P(Y\ge j+1\mid x).
$$

This is often more stable and directly tied to ordinal thresholds.

### Ranked Probability Score

For cumulative predicted probabilities `F_k` and observed class `y`,

$$
\operatorname{RPS}
=
\sum_{k=0}^{K-2}
\left[
F_k-\mathbf1(y\le k)
\right]^2.
$$

RPS penalizes errors according to ordinal distance through cumulative events.

### Mean absolute class error

$$
rac1n
\sum_i|\hat y_i-y_i|.
$$

This assumes unit spacing between adjacent classes, but respects order better
than ordinary accuracy.

---

## 26. Identifiability and diagnostics

Inspect:

- ordered cutpoints;
- gap sizes;
- latent mean distribution relative to cutpoints;
- class frequencies;
- inducing-point coverage;
- predictive class calibration;
- whether one class receives nearly zero probability everywhere;
- sensitivity to cutpoint initialization;
- sensitivity to fixed-first versus centered convention.

Very large cutpoint gaps can isolate a class; very small gaps can make a class
rare or unstable.  A missing class in training data makes its cutpoint region
weakly identified.

---

## 27. Heteroscedastic ordinal interpretation

A principled input-dependent scale model is

$$
P(Y\le j\mid x)
=
\sigma\left(
\frac{c_j-f(x)}{s(x)}
\right),
\qquad s(x)>0.
$$

Large `s(x)` creates diffuse class transitions.

This is not the same as:

- adding a noise variance to latent posterior variance;
- adding noise to class-utility variance;
- multiplying an acquisition score by a noise weight.

Current robust ordinal wrappers may use auxiliary noise models for posterior or
acquisition adjustments.  They should be described according to their actual
implementation rather than as a full input-dependent-scale ordered-logit model.

See Chapter 13.

---

## 28. Deep and high-dimensional ordinal variants

Ordinal model families include:

- DeepGP;
- Deep Kernel GP;
- Deep Kernel DeepGP;
- SAAS;
- decomposition or embedding variants;
- heteroscedastic and robust variants.

These change the latent function representation but retain the ordered
likelihood and cutpoint interpretation.

See Chapter 14 for the representation assumptions.

---

## 29. Multi-output ordinal models

Independent ordinal outputs use separate latent functions and cutpoints:

$$
p(\mathbf y\mid x)
=
\prod_{r=1}^{m}
p_r(y_r\mid x).
$$

This supports different:

- class counts;
- utilities;
- cutpoints;
- kernels;
- missingness patterns.

A correlated ordinal multitask model introduces latent covariance

$$
\operatorname{Cov}[f_r(x),f_s(x')]
=
B_{rs}k(x,x').
$$

and task-specific cutpoints.  The current practical wrappers frequently use
ModelList-style independent submodels.  A learned task covariance should not be
interpreted as raw label correlation.

---

## 30. `bochan` source map

| Component | Source |
|---|---|
| Ordered-logit likelihood | `src/bochan/likelihoods/ordinal.py` |
| Continuous and mixed base models | `src/bochan/models/ordinal/base/` |
| Variational fit helper | `src/bochan/fit/ordinal.py` |
| Ordinal posterior transforms | `src/bochan/models/transforms/posterior/ordinal.py` |
| Ordinal objective classes | `src/bochan/acquisition/objective/ordinal.py` |
| Ordinal BO | `src/bochan/acquisition/ordinal/bayesian_optimization/` |
| Ordinal Active Learning | `src/bochan/acquisition/ordinal/active_learning/` |
| Ordinal LSE | `src/bochan/acquisition/ordinal/levelset_estimation/` |
| Ordinal robust models | `src/bochan/models/ordinal/robust/` |
| Ordinal deep models | `src/bochan/models/ordinal/deep/` |
| Ordinal high-dimensional models | `src/bochan/models/ordinal/high_dim/` |
| High-level registration | `src/bochan/api/model_registry.py` |

---

## 31. Model checklist

1. Are labels truly ordered?
2. How many valid classes exist?
3. Are labels encoded consecutively from zero?
4. Ordered logit or another link?
5. Cutpoint location convention?
6. Cutpoint initialization and minimum gap?
7. Kernel and ARD configuration?
8. Inducing-point count and learning?
9. Continuous or mixed inputs?
10. `posterior()` latent or predictive?
11. Class probabilities marginalized or plug-in?
12. Utility values and their spacing?
13. Calibration metric?
14. Conditioning/fantasy support?
15. Missing-class and imbalance handling?
16. Deep, high-dimensional, or heteroscedastic extension?

---

## 32. References

- McCullagh, *Regression Models for Ordinal Data*, 1980.
- Chu and Ghahramani, *Gaussian Processes for Ordinal Regression*, 2005.
- Titsias, *Variational Learning of Inducing Variables in Sparse Gaussian Processes*, 2009.
- Gneiting and Raftery, work on proper scoring rules, including ranked probability scoring concepts.
