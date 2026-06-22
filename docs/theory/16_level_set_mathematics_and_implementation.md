# 16. Level-set Mathematics and Implementation

Level-set Estimation (LSE) is a sequential design problem whose goal is to
identify a region or boundary, not necessarily to find the global optimum.
This chapter gives the mathematical definitions and maps them to the regression,
binary, multiclass, ordinal, multi-output, heteroscedastic, and batch
implementations in `bochan`.

---

## 1. Problem definition

Let

\[
f:\mathcal X\rightarrow\mathbb R
\]

be unknown and let `h` be a threshold.  The upper and lower level sets are

\[
L_h^+=\{x\in\mathcal X:f(x)\ge h\},
\]

\[
L_h^-=\{x\in\mathcal X:f(x)<h\}.
\]

The boundary or contour is

\[
B_h=\{x\in\mathcal X:f(x)=h\}.
\]

After `t` observations, an estimator partitions the domain into

\[
\widehat L_{h,t}^+,
\qquad
\widehat L_{h,t}^-,
\qquad
U_t,
\]

where `U_t` is an undecided region.

LSE differs from Bayesian optimization:

- BO concentrates on inputs with large objective value;
- LSE values observations that reduce uncertainty about membership in a set;
- a point far below the maximum can be highly valuable if it lies near the
  boundary.

---

## 2. Loss functions for LSE

A complete LSE specification requires a loss, not only an acquisition score.

### 2.1 Pointwise set-classification loss

For a finite evaluation set \(\mathcal G\),

\[
\mathcal L_{\mathrm{mis}}
=
\frac1{|\mathcal G|}
\sum_{x\in\mathcal G}
\mathbf 1\left[
\widehat z(x)\ne z(x)
\right],
\]

where

\[
z(x)=\mathbf 1[f(x)\ge h].
\]

### 2.2 Weighted false-safe and false-unsafe loss

In safety problems, the two errors have different costs:

\[
\mathcal L
=
\lambda_{\mathrm{FS}}
P(\widehat z=1,z=0)
+
\lambda_{\mathrm{FU}}
P(\widehat z=0,z=1).
\]

Usually `lambda_FS` is larger when incorrectly declaring an unsafe condition
safe is critical.

### 2.3 Symmetric-difference measure

For continuous domains,

\[
\mathcal L_{\Delta}
=
\mu\left(
\widehat L_h^+\triangle L_h^+
\right),
\]

where `triangle` is the symmetric difference and `mu` is a reference measure on
the design space.

### 2.4 Boundary loss

In two or low dimensions, a contour metric such as Hausdorff distance may be
used:

\[
d_H(\widehat B_h,B_h)
=
\max\left\{
\sup_{x\in\widehat B_h}\inf_{y\in B_h}\|x-y\|,
\sup_{y\in B_h}\inf_{x\in\widehat B_h}\|x-y\|
\right\}.
\]

An acquisition cannot be judged only by its own score; it should be evaluated
against one or more of these external losses.

---

## 3. GP confidence classification

For a regression GP posterior,

\[
f(x)\mid\mathcal D_t
\sim\mathcal N(\mu_t(x),\sigma_t^2(x)).
\]

Define a confidence interval

\[
C_t(x)
=
[\mu_t(x)-\beta_t^{1/2}\sigma_t(x),
 \mu_t(x)+\beta_t^{1/2}\sigma_t(x)].
\]

A conservative partition is

\[
H_t
=
\{x:\mu_t(x)-\beta_t^{1/2}\sigma_t(x)\ge h\},
\]

\[
L_t
=
\{x:\mu_t(x)+\beta_t^{1/2}\sigma_t(x)< h\},
\]

\[
U_t=\mathcal X\setminus(H_t\cup L_t).
\]

The width of the unresolved interval relative to `h` motivates Straddle-like
criteria.

---

## 4. Regression LSE acquisitions

The main implementations are in

```text
src/bochan/acquisition/regression/levelset_estimation/single_output.py
```

### 4.1 Pointwise Straddle

The implemented score is

\[
\alpha_{\mathrm{straddle}}(x)
=
\beta\sigma(x)-|\mu(x)-h|.
\]

Code:

```text
qRegressionStraddle
```

Interpretation:

- `beta * sigma` rewards uncertainty;
- `-|mu-h|` rewards proximity to the threshold;
- the score is maximized;
- a negative score is valid and means the point is not especially useful.

The implementation parameter `beta` multiplies the standard deviation directly.
It is therefore not identical to a theoretical confidence parameter written as
`sqrt(beta_t)` unless the convention is adjusted.

### 4.2 Joint Straddle

For a q-batch with posterior mean vector \(\boldsymbol\mu\) and covariance
\(\Sigma\), the implementation uses

\[
\alpha_{\mathrm{joint}}
=
-\frac1q\sum_{i=1}^q|\mu_i-h|
+
\beta U(\Sigma),
\]

where `U` can be

\[
U_{\mathrm{trace}}(\Sigma)=\operatorname{tr}(\Sigma),
\]

\[
U_{\log\det}(\Sigma)=\log\det(\Sigma+\epsilon I),
\]

or

\[
U_{\log\det 1p}(\Sigma)
=
\log\det(I+\Sigma+\epsilon I).
\]

Code:

```text
qRegressionJointStraddle
```

A log determinant rewards joint volume and accounts for correlation.  In
contrast, a sum of marginal variances can select redundant nearby points.

### 4.3 ICU-style score

The implemented local contour-uncertainty proxy is

\[
\alpha_{\mathrm{ICU}}(x)
=
\exp\left[
-\frac12
\left(
\frac{\mu(x)-h}{b(x)}
\right)^2
\right]
\sigma(x),
\]

where `b` is either a supplied bandwidth or the posterior standard deviation.

Code:

```text
qRegressionICU
```

This is an ICU-style local weighting, not necessarily the exact globally
integrated expected reduction in contour uncertainty used by every paper under
the name ICU.  The implementation name should therefore be interpreted through
this explicit formula.

### 4.4 Boundary variance

The implemented score is

\[
w_h(x)
=
\exp\left[
-\frac12
\left(
\frac{\mu(x)-h}{\tau}
\right)^2
\right],
\]

\[
\alpha_{\mathrm{BV}}(x)
=
\sigma^2(x)w_h(x).
\]

Code:

```text
qRegressionBoundaryVariance
```

`tau` controls the width of the region treated as near the boundary.

### 4.5 Probability of exceedance

For a Gaussian latent posterior,

\[
P(f(x)\ge h\mid\mathcal D)
=
\Phi\left(
\frac{\mu(x)-h}{\sigma(x)}
\right).
\]

Similarly,

\[
P(f(x)<h\mid\mathcal D)
=
\Phi\left(
\frac{h-\mu(x)}{\sigma(x)}
\right).
\]

For an interval `[l,u]`,

\[
P(l\le f(x)\le u)
=
\Phi\left(\frac{u-\mu}{\sigma}\right)
-
\Phi\left(\frac{l-\mu}{\sigma}\right).
\]

Code:

```text
qRegressionProbabilityOfExceedance
```

Probability of exceedance estimates membership; by itself it tends to favor
points already confidently above the threshold.  It is not always an efficient
boundary-learning acquisition unless combined with an ambiguity term.

---

## 5. Binary classification LSE

For binary classification, two spaces are available.

### 5.1 Latent-space boundary

Let

\[
f(x)\mid\mathcal D
\sim\mathcal N(\mu_f(x),\sigma_f^2(x)).
\]

For a symmetric link, the class boundary at probability `0.5` corresponds to

\[
f(x)=0.
\]

A latent Straddle score is

\[
\alpha(x)
=
\beta\sigma_f(x)-|\mu_f(x)-h_f|.
\]

The binary implementation uses a smoothed absolute value:

\[
\alpha(x)
=
\beta\sigma_f(x)
-
\sqrt{(\mu_f(x)-h_f)^2+\epsilon}.
\]

Code:

```text
qBinaryLatentStraddleAcquisition
```

in

```text
src/bochan/acquisition/binary/levelset_estimation/single_output.py
```

This acquisition explicitly asks the model base class for a latent
distribution.  It should not use `SimpleBernoulliPosterior.variance` as a latent
variance.

### 5.2 Probability-space boundary

For a target probability `h_p`, define

\[
B_{h_p}=\{x:p(Y=1\mid x,\mathcal D)=h_p\}.
\]

Typical ambiguity weights are

\[
p(1-p)
\]

or Bernoulli entropy

\[
H(p)=-p\log p-(1-p)\log(1-p).
\]

At `h_p=0.5`, these peak at the ordinary decision boundary.  For `h_p != 0.5`,
a boundary-specific weight should be centered on `h_p`, not automatically on
`0.5`.

### 5.3 Joint latent Straddle

For batch mean vector and covariance,

\[
\alpha(X)
=
\beta U(\Sigma_f)-D(\boldsymbol\mu_f,h_f),
\]

where the implementation supports joint uncertainty based on log determinant
or square-root trace, and boundary distance based on mean absolute, root mean
square, or maximum absolute distance.

Code:

```text
qBinaryJointLatentStraddleAcquisition
```

The implementation also applies same-batch, pending-point, and observed-point
repulsion.

---

## 6. Multiclass LSE

There is no unique scalar boundary for unordered multiclass classification.
`bochan` defines a target-class probability function

\[
p_T(x)
=
\operatorname{reduce}_{k\in T}p(Y=k\mid x),
\]

where `T` is one class or a selected set of classes.

The target level set is

\[
L_{h,T}
=
\{x:p_T(x)\ge h\}.
\]

### 6.1 Target-probability Straddle

The implementation uses

\[
\alpha(x)
=
\beta u(x)-|p_T(x)-h|.
\]

Code:

```text
qMulticlassLatentStraddleAcquisition
```

Despite the historical name `LatentStraddle`, the current formula is centered
on target-class probability.  Its `uncertainty_mode` controls `u(x)`:

#### Bernoulli ambiguity

Treat target membership as a binary event:

\[
u_{\mathrm{Bern}}(x)
=
\sqrt{p_T(x)[1-p_T(x)]}.
\]

#### Posterior uncertainty

From probability samples \(p_T^{(s)}(x)\),

\[
u_{\mathrm{post}}(x)
=
\operatorname{Std}_s[p_T^{(s)}(x)].
\]

#### Combined

\[
u_{\mathrm{combined}}(x)
=
\sqrt{
\operatorname{Var}_s[p_T^{(s)}(x)]
+p_T(x)[1-p_T(x)]
}.
\]

The combined quantity mixes epistemic probability uncertainty with categorical
observation ambiguity.  It is useful as a heuristic score but should not be
called purely epistemic.

Implementation:

```text
src/bochan/acquisition/multiclass/levelset_estimation/single_output.py
```

### 6.2 Class reduction

When several target classes are supplied, `class_reduction` determines whether
the target probability is a mean, sum, max, or another supported reduction.
The mathematical event changes with the reduction.  In particular,

\[
\sum_{k\in T}p_k
\]

is the probability of membership in the union of mutually exclusive target
classes, while

\[
\frac1{|T|}\sum_{k\in T}p_k
\]

is only a scaled score.  The distinction matters when the threshold is given a
probabilistic interpretation.

---

## 7. Ordinal LSE

Ordinal models have `K-1` natural boundaries at cutpoints

\[
c_0,\ldots,c_{K-2}.
\]

### 7.1 Latent boundary

Boundary `j` is

\[
B_j=\{x:f(x)=c_j\}.
\]

A latent score can use

\[
\alpha_j(x)
=
\beta\sigma_f(x)-|\mu_f(x)-c_j|.
\]

### 7.2 Cumulative boundary probability

From class probabilities,

\[
q_j(x)
=P(Y\ge j+1\mid x)
=
\sum_{k=j+1}^{K-1}p_k(x).
\]

The implementation computes cumulative upper probabilities and uses the
boundary ambiguity

\[
A_j(x)=4q_j(x)[1-q_j(x)].
\]

This is one at the boundary-probability midpoint and zero at confident extremes.

Code helpers:

```text
ordinal_cumulative_ge_probs_from_class_probs
ordinal_boundary_uncertainty
```

in

```text
src/bochan/acquisition/ordinal/levelset_estimation/single_output.py
```

### 7.3 Selecting and reducing boundaries

`target_boundary_idx=j` selects cutpoint `c_j`.  Without a specific target,
boundary-wise scores can be reduced by

\[
\operatorname{mean}_j,
\quad
\sum_j,
\quad
\max_j,
\quad
\min_j.
\]

Interpretation:

- `max`: find whichever boundary is currently most informative;
- `mean`: balance all transitions;
- `sum`: similar ranking to mean when boundary count is fixed;
- `min`: seek a point useful for every boundary, often overly conservative.

### 7.4 Grade-region estimation

A practical ordinal set is

\[
L_g=\{x:P(Y\ge g\mid x)\ge\gamma\}.
\]

This has two thresholds:

- grade threshold `g`;
- probability confidence threshold `gamma`.

It is different from the latent cutpoint set

\[
\{x:f(x)\ge c_{g-1}\}.
\]

The probability set accounts for posterior and likelihood uncertainty.

---

## 8. Multi-output LSE

For outputs

\[
f_j(x),\qquad j=1,\ldots,m,
\]

with thresholds \(h_j\), define output-wise membership scores or boundary
acquisitions \(a_j(x)\).

### 8.1 Intersection set

A joint feasible region is

\[
L_{\cap}
=
\bigcap_{j=1}^{m}
\{x:f_j(x)\ge h_j\}.
\]

Under independent output posteriors, the membership probability is

\[
P(x\in L_{\cap})
=
\prod_{j=1}^{m}P(f_j(x)\ge h_j).
\]

With correlated outputs, this product is generally incorrect; a joint
multivariate probability is required.

### 8.2 Union set

\[
L_{\cup}
=
\bigcup_{j=1}^{m}
\{x:f_j(x)\ge h_j\}.
\]

Under independence,

\[
P(x\in L_{\cup})
=
1-
\prod_j[1-P(f_j(x)\ge h_j)].
\]

### 8.3 Score reductions

The implementation commonly supports

\[
\operatorname{mean}_j a_j,
\quad
\sum_j a_j,
\quad
\max_j a_j,
\quad
\min_j a_j.
\]

These are acquisition-score reductions, not necessarily probabilities of an
intersection or union.  A product-of-feasibility reduction has a distinct
probabilistic interpretation.

### 8.4 Heterogeneous outputs

For regression, binary, multiclass, and ordinal outputs, convert each output to
a meaningful boundary event first:

\[
z_j(x)=
\begin{cases}
\mathbf 1[f_j(x)\ge h_j], & \text{regression},\\
\mathbf 1[p_j(x)\ge h_j], & \text{classification},\\
\mathbf 1[P(Y_j\ge g_j\mid x)\ge\gamma_j], & \text{ordinal}.
\end{cases}
\]

Only after this definition should output scores be combined.

---

## 9. Heteroscedastic LSE

Suppose

\[
y(x)=f(x)+\varepsilon(x),
\qquad
\varepsilon(x)\sim\mathcal N(0,\sigma_n^2(x)).
\]

There are two different level sets.

### 9.1 Latent-process level set

\[
L_h^{f}=\{x:f(x)\ge h\}.
\]

Use the posterior of `f` and exclude observation noise from the boundary
uncertainty.  This estimates the underlying mean process.

### 9.2 Future-observation reliability set

For a required success probability `gamma`,

\[
L_{h,\gamma}^{Y}
=
\left\{
 x:
 P(Y(x)\ge h\mid\mathcal D)\ge\gamma
\right\}.
\]

With Gaussian predictive mean and total variance,

\[
P(Y\ge h)
=
\Phi\left(
\frac{\mu_f-h}
{\sqrt{\sigma_f^2+\sigma_n^2}}
\right).
\]

This set shrinks in high-noise regions and is often closer to an engineering
reliability requirement.

A noise penalty applied to an ordinary latent LSE score is a heuristic and is
not automatically equivalent to estimating `L^Y`.

---

## 10. Robust LSE under input perturbation

Let the executed input be

\[
\widetilde x=x+\delta,
\qquad
\delta\sim p(\delta).
\]

Possible robust sets include:

### Mean-response set

\[
L_h^{\mathrm{mean}}
=
\{x:\mathbb E_\delta[f(x+\delta)]\ge h\}.
\]

### Chance-constrained set

\[
L_{h,\gamma}^{\mathrm{chance}}
=
\{x:P_\delta(f(x+\delta)\ge h)\ge\gamma\}.
\]

### Worst-tail or CVaR set

\[
L_h^{\mathrm{CVaR}}
=
\{x:\operatorname{CVaR}_\alpha[f(x+\delta)]\ge h\}.
\]

The current objective pattern expands each nominal point into `n_w` perturbed
points and reduces the corresponding scores.  A mean of pointwise Straddle
scores,

\[
\frac1{n_w}\sum_r
\alpha_{\mathrm{straddle}}(x+\delta_r),
\]

is not identical to applying Straddle to the mean perturbed response.  The
chosen order of operations must be documented.

---

## 11. Batch diversity and repulsion

Pointwise acquisition values do not guarantee a diverse q-batch.  `bochan` LSE
implementations support several penalties.

### 11.1 Same-batch soft repulsion

For transformed candidates \(z_i\), a typical penalty is

\[
P_{\mathrm{batch}}
=
\lambda_b
\sum_{i\ne j}
\exp(-\eta_b\|z_i-z_j\|^2).
\]

### 11.2 Pending or observed-point repulsion

For reference set `R`,

\[
P_R(x)
=
\lambda_R
\exp[-\eta_R d(x,R)],
\]

where

\[
d(x,R)=\min_{r\in R}\|z(x)-z(r)\|.
\]

### 11.3 Hard duplicate penalty

\[
P_{\mathrm{dup}}
=
M\mathbf 1[d(x,R)\le\epsilon].
\]

### 11.4 Distance space

Distances should be computed in a representation consistent with the model:

- normalized continuous space for ordinary models;
- transformed continuous plus categorical representation for mixed models;
- projected space for PCA/REMBO wrappers only if the model uses it for
  similarity;
- not an accidentally expanded `q*n_w` layout without grouping perturbations.

A repulsion penalty changes the acquisition objective and may prevent useful
replicate experiments.  It should be disabled or modified when replication is
scientifically valuable.

---

## 12. Shape contract

For a candidate tensor

```text
X: batch_shape x q x d
```

a pointwise score should normally be

```text
batch_shape x q
```

and the final acquisition value should be

```text
batch_shape
```

For input perturbation,

```text
batch_shape x q x d
    -> batch_shape x (q * n_w) x d
    -> batch_shape x (q * n_w) score
    -> batch_shape x q robust score
    -> batch_shape acquisition value
```

For multiple outputs, intermediate score shapes may be

```text
batch_shape x q x m
```

before output reduction.

DeepGPs and Monte Carlo probability paths may add leading sample dimensions.
The implementation reduces those sample-like axes while preserving t-batch and
q axes.  Silent reduction over a true model-batch or task axis would change the
mathematics; shape handling must therefore be tested with explicit examples.

---

## 13. Implementation inventory

| Family | Main source |
|---|---|
| Regression single-output LSE | `src/bochan/acquisition/regression/levelset_estimation/single_output.py` |
| Regression multi-output LSE | `src/bochan/acquisition/regression/levelset_estimation/multi_output.py` |
| Regression heteroscedastic LSE | corresponding hetero modules under `regression/levelset_estimation/` |
| Binary LSE | `src/bochan/acquisition/binary/levelset_estimation/` |
| Multiclass LSE | `src/bochan/acquisition/multiclass/levelset_estimation/` |
| Ordinal LSE | `src/bochan/acquisition/ordinal/levelset_estimation/` |
| Non-Gaussian regression LSE | `src/bochan/acquisition/non_gaussian/levelset_estimation/` |
| Feasibility wrappers | `src/bochan/acquisition/feasible/` |
| Risk and perturbation objectives | task-specific score objectives and `docs/theory/08_input_perturbation_and_risk.md` |

---

## 14. Acquisition naming and interpretation

Names such as `Straddle`, `ICU`, `BoundaryVariance`, or `JointStraddle` describe
families of ideas.  The exact implementation is defined by the formula in code.
For each public acquisition, documentation should state:

1. model space: latent, response, probability, or utility;
2. threshold scale;
3. uncertainty term;
4. pointwise or joint q-batch formulation;
5. output and class reductions;
6. pending/observed/duplicate penalties;
7. input-perturbation order of operations;
8. whether the result is a published criterion or an implementation-specific
   proxy.

This prevents two acquisitions with similar names but different probability
spaces from being treated as equivalent.

---

## 15. Recommended evaluation protocol

### Synthetic problems

- Branin or Gaussian mixtures in two dimensions for contour visualization;
- Hartmann6 for higher-dimensional set classification;
- several thresholds corresponding to 50th, 80th, and 95th function-value
  quantiles;
- homoscedastic, heteroscedastic, and input-perturbed variants.

### Metrics

- set misclassification rate;
- Jaccard index;
- false-safe and false-unsafe rates;
- symmetric-difference measure;
- Hausdorff distance in low dimension;
- number of observations required to reach a target loss;
- batch duplicate rate and acquisition optimization failure rate.

### Baselines

- random or Sobol sampling;
- maximum posterior variance;
- predictive entropy for classification;
- pointwise Straddle;
- joint Straddle;
- probability-of-exceedance sampling.

All methods should receive identical initial points, noise draws, candidate
budgets, and evaluation grids.

---

## 16. References

- Bryan et al., *Active Learning for Identifying Function Threshold Boundaries*, 2006.
- Gotovos et al., *Active Learning for Level Set Estimation*, 2013.
- Bogunovic et al., work on level-set estimation under input uncertainty.
- Rasmussen and Williams, *Gaussian Processes for Machine Learning*, 2006.
