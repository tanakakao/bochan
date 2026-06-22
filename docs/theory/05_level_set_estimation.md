# 05. Level-set Estimation

Level-set Estimation (LSE) identifies where an unknown function lies above,
below, or near one or more thresholds.  Its objective is a set or boundary, not
the global maximum.

This chapter defines the statistical problem, losses, confidence sets, stopping
rules, and evaluation protocol.  Task-specific acquisition formulas and their
`bochan` implementations are collected in Chapter 16.

---

## 1. Basic definitions

Let

```math
f:\mathcal X\rightarrow\mathbb R
```

be unknown and let `h` be a threshold.

The upper level set is

```math
L_h^+
=
\{x\in\mathcal X:f(x)\ge h\}.
```

The lower level set is

```math
L_h^-
=
\{x\in\mathcal X:f(x)<h\}.
```

The boundary or contour is

```math
B_h
=
\{x\in\mathcal X:f(x)=h\}.
```

The design-space membership indicator is

```math
z_h(x)=\mathbf1[f(x)\ge h].
```

LSE seeks an estimator

```math
\widehat z_{h,t}(x)
```

or estimated sets

```math
\widehat L_{h,t}^+,
\qquad
\widehat L_{h,t}^-,
\qquad
\widehat B_{h,t}
```

using a limited observation budget.

---

## 2. Why Bayesian optimization is not sufficient

Bayesian optimization values high utility.  LSE values accurate membership or
boundary decisions across a region.

Consider a function with one high peak and a long threshold contour.  EI may
sample repeatedly near the peak because that is where improvement is possible.
Those observations can leave most of the contour uncertain.

Typical LSE applications include:

- safe operating region discovery;
- material phase-boundary mapping;
- defect-transition characterization;
- process-window identification;
- reliability region estimation;
- threshold-based medical or environmental classification;
- determining where a response exceeds a specification.

The acquisition must therefore be evaluated by set or boundary error, not only
by the best observed response.

---

## 3. Posterior membership probability

For a Gaussian latent posterior

```math
f(x)\mid\mathcal D_t
\sim
\mathcal N(\mu_t(x),\sigma_t^2(x)),
```

the posterior probability of upper-set membership is

```math
\pi_t(x)
=
P(f(x)\ge h\mid\mathcal D_t)
=
\Phi\left(
\frac{\mu_t(x)-h}{\sigma_t(x)}
\right).
```

A Bayes classifier under symmetric 0-1 loss is

```math
\widehat z_t(x)
=
\mathbf1[\pi_t(x)\ge 1/2].
```

This is equivalent to classifying by

```math
\mu_t(x)\ge h
```

when the posterior is symmetric and `sigma_t(x)>0`.

For asymmetric false-safe and false-unsafe costs, the optimal probability
threshold is not `1/2`.

---

## 4. Confidence-bound classification

Let

```math
l_t(x)
=
\mu_t(x)-\beta_t^{1/2}\sigma_t(x),
```

```math
u_t(x)
=
\mu_t(x)+\beta_t^{1/2}\sigma_t(x).
```

A conservative partition is

```math
H_t
=
\{x:l_t(x)\ge h\},
```

```math
L_t
=
\{x:u_t(x)<h\},
```

```math
U_t
=
\mathcal X\setminus(H_t\cup L_t).
```

Interpretation:

- `H_t`: confidently above threshold;
- `L_t`: confidently below threshold;
- `U_t`: unresolved.

The confidence multiplier can be fixed for practical use or scheduled to obtain
uniform theoretical guarantees.  A larger multiplier reduces premature
classification but increases the unresolved region.

---

## 5. LSE loss functions

The correct acquisition depends on the external loss.

### 5.1 Pointwise misclassification loss

For finite evaluation set `G`,

```math
\mathcal L_{\mathrm{mis}}
=
\frac1{|\mathcal G|}
\sum_{x\in\mathcal G}
\mathbf1[
\widehat z_t(x)\ne z_h(x)
].
```

### 5.2 Weighted classification loss

Let false-safe and false-unsafe costs be

```math
c_{\mathrm{FS}},
\qquad
c_{\mathrm{FU}}.
```

Then

```math
\mathcal L_{\mathrm{weighted}}
=
\frac1{|\mathcal G|}
\sum_{x\in\mathcal G}
\left[
 c_{\mathrm{FS}}
 \mathbf1(\widehat z=1,z=0)
+
 c_{\mathrm{FU}}
 \mathbf1(\widehat z=0,z=1)
\right].
```

Safety applications commonly assign a larger cost to false-safe decisions.

### 5.3 Symmetric-difference measure

For continuous domain with measure `nu`,

```math
\mathcal L_\Delta
=
\nu(
\widehat L_h^+
\triangle
L_h^+
),
```

where `triangle` denotes symmetric difference.

### 5.4 Jaccard loss

```math
J
=
\frac{
\nu(\widehat L_h^+\cap L_h^+)
}{
\nu(\widehat L_h^+\cup L_h^+)
},
```

```math
\mathcal L_J=1-J.
```

Jaccard is useful when the positive region occupies a small fraction of the
domain.

### 5.5 Boundary Hausdorff distance

For estimated and true boundaries,

```math
d_H(\widehat B_h,B_h)
=
\max\left\{
\sup_{x\in\widehat B_h}
\inf_{y\in B_h}\|x-y\|,
\sup_{y\in B_h}
\inf_{x\in\widehat B_h}\|x-y\|
\right\}.
```

It is interpretable in low-dimensional geometry but sensitive to isolated
boundary errors.

### 5.6 Integrated posterior classification risk

Under symmetric loss, posterior pointwise risk is

```math
r_t(x)=\min[\pi_t(x),1-\pi_t(x)].
```

Integrated Bayes risk is

```math
R_t
=
\int_{\mathcal X}
r_t(x)d\nu(x).
```

A principled acquisition values expected reduction in `R_t`.

---

## 6. Region-of-interest weighting

Not every point in the design space has equal importance.  Let `w(x)>=0` be a
scientific or operational weight.

Weighted set loss is

```math
\mathcal L_w
=
\int
w(x)
\mathbf1[
\widehat z_t(x)\ne z_h(x)
]
d\nu(x).
```

Examples:

- emphasize common production settings;
- emphasize conditions near nominal operation;
- emphasize high-cost false-safe regions;
- ignore physically impossible areas;
- prioritize a subset of material compositions.

An unweighted acquisition can waste observations in a large but irrelevant
part of the mathematical domain.

---

## 7. Multiple thresholds

Suppose thresholds are

```math
h_1<h_2<\cdots<h_R.
```

They partition the response into bands:

```math
(-\infty,h_1),
[h_1,h_2),
\ldots,
[h_R,\infty).
```

The acquisition may:

- learn all thresholds equally;
- target one threshold;
- weight thresholds by importance;
- focus on the least-resolved threshold;
- define a multiclass region-labeling problem.

Reducing threshold-wise scores by mean, sum, max, or min produces different
policies.  The reduction is part of the LSE problem definition.

Ordinal models naturally contain multiple latent cutpoint boundaries, but their
likelihood and class uncertainty require additional care; see Chapters 12 and
16.

---

## 8. Excursion sets and reliability regions

A probabilistic excursion set may be defined as

```math
E_{h,\gamma}
=
\left\{
 x:
P(f(x)\ge h\mid\mathcal D)
\ge\gamma
\right\}.
```

Here `h` is a response threshold and `gamma` is a posterior-confidence
threshold.

This differs from the latent level set

```math
L_h^+=\{x:f(x)\ge h\}.
```

A reliability set for future noisy observations is

```math
R_{h,\gamma}
=
\left\{
 x:
P(Y(x)\ge h\mid\mathcal D)
\ge\gamma
\right\}.
```

`R` includes observation noise; `E` may refer to latent function uncertainty.
The scientific question determines which set is correct.

---

## 9. Classification boundaries

For binary classification, a probability level set is

```math
L_\tau
=
\{x:P(Y=1\mid x,\mathcal D)\ge\tau\}.
```

The ordinary class-decision boundary uses `tau=0.5`, but safety or quality
requirements may use another threshold such as `0.9`.

A latent boundary

```math
\{x:f(x)=c\}
```

and probability boundary

```math
\{x:p(x)=\tau\}
```

are equivalent only under a specified monotone link and consistent threshold
conversion.

For multiclass classification, boundaries require a definition:

- equality between two class probabilities;
- target-class probability threshold;
- probability of membership in a class set;
- maximum-class transition surface.

There is no unique multiclass level set without this choice.

---

## 10. Ordinal regions

For ordered classes `0,...,K-1`, useful sets include:

### Latent cutpoint set

```math
\{x:f(x)\ge c_j\}.
```

### Minimum-grade probability set

```math
\left\{
 x:P(Y\ge g\mid x,\mathcal D)\ge\gamma
\right\}.
```

### Expected-utility set

```math
\{x:\mathbb E[U(Y)\mid x,\mathcal D]\ge u_0\}.
```

These sets are different even when they are derived from the same ordinal
model.

---

## 11. Multi-output level sets

Let

```math
\mathbf f(x)
=[f_1(x),\ldots,f_m(x)].
```

### 11.1 Intersection

```math
L_\cap
=
\bigcap_{j=1}^m
\{x:f_j(x)\ge h_j\}.
```

This is a joint feasible region.

### 11.2 Union

```math
L_\cup
=
\bigcup_{j=1}^m
\{x:f_j(x)\ge h_j\}.
```

### 11.3 At-least-r-of-m rule

```math
L_r
=
\left\{
 x:
\sum_{j=1}^m
\mathbf1[f_j(x)\ge h_j]
\ge r
\right\}.
```

### 11.4 Scalarized response set

```math
L_s
=
\{x:s(\mathbf f(x))\ge h\}.
```

Output-wise score averaging does not automatically estimate any of these sets.
The logical region must be defined before acquisition reduction.

---

## 12. Correlated versus independent outputs

If output events are independent,

```math
P(x\in L_\cap)
=
\prod_jP(f_j(x)\ge h_j).
```

With correlated output posterior, the correct probability is multivariate:

```math
P(f_1\ge h_1,\ldots,f_m\ge h_m).
```

The product formula is then generally wrong.

ModelList and current `HybridPosterior` interfaces may imply independent proxy
sampling across outputs.  LSE documentation must state whether joint region
probabilities are exact or factorized approximations.

---

## 13. Sequential LSE policy

A generic one-step LSE policy is

```math
x_{t+1}
\in
\arg\max_x
\mathbb E
\left[
\mathcal L_t-
\mathcal L_{t+1}
\mid x,\mathcal D_t
\right].
```

Direct expected-loss reduction is often expensive because it requires:

1. possible future outcomes at `x`;
2. posterior update for each outcome;
3. recomputation of set loss over the domain.

Practical acquisitions therefore use proxies such as:

- confidence-bound ambiguity;
- Straddle;
- boundary-weighted variance;
- class entropy;
- probability-of-exceedance ambiguity;
- integrated contour-uncertainty approximations;
- joint covariance volume.

Chapter 16 gives exact implemented formulas.

---

## 14. q-batch LSE

A q-batch should reduce set uncertainty at distinct or complementary locations.
The ideal batch value is joint expected loss reduction:

```math
\alpha(X)
=
\mathbb E
\left[
\mathcal L_t-
\mathcal L_{t+1}
\mid X,\mathcal D_t
\right].
```

Practical approximations include:

- joint log-determinant uncertainty;
- sequential greedy selection;
- same-batch repulsion;
- pending and observed-point penalties;
- boundary balancing;
- post-selection diversity filters.

Distance penalties are not substitutes for posterior covariance, but they can
be useful when a custom posterior does not provide exact joint covariance.

---

## 15. Observation noise and replication

For noisy measurements, repeated observations at one point can improve
estimation of the latent mean.  Therefore a hard rule against duplicates may be
incorrect.

Replicates are useful when:

- noise is large relative to latent uncertainty;
- noise variance itself is unknown;
- a safety decision needs high confidence at one operating point;
- measurement replication is cheaper than changing conditions.

Replicates are wasteful when:

- the observation is nearly deterministic;
- the boundary location, rather than local mean precision, is uncertain;
- the model already has adequate local replication.

A batch policy should distinguish exact duplicates intended as replicates from
accidental optimizer duplicates.

---

## 16. Input uncertainty

If the executed condition is

```math
\tilde x=x+\delta,
```

then possible robust level sets include:

### Expected-response set

```math
\left\{
 x:\mathbb E_\delta[f(x+\delta)]\ge h
\right\}.
```

### Chance-constrained set

```math
\left\{
 x:P_\delta(f(x+\delta)\ge h)\ge\gamma
\right\}.
```

### Lower-tail risk set

```math
\left\{
 x:\mathrm{CVaR}_\alpha[f(x+\delta)]\ge h
\right\}.
```

Averaging pointwise LSE scores over perturbations is an acquisition heuristic;
it is not necessarily the acquisition corresponding to one of these robust
sets.  Chapter 08 defines the risk functionals and Chapter 16 explains current
score-level aggregation.

---

## 17. Stopping criteria

### Unresolved-region criterion

Stop when

```math
\nu(U_t)\le\epsilon.
```

### Maximum ambiguity criterion

Stop when

```math
\sup_{x\in\mathcal X}
r_t(x)\le\epsilon.
```

### Integrated risk criterion

Stop when

```math
R_t\le\epsilon.
```

### Boundary-width criterion

Stop when the credible band around the contour is sufficiently narrow.

### Operational criterion

Stop when a connected safe region of required size or an acceptable process
window has been certified.

A generic acquisition-value threshold is less interpretable because acquisition
scales differ across criteria and transformations.

---

## 18. Evaluation protocol

A reliable LSE benchmark should specify:

1. ground-truth function or dense reference measurements;
2. threshold or thresholds;
3. domain measure or evaluation grid;
4. observation-noise model;
5. input-perturbation model if present;
6. initial design;
7. total sample budget;
8. batch size;
9. set estimator and confidence rule;
10. external loss;
11. multiple random seeds.

Recommended plots include:

- set error versus observations;
- false-safe and false-unsafe rates;
- unresolved-region measure;
- Jaccard index;
- boundary Hausdorff distance;
- posterior calibration of membership probabilities;
- candidate locations relative to the true boundary;
- duplicate and optimizer-failure rates.

---

## 19. Relationship to constrained BO

A constraint model may estimate

```math
P(g(x)\le0\mid\mathcal D).
```

Constrained BO uses this estimate to find a high-value feasible candidate.
LSE uses observations to learn the entire constraint boundary or feasible
region.

The same model can support both tasks, but the acquisitions and external losses
are different:

| Task | Primary goal |
|---|---|
| Constrained BO | find high objective within feasible region |
| Feasibility search | find any feasible point |
| Constraint LSE | map feasible/infeasible region |
| Safe BO | improve objective while controlling unsafe evaluations |

---

## 20. `bochan` implementation correspondence

### 20.1 Directory organization

Task-specific LSE acquisitions are stored under:

```text
src/bochan/acquisition/regression/levelset_estimation/
src/bochan/acquisition/binary/levelset_estimation/
src/bochan/acquisition/multiclass/levelset_estimation/
src/bochan/acquisition/ordinal/levelset_estimation/
src/bochan/acquisition/non_gaussian/levelset_estimation/
```

Single-output, multi-output, and heteroscedastic variants are separated into
family-specific modules.

### 20.2 High-level name resolution

```text
src/bochan/api/acquisition_registry.py
```

registers public names such as regression Straddle, joint Straddle, ICU,
boundary variance, probability of exceedance, binary latent Straddle,
multiclass target-probability LSE, and ordinal boundary acquisitions.

### 20.3 Model-space responsibility

The acquisition implementation determines whether it consumes:

- regression latent mean/variance;
- binary latent GP;
- binary probability;
- multiclass target probability;
- ordinal latent posterior and cutpoints;
- ordinal class probabilities;
- heteroscedastic noise model;
- multi-output reductions.

Chapter 16 documents those formulas class by class.

### 20.4 Shared interface concerns

Relevant support code includes:

- task-specific acquisition bases under `src/bochan/acquisition/`;
- posterior transforms under `src/bochan/models/transforms/posterior/`;
- score objectives for `q*n_w` aggregation;
- `X_pending`, observed-point, same-batch, and duplicate penalties;
- high-level construction in `src/bochan/api/`.

---

## 21. New LSE component checklist

Document:

1. target set and threshold scale;
2. latent, predictive, probability, or utility space;
3. external LSE loss;
4. confidence or membership estimator;
5. local versus integrated acquisition;
6. pointwise versus joint q-batch value;
7. output and boundary reductions;
8. observation-noise interpretation;
9. replicate policy;
10. input-perturbation definition;
11. pending and observed-point handling;
12. stopping criterion;
13. tensor shapes;
14. whether the implemented score is exact expected loss reduction or a proxy.

---

## 22. References

- Bryan et al., *Active Learning for Identifying Function Threshold Boundaries*, 2006.
- Gotovos et al., *Active Learning for Level Set Estimation*, 2013.
- Chevalier et al., work on Gaussian-process excursion-set estimation and stepwise uncertainty reduction.
- Rasmussen and Williams, *Gaussian Processes for Machine Learning*, 2006.
