# 08. Input Perturbation and Risk-aware Objectives

Standard Bayesian optimization evaluates a nominal design point.  Robust
optimization recognizes that the executed condition, environment, or response
may vary.  This chapter defines input perturbation, distributional objectives,
chance constraints, Value at Risk, Conditional Value at Risk, and their tensor
implementation in `bochan`.

Observation noise and model uncertainty are treated separately in Chapters 01
and 13.  Robust Level-set Estimation is connected in Chapters 05 and 16.

---

## 1. Nominal and executed inputs

Let the selected nominal input be

```math
x\in\mathcal X.
```

The executed input is random:

```math
\widetilde X
=T(x,W),
```

where `W` is an execution perturbation.  The additive case is

```math
\widetilde X=x+W.
```

The resulting response is

```math
Z_x=f(\widetilde X).
```

A nominal optimizer solves

```math
\max_x f(x),
```

whereas a robust optimizer solves

```math
\max_x\rho(Z_x)
```

for a risk functional `rho`.

---

## 2. Sources of uncertainty

Several random mechanisms can coexist.

### Posterior epistemic uncertainty

```math
f\sim p(f\mid\mathcal D).
```

### Observation noise

```math
Y=f(X)+\varepsilon.
```

### Input perturbation

```math
\widetilde X=T(x,W).
```

### Environmental variability

```math
Y=f(x,E)+\varepsilon.
```

### Random class outcome

```math
Y\sim\operatorname{Categorical}(\mathbf p(x)).
```

The robust objective must state over which randomness `rho` is taken.  For
example,

```math
\mathbb E_W[f(x+W)]
```

is different from

```math
\mathbb E_{f\mid\mathcal D,W}[f(x+W)]
```

and from expected future noisy measurement

```math
\mathbb E_{f,W,\varepsilon}[Y].
```

---

## 3. Mean robustness

The expected perturbed response is

```math
R_{\mathrm{mean}}(x)
=
\mathbb E_W[f(T(x,W))].
```

For maximization, choose

```math
x^*
\in
\arg\max_xR_{\mathrm{mean}}(x).
```

This rewards average neighborhood performance.  It does not directly control
bad-tail outcomes.

A Monte Carlo estimator with perturbations `W_1,...,W_nw` is

```math
\widehat R_{\mathrm{mean}}(x)
=
\frac1{n_w}
\sum_{r=1}^{n_w}
f(T(x,W_r)).
```

For posterior samples, the estimator retains both posterior-sample and
perturbation axes until the objective reduces them.

---

## 4. Variance-penalized objective

A mean-variance utility is

```math
R_{\mathrm{MV}}(x)
=
\mathbb E[Z_x]
-
\lambda
\operatorname{Var}(Z_x),
\qquad \lambda\ge0.
```

or, using standard deviation,

```math
R_{\mathrm{MSD}}(x)
=
\mathbb E[Z_x]
-
\lambda\sqrt{\operatorname{Var}(Z_x)}.
```

Mean-variance objectives are easy to compute but do not distinguish symmetric
variance from one-sided downside risk.

If posterior uncertainty and input perturbation are both random, the law of
total variance gives

```math
\operatorname{Var}(Z_x\mid\mathcal D)
=
\mathbb E_W
[\operatorname{Var}_f(f(T(x,W))\mid W,\mathcal D)]
+
\operatorname{Var}_W
[\mathbb E_f(f(T(x,W))\mid W,\mathcal D)].
```

The first term is average model uncertainty; the second is sensitivity to
execution perturbation.

---

## 5. Worst-case robustness

For perturbation support `W in mathcal W`, worst-case objective is

```math
R_{\mathrm{worst}}(x)
=
\inf_{w\in\mathcal W}
f(T(x,w)).
```

A sample approximation is

```math
\widehat R_{\mathrm{worst}}(x)
=
\min_{r=1,\ldots,n_w}
f(T(x,W_r)).
```

Properties:

- conservative;
- sensitive to perturbation support;
- sample minimum becomes more pessimistic as `n_w` grows;
- one extreme draw can dominate;
- gradients can be nonsmooth when the worst perturbation changes.

Worst-case robustness is appropriate when all perturbations in a bounded set
must be tolerated.  It is not a probability-weighted risk measure.

---

## 6. Quantiles and Value at Risk

Let `Z_x` be a utility to maximize with cumulative distribution function

```math
F_x(z)=P(Z_x\le z).
```

Define lower-tail quantile

```math
q_\alpha(x)
=
F_x^{-1}(\alpha)
=
\inf\{z:F_x(z)\ge\alpha\},
\qquad 0<\alpha<1.
```

For maximization, a conservative Value at Risk can be defined as

```math
\operatorname{VaR}_\alpha^{\mathrm{lower}}(Z_x)
=q_\alpha(x),
```

where small `alpha` targets a bad lower tail.  For example, `alpha=0.1`
represents the 10th percentile of utility.

Some financial conventions define `alpha` as a high confidence level and use
`q_{1-alpha}` for reward.  Implementations must state the convention.  The
`bochan` objective convention should be interpreted from its sorting direction,
`maximize` flag, and tail-size calculation rather than from the word VaR alone.

A sample estimator is the corresponding order statistic of

```math
Z_1,\ldots,Z_{n_w}.
```

VaR ignores outcomes beyond the quantile and can change discontinuously when
sample order changes.

---

## 7. Conditional Value at Risk

For lower-tail utility risk, CVaR is the average utility in the worst `alpha`
fraction:

```math
\operatorname{CVaR}_\alpha^{\mathrm{lower}}(Z_x)
=
\mathbb E[
Z_x\mid Z_x\le q_\alpha(x)
]
```

for continuous distributions.

A variational representation is

```math
\operatorname{CVaR}_\alpha^{\mathrm{lower}}(Z)
=
\sup_{\eta}
\left[
\eta-
\frac1\alpha
\mathbb E[(\eta-Z)_+]
\right].
```

For samples, sort utilities ascending and average the worst

```math
k=\max(1,\lceil\alpha n_w\rceil)
```

values.

CVaR uses all tail samples and is usually smoother and more sensitive to severe
outcomes than VaR.

---

## 8. Loss versus utility conventions

For a loss `L` to minimize, upper-tail risk is often used:

```math
\operatorname{VaR}_{1-\alpha}^{\mathrm{upper}}(L),
```

```math
\operatorname{CVaR}_{1-\alpha}^{\mathrm{upper}}(L).
```

For utility `Z` to maximize, lower-tail risk is natural.

Converting loss to utility

```math
Z=-L
```

changes tail direction.  A robust implementation should expose:

- whether larger values are better;
- whether the lower or upper tail is selected;
- whether `alpha` is tail mass or confidence level;
- sorting direction;
- quantile interpolation rule.

---

## 9. Entropic risk

For loss `L`, entropic risk is

```math
\rho_\lambda(L)
=
\frac1\lambda
\log\mathbb E[e^{\lambda L}],
\qquad\lambda>0.
```

For utility maximization, a risk-averse certainty equivalent can be

```math
R_\lambda(Z)
=-\frac1\lambda
\log\mathbb E[e^{-\lambda Z}].
```

Entropic risk is smooth and emphasizes tails exponentially.  It can be
numerically sensitive and should use log-sum-exp stabilization in sample
estimation.

---

## 10. Probability of meeting a target

For utility threshold `h`, reliability is

```math
R_h(x)
=P_W(f(T(x,W))\ge h).
```

A chance-constrained robust design requires

```math
R_h(x)\ge1-\epsilon.
```

A Monte Carlo estimator is

```math
\widehat R_h(x)
=
\frac1{n_w}
\sum_{r=1}^{n_w}
\mathbf1[f(T(x,W_r))\ge h].
```

A smooth approximation can replace the indicator with

```math
\sigma\left(
\frac{f(T(x,W_r))-h}{\tau}
\right).
```

The smoothing temperature `tau` trades bias for differentiability.

---

## 11. Posterior and execution chance constraints

Consider uncertain latent function and input perturbation.

### Posterior-mean execution reliability

```math
P_W(
\mathbb E_f[f(T(x,W))\mid\mathcal D]
\ge h
).
```

### Joint posterior-execution reliability

```math
P_{f,W}(
f(T(x,W))\ge h
\mid\mathcal D
).
```

### Predictive measurement reliability

```math
P_{f,W,\varepsilon}(
Y(T(x,W))\ge h
\mid\mathcal D
).
```

These probabilities answer different questions.  The last includes future
measurement noise and is usually smaller when noise is large.

---

## 12. Common random numbers

During acquisition optimization, perturbation samples can be fixed across
candidate evaluations:

```math
W_1,\ldots,W_{n_w}
\quad\text{fixed during optimization}.
```

This creates a sample-average approximation with a smoother deterministic
objective.  Resampling perturbations every forward pass introduces stochastic
noise into gradients and restart comparisons.

Across BO iterations, perturbations can be resampled to avoid overfitting one
fixed set, while maintaining reproducibility through recorded seeds.

---

## 13. Perturbation models

### 13.1 Additive Gaussian

```math
W\sim\mathcal N(0,\Sigma_W).
```

### 13.2 Uniform tolerance

```math
W_j\sim\operatorname{Uniform}(-a_j,a_j).
```

### 13.3 Multiplicative error

```math
\widetilde X_j
=x_j(1+W_j).
```

### 13.4 Correlated execution error

```math
W\sim\mathcal N(0,\Sigma_W)
```

with off-diagonal covariance.

### 13.5 Empirical perturbation distribution

Use historical deviations

```math
W_r\in\{w^{(1)},\ldots,w^{(N)}\}
```

or bootstrap samples.

### 13.6 Discrete category error

A categorical transition matrix can model execution or labeling errors:

```math
P(\widetilde C=j\mid C=i)=M_{ij}.
```

Adding Gaussian noise to integer category codes is not a valid category
perturbation model.

---

## 14. Boundary handling

Perturbed points may leave the valid domain.  Possible policies include:

### Clipping

```math
\widetilde x_j
\leftarrow
\min(u_j,\max(l_j,\widetilde x_j)).
```

Clipping creates mass at boundaries.

### Reflection

Reflect points back into the interval.  This preserves distance better but
changes the perturbation distribution.

### Rejection and resampling

Sample until the perturbation is feasible.  This produces a truncated
distribution.

### Constraint-aware transformation

Generate perturbations directly on a simplex, manifold, or feasible polytope.

The policy is part of the robust problem and must be documented.

---

## 15. Mixed and compositional inputs

For mixed inputs, perturb continuous dimensions only unless category uncertainty
has an explicit transition model.

For composition vector

```math
x_j\ge0,
\qquad
\sum_jx_j=1,
```

independent additive perturbation breaks the simplex.  Alternatives include:

- logistic-normal perturbation;
- Dirichlet perturbation;
- tangent-space perturbation followed by projection;
- mass-transfer perturbations that preserve the sum.

A generic clipping-and-renormalization rule changes correlations and should be
viewed as a defined perturbation mechanism, not a neutral correction.

---

## 16. Input perturbation tensor expansion

For candidate tensor

```text
X: batch_shape x q x d
```

and `n_w` perturbations per candidate, the transform produces

```text
X_tilde: batch_shape x (q * n_w) x d
```

with ordering typically

```text
[x_1+w_1, ..., x_1+w_nw,
 x_2+w_1, ..., x_2+w_nw,
 ...]
```

A pointwise score has shape

```text
batch_shape x (q * n_w)
```

and is reshaped as

```text
batch_shape x q x n_w
```

before reducing the perturbation axis.

For posterior samples:

```text
sample_shape x batch_shape x (q * n_w) x m
```

may become

```text
sample_shape x batch_shape x q x n_w x m
```

before risk and output reductions.

---

## 17. Order of reductions

Suppose there are posterior samples `s`, perturbations `w`, q candidates, and
outputs `m`.  Nonlinear operations do not generally commute.

### Expected acquisition of robust utility

1. transform output to utility;
2. reduce perturbations to robust utility;
3. compute improvement or acquisition value;
4. average posterior samples.

### Robust average of pointwise acquisition scores

1. compute acquisition-style score at each perturbation;
2. reduce perturbation scores.

These are generally different:

```math
\operatorname{EI}
\left(
\operatorname{CVaR}_W[f]
\right)
\ne
\operatorname{CVaR}_W
\left(
\operatorname{EI}[f]
\right).
```

The current score-objective pattern in several `bochan` Active Learning and LSE
modules reduces already-computed pointwise scores.  It should be interpreted as
robust score aggregation, not automatically as BO on a robust latent function.

---

## 18. Classification robustness

For binary class probability under perturbation,

```math
p_W(x)=P(Y=1\mid T(x,W),\mathcal D).
```

Possible robust objectives include:

### Mean success probability

```math
\mathbb E_W[p_W(x)].
```

### Worst sampled success probability

```math
\min_Wp_W(x).
```

### Lower-tail CVaR of probability

```math
\operatorname{CVaR}_\alpha^{\mathrm{lower}}[p_W(x)].
```

### Probability that success probability exceeds requirement

```math
P_W(p_W(x)\ge\gamma).
```

### Future-label reliability

If execution and label are both random,

```math
P(Y=1\mid x)
=
\mathbb E_W[p_W(x)].
```

The last identity follows from the law of total probability.

---

## 19. Ordinal robustness

Let class probabilities under perturbation be

```math
p_k(x,W).
```

Expected utility under perturbation is

```math
\mathbb E_W
\left[
\sum_ku_kp_k(x,W)
\right].
```

Because expected utility is linear,

```math
\mathbb E_W
\left[
\sum_ku_kp_k
\right]
=
\sum_ku_k\mathbb E_W[p_k].
```

Nonlinear quantities do not commute:

```math
\operatorname{CVaR}_W
\left[
\sum_ku_kp_k
\right]
\ne
\sum_ku_k
\operatorname{CVaR}_W[p_k].
```

Minimum-grade reliability is

```math
P_W
\left(
P(Y\ge g\mid T(x,W))\ge\gamma
\right).
```

This differs from average probability

```math
\mathbb E_W[P(Y\ge g\mid T(x,W))].
```

---

## 20. Multi-output robustness

For vector response

```math
\mathbf f(T(x,W)),
```

possible objectives are:

- component-wise robust objectives;
- robust scalarization;
- distribution of hypervolume contribution;
- probability all constraints are satisfied under perturbation;
- robust Pareto dominance.

Order matters:

```math
\rho_W[s(\mathbf f)]
```

is not generally equal to

```math
s(\rho_W[f_1],\ldots,\rho_W[f_m]).
```

A component-wise CVaR vector can be overly conservative because worst outcomes
for different objectives may occur at different perturbations.

---

## 21. Robust Level-set Estimation

Possible robust sets include:

### Mean set

```math
L_h^{\mathrm{mean}}
=
\{x:\mathbb E_W[f(T(x,W))]\ge h\}.
```

### Chance set

```math
L_{h,\gamma}^{\mathrm{chance}}
=
\{x:P_W(f(T(x,W))\ge h)\ge\gamma\}.
```

### CVaR set

```math
L_h^{\mathrm{CVaR}}
=
\{x:\operatorname{CVaR}_\alpha^{\mathrm{lower}}[f(T(x,W))]\ge h\}.
```

Each set has a different boundary.  Robust LSE must state which one is being
estimated.

---

## 22. Sample-size effects

Risk estimates depend on `n_w`.

### Mean

Monte Carlo standard error decreases approximately as

```math
O(n_w^{-1/2}).
```

### VaR

Quantile estimates can be unstable with small `n_w`, especially in extreme
tails.

### CVaR

The effective number of tail samples is approximately

```math
\alpha n_w.
```

For `alpha=0.05` and `n_w=20`, only one sample defines the empirical tail.  The
result behaves like a minimum rather than a stable CVaR estimate.

Tail-risk optimization therefore often needs substantially more perturbation
samples than mean robustness.

---

## 23. Evaluation

Evaluate nominal and robust performance separately.

### Nominal metrics

- objective at selected `x`;
- posterior prediction at `x`.

### Execution-distribution metrics

- repeated-execution mean;
- standard deviation;
- lower quantiles;
- CVaR;
- target failure probability;
- constraint violation rate.

### Model metrics

- calibration under perturbed inputs;
- coverage of predictive intervals;
- sensitivity to perturbation-model misspecification.

### Sequential metrics

- robust regret;
- experiments to reach reliability target;
- nominal-versus-robust trade-off;
- runtime as `n_w` increases.

---

## 24. `bochan` implementation correspondence

### 24.1 Input transforms

BoTorch input transforms, including perturbation transforms, are attached to
models.  `bochan` wrappers commonly enforce:

- no training-time q expansion;
- evaluation-time `q -> q*n_w` expansion;
- preservation of categorical columns;
- raw and transformed input traceability.

### 24.2 Score objectives

Task-specific score-objective classes occur in regression, binary, multiclass,
and ordinal acquisition modules.  Representative behavior is:

```text
pointwise score with q*n_w points
    -> reshape (..., q, n_w)
    -> mean, VaR, or CVaR reduction
    -> q reduction
```

For example, regression LSE defines

```text
RegressionLevelSetScoreObjective
```

and ordinal LSE defines an ordinal score-objective counterpart.

### 24.3 Shared heteroscedastic transform helpers

```text
src/bochan/models/components/heteroscedastic.py
```

extracts normalization-only transforms for auxiliary noise models so that
`InputPerturbation` does not expand noise-model training data.

### 24.4 High-level configuration

Risk-related arguments used across custom objectives include:

```text
n_w
risk_type
alpha
maximize
weight
sign
```

The exact tail convention is determined by the class implementation.  Tests
should verify sorting direction with a known sample vector.

### 24.5 Shape contract

Chapter 09 gives the complete tensor convention.  Perturbation-aware components
must accept either already aggregated `q` scores or expanded `q*n_w` scores
without silently reducing the wrong axis.

---

## 25. Configuration checklist

Specify:

1. nominal input and executed-input model;
2. perturbed dimensions;
3. perturbation distribution and correlation;
4. boundary handling;
5. categorical and compositional treatment;
6. random variables included in the risk measure;
7. utility or loss convention;
8. risk functional;
9. tail direction and `alpha` meaning;
10. `n_w` and sampling method;
11. common-random-number policy;
12. order of objective, risk, output, q, and posterior-sample reductions;
13. constraint interpretation;
14. evaluation under repeated execution.

---

## 26. References

- Rockafellar and Uryasev, *Optimization of Conditional Value-at-Risk*, 2000.
- Marzat, Walter, and Piet-Lahanier, work on worst-case and robust Bayesian optimization.
- Bogunovic et al., work on Bayesian optimization and level-set estimation under input uncertainty.
- Balandat et al., BoTorch risk-measure and input-perturbation design patterns.
