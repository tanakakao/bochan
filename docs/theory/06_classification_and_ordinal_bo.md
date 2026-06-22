# 06. Bayesian Optimization with Classification and Ordinal Outputs

Binary, multiclass, and ordinal models do not directly return a continuous
physical response in the same sense as Gaussian regression.  Bayesian
optimization therefore requires an explicit decision transformation from class
probabilities or latent values to utility.

This chapter focuses on that decision layer.  Model likelihoods and posterior
contracts are treated in Chapters 11 and 12.

---

## 1. Why labels are not regression targets

Suppose an observed label is

```math
y\in\{0,1,\ldots,K-1\}.
```

Treating the integer code as an ordinary continuous response imposes assumptions
that may be false:

- distance between classes is numerical;
- class `2` is twice class `1`;
- Gaussian residuals are meaningful;
- predictions outside the valid class range are acceptable;
- unordered classes have a natural order.

A classification or ordinal likelihood instead estimates class probabilities

```math
p_k(x)=P(Y=k\mid x,\mathcal D).
```

BO then optimizes a user-defined functional of the probability vector.

---

## 2. Decision-space transformation

Let the predictive class distribution be

```math
\mathbf p(x)
=[p_0(x),\ldots,p_{K-1}(x)].
```

A scalar decision objective is

```math
u(x)=T(\mathbf p(x)).
```

A multi-objective decision vector is

```math
\mathbf u(x)=
[T_1(\mathbf p(x)),\ldots,T_m(\mathbf p(x))].
```

The transformation `T` defines what is being optimized.  The classifier only
provides probabilities.

---

## 3. Binary probability objective

For binary outcome `Y in {0,1}`, let

```math
p(x)=P(Y=1\mid x,\mathcal D).
```

To maximize probability of class `1`, use

```math
u(x)=p(x).
```

To maximize probability of class `0`, use

```math
u(x)=1-p(x).
```

This is appropriate when one class is unambiguously preferred and all successful
outcomes have equal value.

### Probability threshold

A requirement such as

```math
p(x)\ge0.95
```

is a probabilistic constraint or level set.  Maximizing `p(x)` is not identical
to finding all points that satisfy the threshold.

---

## 4. Binary expected utility

Assign utilities

```math
u_0,\quad u_1.
```

The expected utility is

```math
U(x)
=u_0[1-p(x)]+u_1p(x).
```

This is affine in `p(x)`:

```math
U(x)=u_0+(u_1-u_0)p(x).
```

If `u_1>u_0`, maximizing expected utility gives the same ranking as maximizing
`p(x)`, but the utility scale matters for:

- `best_f`;
- EI magnitude;
- multi-objective reference points;
- weighted scalarization;
- economic interpretation.

When utility depends on process cost or input `x`, use

```math
U(x)
=u_0(x)[1-p(x)]+u_1(x)p(x).
```

The ranking is then not determined by probability alone.

---

## 5. Multiclass target probability

For target class `k*`, define

```math
U_{k^*}(x)=p_{k^*}(x).
```

For a set of acceptable classes `A`, the probability of acceptance is

```math
U_A(x)
=P(Y\in A\mid x)
=
\sum_{k\in A}p_k(x).
```

Because classes are mutually exclusive, the sum has a direct probability
interpretation.  A mean over selected class probabilities differs by the factor
`1/|A|` and changes threshold meaning.

---

## 6. Multiclass expected utility

Assign class utilities

```math
\mathbf u=[u_0,\ldots,u_{K-1}].
```

Then

```math
U(x)
=
\sum_{k=0}^{K-1}u_kp_k(x).
```

Examples:

- failure modes with different repair costs;
- product categories with different revenue;
- manufacturing states with different downstream yield;
- decisions where some classes are acceptable and others are not.

Expected utility is a decision model layered on top of classification.  It does
not make an unordered multiclass likelihood ordinal.

---

## 7. Ordinal expected utility

For ordered classes

```math
0<1<\cdots<K-1,
```

class utility may respect the order:

```math
u_0\le u_1\le\cdots\le u_{K-1}.
```

The expected utility is

```math
U(x)
=
\sum_{k=0}^{K-1}u_kp_k(x).
```

### Equal-spacing utility

```math
u_k=k
```

assumes each adjacent grade improvement has equal value.

### Normalized utility

```math
u_k=\frac{k}{K-1}
```

maps utility to `[0,1]` but retains equal spacing.

### Domain-specific utility

```math
\mathbf u=[0,0.1,0.7,1.0]
```

can represent a large value jump between grades 1 and 2.

The likelihood encodes order statistically; the utility encodes preference
operationally.  They are separate choices.

---

## 8. Minimum-grade probability

For required grade `g`, define

```math
P(Y\ge g\mid x)
=
\sum_{k=g}^{K-1}p_k(x).
```

This can be used as:

- an objective to maximize;
- a feasibility probability;
- a Level-set Estimation target;
- a reliability metric.

It is often more interpretable than expected class index when the engineering
requirement is explicitly grade based.

---

## 9. Latent-score optimization

A GP classifier or ordinal model contains latent score `f(x)`.  One can define

```math
u(x)=f(x)
```

or optimize a posterior functional of `f`.

This can be useful when:

- the latent ordering itself is meaningful;
- the probability link is monotone and only ranking matters;
- boundary exploration is the goal;
- numerical probability saturation is undesirable.

However, latent-score optimization has limitations:

- latent scale is model dependent;
- cutpoint or link calibration affects interpretation;
- a latent improvement has no direct probability or economic meaning;
- `best_f` cannot be copied from observed class labels.

Probability or utility space is usually preferable for user-facing BO.

---

## 10. Improvement in probability or utility space

Let transformed objective be

```math
U(x)=T(Y(x)).
```

Improvement is

```math
I(x)
=
\max[U(x)-U_{\mathrm{best}},0].
```

Expected Improvement is

```math
\operatorname{EI}_U(x)
=
\mathbb E[I(x)\mid\mathcal D].
```

The expectation must account for uncertainty in transformed utility.  In
general,

```math
\operatorname{EI}(\mathbb E[U])
\ne
\mathbb E[\operatorname{I}(U)].
```

A deterministic EI formula applied only to posterior mean probability ignores
uncertainty in the probability function.

### `best_f` scale

For binary probability objective:

```math
U_{\mathrm{best}}\in[0,1].
```

For ordinal expected utility:

```math
U_{\mathrm{best}}
\in[\min u_k,\max u_k].
```

It must not be an observed class integer unless that integer is exactly the
chosen utility scale.

---

## 11. UCB in transformed spaces

A generic transformed UCB is

```math
\operatorname{UCB}_U(x)
=
\mu_U(x)+\lambda\sigma_U(x).
```

Defining `sigma_U` requires care.

### Probability posterior variance

If posterior samples produce probabilities `p^{(s)}`, then

```math
\sigma_p^2
=
\operatorname{Var}_s[p^{(s)}].
```

### Bernoulli observation variance

```math
p(1-p)
```

is variance of the next binary label, not uncertainty in the probability
function.

### Utility-distribution variance

Given fixed class probabilities,

```math
\operatorname{Var}(u_Y\mid x)
=
\sum_kp_k(u_k-U)^2.
```

This is uncertainty in realized class utility, not necessarily epistemic
uncertainty in expected utility.

A class-specific UCB implementation must state which variance it uses.

---

## 12. Probability of Improvement

For transformed utility threshold `tau`,

```math
\operatorname{PI}_U(x)
=
P(U(x)\ge\tau\mid\mathcal D).
```

For binary probability objective, this asks whether the uncertain probability
function exceeds a probability target.  It is not the same as

```math
P(Y=1\mid x)=p(x).
```

The first probability is over posterior uncertainty in `p(x)`; the second is
over the future class label conditional on the model.

---

## 13. Classification as a constraint

Suppose the objective is continuous response `f(x)` and binary classifier
predicts feasibility.

A probability-weighted acquisition is

```math
\alpha_c(x)
=
\alpha_f(x)
P(Y_{\mathrm{feasible}}=1\mid x).
```

For multiple independent constraints,

```math
P(\mathrm{all\ feasible}\mid x)
=
\prod_jp_j(x).
```

If constraint outputs are correlated, the product can be inaccurate.  Joint
posterior samples or a correlated model are needed for the joint event.

### Hard probability threshold

A chance constraint is

```math
P(Y_{\mathrm{feasible}}=1\mid x)
\ge\gamma.
```

This defines a feasible decision region.  Multiplying acquisition by
probability does not strictly enforce the threshold.

---

## 14. Ordinal constraint

A minimum-grade constraint is

```math
P(Y\ge g\mid x)
\ge\gamma.
```

Expected utility constraint is

```math
\mathbb E[u_Y\mid x]
\ge u_0.
```

These are different.  A distribution with small probability of catastrophic
low grade can have acceptable expected utility but fail the minimum-grade
reliability requirement.

---

## 15. Calibration and BO

Optimization exploits errors.  If a probability model is overconfident, BO can
seek regions where predicted probability is spuriously high.

Evaluate:

- reliability diagrams;
- Brier score;
- log loss;
- expected calibration error;
- class-specific calibration;
- ordinal cumulative-probability calibration;
- out-of-distribution confidence.

Calibration estimated on random validation data may not hold at adaptively
selected BO candidates.  Sequential monitoring is important.

Temperature scaling or other calibration methods must be fitted without leaking
future BO outcomes.

---

## 16. Risk-sensitive class utility

The full class-utility distribution is

```math
P(U=u_k\mid x)=p_k(x).
```

Possible risk summaries include:

### Mean

```math
\mathbb E[U]=\sum_kp_ku_k.
```

### Lower quantile

```math
\operatorname{VaR}_\alpha(U).
```

### Lower-tail CVaR

```math
\operatorname{CVaR}_\alpha(U)
=
\mathbb E[U\mid U\text{ is in the lower tail}].
```

### Probability of unacceptable utility

```math
P(U<u_{\min}).
```

For discrete class distributions, VaR is discontinuous in probabilities when
the quantile crosses a class boundary.  CVaR and chance constraints may provide
more stable decision criteria.

---

## 17. Input perturbation

For uncertain execution

```math
\tilde x=x+\delta,
```

possible robust class objectives are:

```math
\mathbb E_\delta[p(Y=1\mid x+\delta)],
```

```math
P_\delta(p(Y=1\mid x+\delta)\ge\gamma),
```

```math
\operatorname{CVaR}_\alpha
[U(x+\delta)].
```

The order of operations matters:

```math
U\left(\mathbb E_\delta[\mathbf p]\right)
```

may equal

```math
\mathbb E_\delta[U(\mathbf p)]
```

for linear expected utility, but nonlinear risk measures or target-distance
transforms do not commute with expectation.

---

## 18. Multi-objective discrete outputs

Several transformed outputs can define a Pareto problem:

```math
\mathbf U(x)
=[U_1(x),\ldots,U_m(x)].
```

Examples:

- continuous strength and binary success probability;
- yield and ordinal quality utility;
- probabilities of several desirable classes;
- expected utility and failure probability.

Before EHVI or NEHVI:

1. transform all directions to maximization;
2. choose objective units and scaling;
3. define a reference point in transformed space;
4. decide whether probabilities are objectives or constraints;
5. verify whether the posterior samples preserve cross-output dependence.

Chapter 07 treats Pareto theory and Chapter 15 treats heterogeneous output
wrappers.

---

## 19. Hybrid model lists

A hybrid experiment may use separate submodels:

```text
regression property model
binary feasibility model
ordinal quality model
multiclass failure-mode model
```

The combined decision layer converts each output to one scalar channel, for
example:

```math
[t_{\mathrm{property}},
 p_{\mathrm{feasible}},
 E[u_{\mathrm{grade}}],
 p_{\mathrm{acceptable\ failure\ class}}].
```

This vector can be scalarized, optimized by hypervolume, or split into objectives
and constraints.

The current `HybridMultiOutputModel` provides a common objective-space posterior
interface but does not automatically create cross-output covariance; see
Chapter 15.

---

## 20. `bochan` implementation correspondence

### 20.1 Binary BO

```text
src/bochan/acquisition/binary/bayesian_optimization/
```

Registered classes include:

- `qBinaryProbabilityOfFeasibility`;
- `qBinaryExpectedImprovement`;
- `qBinaryProbabilityOfImprovement`;
- `qBinaryUpperConfidenceBound`;
- multi-output and heteroscedastic variants.

The base binary model exposes probability through `posterior(X)` and latent GP
through `latent_posterior(X)`.

### 20.2 Multiclass BO

```text
src/bochan/acquisition/multiclass/bayesian_optimization/
```

The acquisition base selects target classes or class reductions from a
probability posterior.  Utility and target-class transformations must preserve
the class axis until reduction.

### 20.3 Ordinal BO

```text
src/bochan/acquisition/ordinal/bayesian_optimization/
```

Registered classes include:

- `qOrdinalExpectedImprovement`;
- `qOrdinalProbabilityOfImprovement`;
- `qOrdinalUpperConfidenceBound`;
- `qOrdinalProbabilityOfFeasibility`;
- multi-output and heteroscedastic utility acquisitions.

Ordinal class probabilities are obtained from the latent posterior and
`OrdinalLogitLikelihood`; the base ordinal `posterior()` itself is latent.

### 20.4 Objective implementations

```text
src/bochan/acquisition/objective/ordinal.py
src/bochan/acquisition/objective/hybrid.py
src/bochan/models/transforms/posterior/classification.py
src/bochan/models/transforms/posterior/ordinal.py
```

### 20.5 Hybrid wrapper

```text
src/bochan/models/hybrid/multi_output.py
src/bochan/models/hybrid/specs.py
src/bochan/models/hybrid/posterior.py
```

`OutputSpec` records task type, sign, weight, target class, utility values, and
optional target-distance transformation.

---

## 21. Configuration checklist

For every discrete-output BO channel, specify:

1. class semantics;
2. target class or acceptable class set;
3. probability, latent, or utility objective;
4. utility values and units;
5. maximization direction;
6. `best_f` scale;
7. uncertainty definition used by UCB or EI;
8. calibration method;
9. objective versus constraint role;
10. risk treatment;
11. input-perturbation aggregation;
12. multi-output dependence assumption;
13. acquisition class and posterior accessor.

---

## 22. References

- Rasmussen and Williams, *Gaussian Processes for Machine Learning*, 2006.
- Chu and Ghahramani, *Gaussian Processes for Ordinal Regression*, 2005.
- Berger, *Statistical Decision Theory and Bayesian Analysis*.
- Balandat et al., *BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization*, 2020.
