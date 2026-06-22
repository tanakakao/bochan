# 03. Acquisition Functions for Bayesian Optimization

An acquisition function converts a posterior distribution into the value of
collecting data at one or more candidate points.  It is the decision rule of
Bayesian optimization.

This chapter focuses on optimization acquisitions.  Active Learning is treated
in Chapter 04, Level-set Estimation in Chapters 05 and 16, and multi-objective
acquisitions in Chapter 07.

---

## 1. General definition

For posterior state `D_t` and candidate batch

\[
X=[x_1,\ldots,x_q],
\]

an acquisition function is

\[
\alpha_t(X)
=
\mathbb E
\left[
V(X,F;\mathcal D_t)
\mid\mathcal D_t
\right],
\]

where `V` is a decision-specific value function and the expectation is over
posterior uncertainty.

The optimizer solves

\[
X_{t+1}
\in
\arg\max_{X\in\mathcal X^q}
\alpha_t(X).
\]

In BoTorch tensor notation:

```text
X:         batch_shape x q x d
acq_value: batch_shape
```

A valid acquisition for `optimize_acqf` should remove sample, output, and q
axes according to its definition and return only the t-batch shape.

---

## 2. Analytic versus Monte Carlo acquisitions

### 2.1 Analytic acquisitions

When the posterior and objective are simple, an acquisition can be evaluated in
closed form.  Analytic EI for one Gaussian output and `q=1` is the standard
example.

Advantages:

- low Monte Carlo noise;
- fast evaluation;
- stable gradients.

Limitations:

- commonly restricted to `q=1`;
- difficult with nonlinear objectives;
- difficult with non-Gaussian, multi-output, or constrained posteriors.

### 2.2 Monte Carlo acquisitions

Draw reparameterized samples

\[
f^{(s)}(X)
\sim p(f(X)\mid\mathcal D_t),
\qquad s=1,\ldots,S,
\]

apply an objective

\[
u^{(s)}(X)=T(f^{(s)}(X),X),
\]

and estimate

\[
\alpha(X)
\approx
\frac1S
\sum_{s=1}^{S}
V(u^{(s)}(X)).
\]

MC acquisitions support q-batches, nonlinear objectives, constraints, and
multi-output models.  Their accuracy depends on sample count and base-sample
handling.

Sobol QMC normal samplers reduce integration error relative to independent
pseudo-random Gaussian draws in many low- to moderate-dimensional acquisition
integrals.

---

## 3. Probability of Improvement

For maximization, define improvement threshold

\[
\tau=f_{\mathrm{best}}+\xi,
\]

where `xi>=0` is an optional exploration margin.  Probability of Improvement is

\[
\operatorname{PI}(x)
=
P(f(x)\ge\tau\mid\mathcal D).
\]

For Gaussian posterior

\[
f(x)\sim\mathcal N(\mu(x),\sigma^2(x)),
\]

\[
\operatorname{PI}(x)
=
\Phi\left(
\frac{\mu(x)-\tau}{\sigma(x)}
\right).
\]

PI ignores the magnitude of improvement.  A tiny likely improvement can be
preferred over a large but less likely improvement.

For transformed probability or utility objectives, `best_f` and `xi` must be in
that transformed scale.

---

## 4. Expected Improvement

Define

\[
I(x)=\max(f(x)-f_{\mathrm{best}},0).
\]

Expected Improvement is

\[
\operatorname{EI}(x)
=
\mathbb E[I(x)\mid\mathcal D].
\]

For Gaussian posterior and

\[
z=rac{\mu-f_{\mathrm{best}}}{\sigma},
\]

\[
\operatorname{EI}(x)
=
(\mu-f_{\mathrm{best}})\Phi(z)
+
\sigma\phi(z),
\]

for `sigma>0`.  The first term is mean improvement weighted by its probability;
the second is an uncertainty contribution.

For q-batch EI,

\[
I(X)
=
\max\left(
\max_{i=1,\ldots,q}f(x_i)-f_{\mathrm{best}},
0
\right),
\]

and the expectation is usually evaluated by Monte Carlo.

### Log Expected Improvement

When improvement is extremely small, ordinary EI may underflow or have poor
gradients.  LogEI evaluates a numerically stable logarithmic representation.
Modern BoTorch workflows often prefer log-improvement variants when applicable.

### Limitations

- ordinary EI assumes a meaningful fixed `best_f`;
- noisy observations can make best observed value unreliable;
- EI can become over-exploitative with underestimated variance;
- qEI does not automatically prevent candidates from duplicating existing
  pending points unless pending/baseline handling is supplied.

---

## 5. Noisy Expected Improvement

Let `X_baseline` be observed or considered baseline inputs.  Their latent values
are uncertain under noisy observations.

A conceptual qNEI definition is

\[
\operatorname{qNEI}(X)
=
\mathbb E
\left[
\max\left(
\max f(X)-\max f(X_{\mathrm{baseline}}),
0
\right)
\right].
\]

The expectation is joint over candidate and baseline latent values.

Important arguments include:

- `X_baseline`;
- objective;
- constraints;
- sampler;
- `X_pending`;
- baseline pruning;
- cached root decompositions when supported.

NEI is not simply EI with a larger variance.  It changes the reference value
from fixed best observation to a posterior-distributed latent baseline.

---

## 6. Confidence-bound acquisitions

For maximization, a common UCB form is

\[
\operatorname{UCB}(x)
=
\mu(x)+\sqrt\beta\,\sigma(x).
\]

Some implementations parameterize the coefficient directly as `beta` rather
than `sqrt(beta)`.  Always check the concrete class definition.

For minimization, use

\[
\operatorname{LCB}(x)
=
\mu(x)-\sqrt\beta\,\sigma(x)
\]

or negate the objective and maximize UCB.

Interpretation:

- small uncertainty coefficient: exploitation;
- large coefficient: exploration;
- theoretical schedules may increase `beta_t` with iteration;
- fixed practical values require calibration to output standardization.

For qUCB, BoTorch uses posterior samples and a batch utility rather than simply
summing pointwise UCB values.

---

## 7. Thompson sampling and posterior sampling

Thompson sampling draws a function sample

\[
\tilde f\sim p(f\mid\mathcal D)
\]

and selects

\[
x_{t+1}\in\arg\max_x\tilde f(x).
\]

Repeated function samples naturally randomize exploration.  For a finite
candidate set, sampling posterior values and choosing the maximum is direct.
For continuous domains, exact function-sample optimization is harder and may
use random features, pathwise samples, or candidate pools.

Advantages:

- simple conceptual policy;
- natural exploration;
- easy parallelization using independent or coordinated samples.

Limitations:

- quality depends on function-sampling approximation;
- continuous optimization can be expensive;
- naive independent marginal samples do not preserve function correlation.

---

## 8. Knowledge Gradient

Knowledge Gradient measures expected improvement in the best posterior decision
after observing a candidate.

Let

\[
M_t=\max_{x'}\mu_t(x').
\]

Then a one-step KG concept is

\[
\operatorname{KG}(x)
=
\mathbb E_{y_x}
\left[
\max_{x'}\mu_{t+1}(x';y_x)
\right]
-
M_t.
\]

KG differs from EI:

- EI values direct improvement at the sampled candidate;
- KG values how the observation changes the best future decision;
- KG can sample a point that is not expected to be good itself if it is
  informative about another region.

Implementation requires fantasy observations and an inner optimization over
future decisions.

---

## 9. Multi-step look-ahead

A multi-step policy recursively values a decision tree:

\[
\alpha_t(x_1)
=
\mathbb E_{y_1}
\left[
\max_{x_2}
\mathbb E_{y_2\mid y_1}
\left[
\cdots
\right]
\right].
\]

BoTorch's `qMultiStepLookahead` represents this with fantasy samples, stage
batch sizes, and stage value functions.

Practical challenges:

- nested optimization;
- large fantasy tensors;
- differentiability of model conditioning;
- mixed-variable and constrained inner problems;
- approximation error from limited fantasies;
- substantial computational cost.

Look-ahead is most useful when future experimentation policy matters enough to
justify the cost.

---

## 10. Information-theoretic BO

Information-theoretic acquisitions target uncertainty about an optimum-related
quantity.

### Predictive Entropy Search

Targets mutual information between the observation and optimizer `x*`:

\[
I(y_x;x^*\mid\mathcal D).
\]

### Max-value Entropy Search

Targets information about the optimum value

\[
f^*=\max_xf(x):
\]

\[
I(y_x;f^*\mid\mathcal D).
\]

### Joint Entropy Search

Targets joint information about optimizer and optimum value:

\[
I(y_x;(x^*,f^*)\mid\mathcal D).
\]

These methods can be sample efficient but require approximations to optimum or
max-value distributions.

---

## 11. Objectives and posterior transforms

A model output need not equal the acquisition objective.

For posterior sample

\[
Y^{(s)}\in\mathbb R^m,
\]

an MC objective maps

\[
T:\mathbb R^m\rightarrow\mathbb R
\]

or

\[
T:\mathbb R^m\rightarrow\mathbb R^{m_{\mathrm{obj}}}.
\]

Examples:

- select one output;
- change minimization to maximization;
- weighted scalarization;
- convert ordinal class probabilities to expected utility;
- aggregate input perturbations;
- apply feasibility constraints.

A posterior transform changes the posterior representation before acquisition
evaluation.  An MC objective changes samples after sampling.  They are related
but not interchangeable, especially for nonlinear transformations because

\[
T(\mathbb E[Y])
e\mathbb E[T(Y)].
\]

---

## 12. Sample-level constraints

Suppose sample-level constraint functions satisfy

\[
c_j(Y)\le0.
\]

A constrained improvement value may be written

\[
V(Y)
=
I(Y)
\prod_j\mathbf 1[c_j(Y)\le0].
\]

Smooth approximations replace hard indicators by sigmoid functions.  This
improves gradients but introduces a temperature parameter and changes the
constraint interpretation.

Probability-of-feasibility weighting is

\[
\alpha_c(x)
=
\alpha_0(x)P(\text{feasible}\mid x).
\]

This factorization is exact only under particular independence and value
assumptions.  MC constrained acquisitions can represent joint samples more
directly.

---

## 13. Batch dependence

For q-batch acquisition, candidate values are correlated.  Let

\[
\mathbf f_X
\sim
\mathcal N(\boldsymbol\mu_X,\Sigma_X).
\]

The covariance affects:

- probability that at least one candidate improves;
- expected maximum within the batch;
- diversity;
- constrained joint outcomes;
- information gain.

Using only marginal means and variances can overvalue redundant points.

For custom posteriors that do not expose cross-candidate covariance, q-batch MC
sampling may become an independent proxy rather than an exact joint posterior.
This limitation must be documented.

---

## 14. Pending points

An acquisition may handle `X_pending` by:

- including pending points in fantasy or joint sample logic;
- conditioning on pending values;
- applying local penalization;
- using sequential batch construction;
- rejecting duplicates after optimization.

A simple distance penalty has form

\[
P(x)
=
\lambda
\exp[-\eta d(x,X_{\mathrm{pending}})].
\]

It encourages diversity but does not represent uncertainty about pending
outcomes.

---

## 15. Acquisition optimization

The acquisition surface is usually nonconvex, even when the posterior mean is
smooth.

### 15.1 Multistart gradient optimization

A common procedure is:

1. draw raw samples;
2. score or filter initialization candidates;
3. choose `num_restarts` initial points;
4. optimize each with gradients;
5. select the best result.

### 15.2 Mixed optimization

For categorical variables, enumerate or specify fixed-feature combinations and
optimize continuous variables conditionally.

### 15.3 Evolutionary optimization

Genetic algorithms, particle swarm, or CMA-ES can handle discontinuity,
non-differentiability, and complex repair.  They may require more acquisition
evaluations.

### 15.4 Torch optimization

Direct Adam-style optimization can be useful for custom differentiable
acquisitions.  It requires projection to bounds and careful restart handling.

### 15.5 Post-processing

After optimization, candidates may be rounded or repaired.  Because repair can
move the point, recompute:

- acquisition value;
- deterministic constraints;
- category validity;
- duplicate status.

---

## 16. Numerical considerations

### 16.1 Monte Carlo variance

Use fixed base samples during local optimization to create a stable sample
average approximation.  Changing random samples every forward call makes the
objective noisy and gradients unreliable.

### 16.2 Log acquisitions

LogEI and related variants reduce underflow and improve gradients when
improvement probabilities are tiny.

### 16.3 Standardization

Acquisition parameters such as `best_f`, `beta`, smoothing temperature, and
reference points depend on objective scale.

### 16.4 Posterior variance floors

Clamping negative numerical variances to a small nonnegative value is common.
Large or frequent negative corrections indicate a deeper covariance issue.

### 16.5 Duplicate points

Exact duplicates can destabilize GP covariance and waste batch budget.  A hard
duplicate penalty may be appropriate unless replicate measurements are
scientifically useful.

---

## 17. Selection guide

| Situation | Useful starting acquisition |
|---|---|
| Low-noise single-objective regression | LogEI / qLogEI |
| Noisy experimental response | qNEI / log noisy improvement variant |
| Simple exploration baseline | qUCB |
| Direct probability target | task-specific probability EI, PI, or UCB |
| Valuable look-ahead information | qKG |
| Explicit multistep planning | qMultiStepLookahead |
| Multi-objective, low noise | qEHVI |
| Multi-objective, noisy baseline | qNEHVI |
| Expensive mixed domain | mixed optimizer with an appropriate acquisition |

The model, objective, and posterior contract must be checked before applying the
name in this table.

---

## 18. `bochan` implementation correspondence

### 18.1 Standard acquisitions resolved by the registry

`src/bochan/api/acquisition_registry.py` maps aliases to BoTorch classes.
Representative mappings include:

| Alias family | Resolved class |
|---|---|
| `qei`, `ei` | `qExpectedImprovement` |
| `qlogei`, `logei` | `qLogExpectedImprovement` |
| `qnei`, `nei` | `qNoisyExpectedImprovement` |
| `qucb`, `ucb` | `qUpperConfidenceBound` |
| `qpi`, `pi` | `qProbabilityOfImprovement` |
| `qkg`, `kg` | `qKnowledgeGradient` |
| `lookahead` | `qMultiStepLookahead` |
| `qehvi`, `ehvi` | `qExpectedHypervolumeImprovement` |
| `qnehvi`, `nehvi` | `qNoisyExpectedHypervolumeImprovement` |

### 18.2 Task-specific acquisitions

Custom BO acquisitions are organized by task:

```text
src/bochan/acquisition/regression/bayesian_optimization/
src/bochan/acquisition/binary/bayesian_optimization/
src/bochan/acquisition/multiclass/bayesian_optimization/
src/bochan/acquisition/ordinal/bayesian_optimization/
src/bochan/acquisition/non_gaussian/bayesian_optimization/
```

Examples registered in the current API include:

- `qBinaryExpectedImprovement`;
- `qBinaryProbabilityOfImprovement`;
- `qBinaryUpperConfidenceBound`;
- `qOrdinalExpectedImprovement`;
- `qOrdinalProbabilityOfImprovement`;
- `qOrdinalUpperConfidenceBound`;
- heteroscedastic regression and ordinal variants;
- task-specific multi-output EHVI, NEHVI, and NParEGO wrappers.

### 18.3 Optimizers

```text
src/bochan/optim/
```

contains alternative optimization backends and constraint/post-processing
logic.  The theoretical acquisition remains the function being maximized; the
optimizer determines how approximately that maximum is found.

---

## 19. New-acquisition checklist

Document:

1. optimization target and direction;
2. posterior space consumed;
3. analytic or Monte Carlo estimator;
4. definition of `best_f` or baseline;
5. objective and constraint handling;
6. q-batch semantics;
7. pending-point behavior;
8. input-perturbation behavior;
9. output and class reductions;
10. returned tensor shape;
11. differentiability assumptions;
12. acquisition optimizer compatibility;
13. whether the implementation is a standard published criterion or a custom
    proxy.

---

## 20. References

- Jones, Schonlau, and Welch, *Efficient Global Optimization of Expensive Black-Box Functions*, 1998.
- Frazier, *A Tutorial on Bayesian Optimization*, 2018.
- Balandat et al., *BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization*, 2020.
- Wilson et al., work on maximizing acquisition functions and sample-average approximations in Bayesian optimization.
