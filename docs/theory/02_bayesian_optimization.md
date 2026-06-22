# 02. Bayesian Optimization

Bayesian optimization (BO) is a sequential decision framework for expensive,
noisy, or analytically unavailable objective functions.  It combines a
probabilistic surrogate with a decision rule that trades immediate performance
against the value of collecting information.

This chapter defines the BO problem and loop.  Chapter 03 derives acquisition
functions.  Chapters 06 and 07 explain discrete-output and multi-objective BO.

---

## 1. Optimization problem

The standard single-objective problem is

$$
x^*\in\arg\max_{x\in\mathcal X} f(x),
$$

where evaluating `f(x)` is expensive.  The observation may be noisy:

$$
y(x)=f(x)+\varepsilon(x).
$$

The design space may include:

- continuous variables;
- integer or grid variables;
- categorical variables;
- linear or nonlinear constraints;
- sparse-composition constraints;
- multiple coupled experimental settings.

BO is appropriate when the number of available evaluations is much smaller than
the number required by ordinary global optimization.

---

## 2. Data and posterior state

After `t` rounds, define

$$
\mathcal D_t
=
\{(x_i,y_i)\}_{i=1}^{n_t}.
$$

The surrogate posterior is

$$
p(f\mid\mathcal D_t).
$$

For non-Gaussian observations, `f` may be a latent function and the decision
quantity may instead be a probability or expected utility.  It is useful to
write the decision function as

$$
u(x)=T[p(y\mid x,\mathcal D_t)],
$$

where `T` maps a predictive distribution to what the user values.

Examples:

$$
u(x)=\mathbb E[f(x)\mid\mathcal D_t],
$$

$$
u(x)=P(Y=1\mid x,\mathcal D_t),
$$

$$
u(x)=\sum_k u_kP(Y=k\mid x,\mathcal D_t).
$$

The acquisition must use a posterior or sample representation consistent with
this objective space.

---

## 3. Sequential policy

A BO policy chooses

$$
x_{t+1}
\in
\arg\max_{x\in\mathcal X}
\alpha_t(x;\mathcal D_t).
$$

For a batch of `q` candidates,

$$
X_{t+1}
=
[x_{t+1,1},\ldots,x_{t+1,q}]
\in\mathcal X^q,
$$

$$
X_{t+1}
\in
\arg\max_{X\in\mathcal X^q}
\alpha_t(X;\mathcal D_t).
$$

The policy is adaptive because every new observation changes the posterior and
therefore the next acquisition surface.

---

## 4. Exploitation and exploration

Exploitation prefers candidates with high predicted utility.  Exploration
prefers candidates whose observations may improve future decisions.

A purely greedy policy is

$$
x_{t+1}\in\arg\max_x\mathbb E[u(x)\mid\mathcal D_t].
$$

It can fail when the posterior mean is inaccurate in unexplored regions.

A purely uncertainty-driven policy can spend observations in regions with no
chance of useful objective value.  BO acquisitions combine the two through
improvement, confidence bounds, information gain, posterior sampling, or
look-ahead value.

---

## 5. Utility, direction, and scale

BoTorch acquisition functions generally use a maximization convention.  A
minimization objective `g(x)` is commonly transformed as

$$
u(x)=-g(x).
$$

For a target-matching problem,

$$
u(x)=-|g(x)-a|
$$

or a smooth alternative can be used.

The following must be in the same scale:

- posterior samples passed to the objective;
- `best_f`;
- constraint outputs;
- multi-objective reference point;
- observed baselines used to construct the acquisition.

If a model fits standardized outputs but untransforms its posterior, `best_f`
should usually be in original response units.  If a custom objective acts in
standardized space, its references must be standardized too.

---

## 6. Regret

### 6.1 Instantaneous regret

For maximization,

$$
r_t=f(x^*)-f(x_t).
$$

### 6.2 Cumulative regret

$$
R_T=\sum_{t=1}^{T}r_t.
$$

This is important when every evaluated candidate is deployed and poor trials
have real cost.

### 6.3 Simple regret

$$
s_T
=f(x^*)-\max_{1\le t\le T}f(x_t).
$$

Simple regret is often the main metric in experimental BO, where the final best
condition matters more than intermediate trial quality.

### 6.4 Recommendation regret

If the final recommendation is

$$
\hat x_T
\in
\arg\max_x\mathbb E[f(x)\mid\mathcal D_T],
$$

then

$$
r_{\mathrm{rec}}
=f(x^*)-f(\hat x_T).
$$

Evaluation should state whether it uses best observed, best latent, or posterior
recommended value.

---

## 7. Noisy observations

Suppose

$$
y_i=f(x_i)+\varepsilon_i,
\qquad
\varepsilon_i\sim\mathcal N(0,\sigma_i^2).
$$

The best observed value

$$
\max_i y_i
$$

is biased upward under noise.  An unusually positive noise realization can look
like an optimum.

Noise-aware acquisitions treat latent baseline values as uncertain.  Noisy
Expected Improvement integrates over posterior samples at baseline points:

$$
\alpha_{\mathrm{NEI}}(X)
=
\mathbb E_{\mathbf f_B,\mathbf f_X}
\left[
\max\left(
\max \mathbf f_X-\max \mathbf f_B,
0
\right)
\right].
$$

Here `B` denotes baseline inputs.  The exact implementation includes q-batch,
objective, and constraint handling.

Observation noise and input perturbation are distinct.  Chapter 08 treats input
uncertainty and risk.

---

## 8. Initial design

Before fitting a useful surrogate, BO needs an initial dataset.  Common choices
include:

- Sobol sequences;
- Latin hypercube sampling;
- factorial or fractional-factorial designs;
- historical data;
- expert-selected safe points;
- stratified designs over categorical combinations.

A good initial design should cover important dimensions and categories without
using too much of the experimental budget.

For mixed spaces, continuous coverage alone is not sufficient.  Rare or
scientifically important categories may need explicit representation.

For constrained problems, a design containing no feasible points can make
objective optimization difficult.  A separate feasibility-search phase may be
needed.

---

## 9. q-batch optimization

A batch acquisition evaluates the joint value of multiple points:

$$
\alpha(X),
\qquad X=[x_1,\ldots,x_q].
$$

The value is generally not additive:

$$
\alpha(X)
e\sum_{i=1}^q\alpha(x_i).
$$

Correlated candidates provide redundant information.  A proper joint
acquisition uses joint posterior samples or covariance.

### 9.1 Joint selection

Optimize all `q` candidates simultaneously.  This captures dependence but
creates a `q*d`-dimensional optimization problem.

### 9.2 Sequential greedy batch construction

Select one candidate, mark it pending, then select the next.  This is cheaper
and often robust but only approximates joint batch optimization.

### 9.3 Top-q pointwise selection

Selecting the `q` largest pointwise scores ignores dependence and often returns
clustered candidates.  It should be used only with explicit diversity logic or
when evaluations are naturally independent across discrete groups.

---

## 10. Pending and asynchronous experiments

In asynchronous BO, some candidates have been launched but their outcomes are
unknown.  Let

$$
X_{\mathrm{pending}}
$$

denote those points.

Possible treatments include:

- fantasy observations;
- explicit `X_pending` support in the acquisition;
- sequential conditioning;
- local penalization;
- distance-based duplicate avoidance.

A distance penalty prevents duplication but does not account for the possible
information or objective value of pending outcomes.  It is an engineering
approximation rather than Bayesian conditioning.

---

## 11. Constraints

The general constrained problem is

$$
\max_x f(x)
\quad\text{subject to}\quad
c_j(x)\le 0,
\qquad j=1,\ldots,J.
$$

There are three distinct layers.

### 11.1 Known input constraints

These are deterministic functions of `x`, such as

$$
Ax\le b.
$$

They belong in acquisition optimization or candidate repair.

### 11.2 Unknown outcome constraints

These require surrogate models.  A feasibility probability is

$$
P(c_j(x)\le0\mid\mathcal D_t).
$$

A constrained acquisition may multiply utility by feasibility or include
sample-level constraint indicators.

### 11.3 Operational post-processing

Rounding, compositional repair, k-sparsity, and valid category combinations may
be enforced after continuous optimization.  Repair can alter the acquisition
value, so repaired candidates should be re-evaluated and checked.

Chapter 07 gives detailed multi-objective and constraint theory.

---

## 12. Mixed continuous and categorical spaces

Let

$$
x=(x_c,x_g)
$$

contain continuous and categorical components.

Optimization strategies include:

- enumerate category assignments and optimize continuous variables;
- use `optimize_acqf_mixed` with `fixed_features_list`;
- evolutionary optimization over the full mixed representation;
- custom discrete neighborhood search;
- relax categories only when the relaxation has a valid interpretation.

Continuous gradient optimization over integer category codes is generally
invalid.  Input transforms must preserve categorical columns.

---

## 13. High-dimensional BO

High-dimensional problems are difficult because:

- posterior geometry is weakly identified with few data;
- acquisition optimization becomes harder;
- most candidate directions may be irrelevant;
- distance concentration weakens stationary kernels.

Possible structural assumptions are:

- sparse relevant original dimensions: SAAS;
- low-dimensional linear subspace: REMBO;
- high-variance linear manifold: PCA;
- nonlinear learned representation: VAE-GP or DKL;
- local trust-region structure.

These assumptions are not interchangeable.  Chapter 14 compares them.

---

## 14. Look-ahead decision making

One-step acquisitions value the next observation.  A look-ahead policy values
future decisions after one or more hypothetical observations.

### 14.1 Knowledge Gradient

The one-step KG concept is

$$
\operatorname{KG}(x)
=
\mathbb E_{y_x}
\left[
\max_{x'}\mu_{t+1}(x')
\right]
-
\max_{x'}\mu_t(x').
$$

It values improvement in the best posterior decision rather than immediate
objective improvement at `x`.

### 14.2 Multi-step look-ahead

A multi-step policy nests future acquisition optimizations.  It is more
principled but computationally expensive because it introduces fantasy branches
and inner optimization problems.

### 14.3 Practical limitations

- model conditioning or fantasizing must be supported;
- custom classification and ordinal wrappers may only approximate conditioning;
- inner optimizations increase variance and runtime;
- mixed and constrained domains make look-ahead substantially harder.

---

## 15. Stopping criteria

Possible stopping rules include:

- experimental budget exhausted;
- acquisition value below a threshold;
- no meaningful improvement over several iterations;
- posterior probability of reaching a target is sufficiently small;
- recommended candidate stabilizes;
- credible interval around best utility is sufficiently narrow;
- scientific or operational success criterion achieved.

Acquisition values are not directly comparable across different acquisition
families, objective scales, or model standardizations.  A stopping threshold
must be calibrated for the specific configuration.

---

## 16. BO evaluation protocol

A credible benchmark should include:

1. multiple random initial designs;
2. multiple model and acquisition seeds;
3. identical evaluation budgets;
4. the same noise and perturbation distributions;
5. both predictive and decision metrics;
6. optimizer-failure and duplicate-candidate statistics;
7. wall-clock or model-fitting cost when methods differ substantially.

Recommended decision metrics include:

- best latent value;
- best observed value;
- simple regret;
- recommendation regret;
- feasible regret;
- hypervolume for multi-objective problems;
- probability of meeting a target;
- robustness under repeated execution.

---

## 17. High-level `bochan` workflow

A typical workflow is:

```python
optimizer = BayesianOptimizer(
    model_config=...,
    acquisition_config=...,
    fit_config=...,
    bounds=bounds,
)

optimizer.fit(train_X, train_Y)
candidates = optimizer.suggest(q=q)
```

The exact API may expose additional configuration, but the conceptual pipeline
is:

```text
ModelConfig
    -> model_registry
    -> model construction
FitConfig
    -> model-specific fit function
AcquisitionConfig
    -> acquisition_registry
    -> objective / baseline / reference setup
Optimizer configuration
    -> optimize_acqf, mixed, torch, or evolutionary backend
Candidate post-processing
    -> rounding / repair / constraints
```

---

## 18. Implementation correspondence

| Theory object | `bochan` / BoTorch implementation |
|---|---|
| Surrogate construction | `src/bochan/api/model_registry.py` and task-specific model folders |
| Standard qEI / qNEI / qUCB / qPI | resolved through `src/bochan/api/acquisition_registry.py` to BoTorch classes |
| qKG | BoTorch `qKnowledgeGradient` through the acquisition registry |
| Multi-step look-ahead | BoTorch `qMultiStepLookahead` through the registry |
| qEHVI / qNEHVI | BoTorch multi-objective acquisitions through the registry |
| High-level orchestration | `src/bochan/api/factory.py` and high-level optimizer classes |
| Gradient acquisition optimization | BoTorch `optimize_acqf` / `optimize_acqf_mixed` paths |
| Alternative optimizers | `src/bochan/optim/` |
| Candidate constraints and repair | optimizer and post-processing modules under `src/bochan/optim/` |

The registry contains aliases such as `qei`, `qlogei`, `qnei`, `qucb`, `qpi`,
`qkg`, `lookahead`, `qehvi`, and `qnehvi`.  An alias identifies a class but does
not specify required arguments such as `best_f`, `X_baseline`, sampler,
objective, constraints, or reference point.

---

## 19. Configuration checklist

Before running BO, specify:

1. objective and direction;
2. observation type and likelihood;
3. input bounds and category definitions;
4. known input constraints;
5. unknown outcome constraints;
6. model and inference method;
7. input and outcome transforms;
8. acquisition function;
9. baseline or current-best definition;
10. batch size and pending points;
11. acquisition optimizer and restarts;
12. candidate rounding or repair;
13. stopping and evaluation metrics;
14. random seeds and reproducibility policy.

---

## 20. References

- Mockus, foundational work on Bayesian global optimization.
- Jones, Schonlau, and Welch, *Efficient Global Optimization of Expensive Black-Box Functions*, 1998.
- Frazier, *A Tutorial on Bayesian Optimization*, 2018.
- Balandat et al., *BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization*, 2020.
