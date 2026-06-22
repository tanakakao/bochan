# 07. Multi-objective Optimization and Constraints

Many experiments return several responses, but not every response is an
objective.  Some are constraints, diagnostics, costs, or auxiliary predictions.
This chapter separates multi-output modeling from multi-objective decision
making and develops Pareto dominance, hypervolume, scalarization, and
probabilistic constraints.

Heterogeneous-output posterior construction is treated in Chapter 15.

---

## 1. Multi-output, multi-objective, and multitask

A model is multi-output if it predicts

$$
\mathbf y(x)
=[y_1(x),\ldots,y_m(x)].
$$

A problem is multi-objective if several transformed outputs are optimized:

$$
\mathbf f(x)
=[f_1(x),\ldots,f_M(x)].
$$

The number of model outputs `m` and objective dimensions `M` need not match.
For example:

```text
model output 0: strength             -> objective
model output 1: cost                 -> objective after sign reversal
model output 2: feasibility label    -> constraint probability
model output 3: ordinal grade        -> expected-utility objective
model output 4: sensor diagnostic    -> auxiliary only
```

A multitask model explicitly shares statistical structure between outputs.  An
independent ModelList is multi-output but does not model cross-output
covariance.

---

## 2. Direction normalization

BoTorch multi-objective acquisitions generally use maximization.  For original
objectives with direction indicators

$$
s_j=
\begin{cases}
+1,&\text{maximize},\\
-1,&\text{minimize},
\end{cases}
$$

define

$$
f_j(x)=s_jg_j(x).
$$

All Pareto, dominance, and reference-point calculations must use the transformed
maximization space.

For target matching,

$$
f_j(x)=-|g_j(x)-a_j|
$$

or a smooth alternative can be used.  This is a nonlinear transformation, so
posterior moments should be transformed by sampling or an appropriate
approximation.

---

## 3. Pareto dominance

For maximization, vector `a` weakly dominates `b` if

$$
a_j\ge b_j
\quad\forall j.
$$

It strictly dominates `b` if it weakly dominates and

$$
\exists j:a_j>b_j.
$$

The Pareto set in input space is

$$
\mathcal P_X
=
\left\{
 x\in\mathcal X:
\nexists x'\in\mathcal X
\text{ such that }
\mathbf f(x')\succ\mathbf f(x)
\right\}.
$$

The Pareto frontier is its image in objective space:

$$
\mathcal P_Y
=
\{\mathbf f(x):x\in\mathcal P_X\}.
$$

A multi-objective optimizer approximates a set of trade-offs, not one unique
optimum, unless an additional preference model is supplied.

---

## 4. Hypervolume

Let the maximization reference point be

$$
\mathbf r=(r_1,\ldots,r_M),
$$

chosen worse than the objective region of interest.  For a nondominated set
`P`, dominated hypervolume is

$$
\operatorname{HV}(P;\mathbf r)
=
\lambda_M\left(
\bigcup_{\mathbf y\in P}
[\mathbf r,\mathbf y]
\right),
$$

where `lambda_M` is `M`-dimensional volume.

For a candidate outcome `y`, hypervolume improvement is

$$
\operatorname{HVI}(\mathbf y)
=
\operatorname{HV}(P\cup\{\mathbf y\};\mathbf r)
-
\operatorname{HV}(P;\mathbf r).
$$

Expected Hypervolume Improvement is

$$
\operatorname{EHVI}(x)
=
\mathbb E
[
\operatorname{HVI}(\mathbf f(x))
].
$$

For q-batch candidates,

$$
\operatorname{qEHVI}(X)
=
\mathbb E
\left[
\operatorname{HV}(P\cup\mathbf f(X);\mathbf r)
-
\operatorname{HV}(P;\mathbf r)
\right].
$$

---

## 5. Noisy hypervolume improvement

When baseline outcomes are noisy, the latent Pareto frontier is uncertain.
qNEHVI integrates over posterior latent values at both baseline and candidate
points.

Conceptually,

$$
\operatorname{qNEHVI}(X)
=
\mathbb E
\left[
\operatorname{HV}
(P(\mathbf f_B)\cup\mathbf f_X;\mathbf r)
-
\operatorname{HV}
(P(\mathbf f_B);\mathbf r)
\right],
$$

where `B` is `X_baseline` and `P(.)` extracts nondominated latent baseline
outcomes for each posterior sample.

qNEHVI is not qEHVI with larger predictive variance.  It treats the current
frontier itself as uncertain.

---

## 6. Reference-point selection

A valid reference point should be worse than outcomes whose hypervolume should
matter.

### 6.1 Objective-space consistency

If an output is transformed by

$$
f_j=s_jw_jg_j,
$$

then `r_j` must be expressed after the same sign and weight transformation.

Probability objectives commonly lie in `[0,1]`; ordinal expected utilities lie
in the utility range; standardized regression outputs may be returned in
original or standardized units depending on the posterior contract.

### 6.2 Too optimistic reference point

If the reference point is better than much of the frontier, valid solutions may
contribute no hypervolume.

### 6.3 Too pessimistic reference point

A very poor reference point can make hypervolume dominated by objective scale
rather than local frontier improvement.

### 6.4 Dynamic reference points

Updating the reference point during optimization changes the metric over time.
This may be operationally useful but complicates comparison of hypervolume
curves.  Benchmark evaluations should usually use a fixed reference point.

---

## 7. Objective scaling

Hypervolume and scalarization are sensitive to objective scale.  Suppose

$$
f_1\in[0,1],
\qquad
f_2\in[0,1000].
$$

Without normalization, changes in `f_2` dominate Euclidean geometry and many
scalarizations.

Possible scaling includes:

$$
\tilde f_j
=
\frac{f_j-a_j}{b_j-a_j},
$$

or standardization.  Scaling parameters should be based on:

- domain knowledge;
- stable historical bounds;
- a fixed pilot dataset;
- clearly documented adaptive estimates.

Adaptive normalization can change preferences between iterations.

---

## 8. Weighted-sum scalarization

For weights

$$
w_j\ge0,
\qquad
\sum_jw_j=1,
$$

define

$$
s_w(\mathbf f)
=
\sum_jw_jf_j.
$$

Advantages:

- simple;
- compatible with scalar acquisitions;
- easy to encode explicit preferences.

Limitations:

- only supported Pareto points are reachable on nonconvex frontiers;
- weights depend on units;
- one fixed weight explores only one trade-off region.

---

## 9. Chebyshev scalarization

For reference or ideal point `z`, augmented Chebyshev scalarization is

$$
s_w(\mathbf f)
=
\min_j
w_j(f_j-z_j)
+
\rho
\sum_jw_j(f_j-z_j),
$$

under a maximization-oriented convention.  Sign conventions vary; the
implementation must be checked.

Chebyshev scalarization can recover nonconvex parts of a Pareto frontier more
readily than a weighted sum.

---

## 10. NParEGO

NParEGO samples random scalarization weights and applies a scalar noisy
improvement acquisition.  A typical iteration is:

1. sample weight vector `w`;
2. scalarize baseline and candidate outputs;
3. construct scalar qNEI or related acquisition;
4. optimize the scalar acquisition;
5. repeat with new weights across BO iterations.

Random scalarization spreads evaluations across trade-off directions.

### Strong negative correlation caveat

If objectives are strongly negatively correlated and a simple weighted sum is
used, improvement in one can cancel the other.  Chebyshev-style scalarization
or hypervolume methods are often more suitable.

---

## 11. Preference-based utility

When the user ultimately needs one decision, a utility function may be more
appropriate than a Pareto frontier:

$$
U(\mathbf f).
$$

Examples:

- monetary value;
- desirability functions;
- target-distance penalties;
- piecewise penalties for specifications;
- risk-adjusted utility;
- learned pairwise preference model.

A utility function resolves trade-offs but requires preference assumptions.
Hypervolume avoids committing to one preference but returns a set that still
requires downstream selection.

---

## 12. Deterministic input constraints

Known design-space constraints include

$$
A_{\mathrm{eq}}x=b_{\mathrm{eq}},
$$

$$
A_{\mathrm{ineq}}x\le b_{\mathrm{ineq}}.
$$

They can be enforced by:

- acquisition optimizer constraints;
- reparameterization;
- feasible initialization;
- repair operators;
- rejection;
- discrete enumeration.

### Equality constraints

Exact equality constraints define a lower-dimensional manifold.  Penalty-only
optimization may leave residual violations unless the tolerance is explicitly
accepted.

### Step or grid constraints

For base `a_j` and step `s_j`,

$$
x_j=a_j+k_js_j,
\qquad k_j\in\mathbb Z.
$$

Rounding after optimization can violate coupled constraints.  Constraint-aware
rounding or repair is needed.

### Compositional constraints

Examples include

$$
\sum_{j\in S}x_j=c
$$

with lower/upper bounds.  Independent rounding of components can break the sum.

### k-sparsity

A candidate may require

$$
\|x\|_0\le k.
$$

This is a design constraint, not the same as SAAS feature sparsity in the GP.

---

## 13. Unknown outcome constraints

Suppose constraint latent response is

$$
g_j(x)
$$

and feasibility is

$$
g_j(x)\le0.
$$

Posterior probability of feasibility is

$$
p_j^{\mathrm{feas}}(x)
=
P(g_j(x)\le0\mid\mathcal D).
$$

For Gaussian posterior,

$$
p_j^{\mathrm{feas}}(x)
=
\Phi\left(
\frac{-\mu_j(x)}{\sigma_j(x)}
\right).
$$

If constraints are independent,

$$
P(\text{all feasible})
=
\prod_jp_j^{\mathrm{feas}}.
$$

With correlated constraints, use a joint multivariate probability or posterior
samples.

---

## 14. Classification constraints

A binary classifier can directly model

$$
P(\mathrm{feasible}\mid x).
$$

A multiclass constraint may define acceptable class set `A`:

$$
P(Y\in A\mid x)
=
\sum_{k\in A}p_k(x).
$$

An ordinal minimum-grade constraint is

$$
P(Y\ge g\mid x)
=
\sum_{k=g}^{K-1}p_k(x).
$$

These probabilities should be calibrated.  Using hard predicted class labels
discards uncertainty and produces discontinuous feasibility decisions.

---

## 15. Chance constraints

A chance constraint is

$$
P(g(x,\omega)\le0)
\ge1-\epsilon,
$$

where `omega` represents observation noise, environmental uncertainty, input
perturbation, or posterior uncertainty depending on the problem.

The probability space must be stated.  For example:

- posterior chance constraint over unknown latent function;
- execution chance constraint over input perturbation;
- predictive chance constraint over future measurement noise;
- joint chance constraint over several outcomes.

These lead to different feasible sets.

---

## 16. Feasibility-weighted acquisition

A common approximation is

$$
\alpha_c(x)
=
\alpha_0(x)
P(\mathrm{feasible}\mid x).
$$

For several independent constraints,

$$
\alpha_c(x)
=
\alpha_0(x)
\prod_jp_j^{\mathrm{feas}}(x).
$$

Advantages:

- simple;
- smooth;
- can guide search before feasible points are known.

Limitations:

- assumes a particular value factorization;
- can become nearly zero when many constraints are multiplied;
- may under-explore uncertain feasibility boundaries;
- does not strictly enforce a probability threshold.

---

## 17. Sample-level constrained acquisition

For posterior sample `Y^(s)` and constraint functions `c_j`, define feasible
indicator

$$
I^{(s)}
=
\prod_j
\mathbf1[c_j(Y^{(s)})\le0].
$$

A constrained improvement estimator is

$$
\hat\alpha(X)
=
\frac1S
\sum_s
I^{(s)}V(Y^{(s)}).
$$

This preserves sample-level dependence between objective and constraints if the
posterior samples contain it.

Smooth sigmoid approximations can improve gradients but add temperature and
bias.

---

## 18. Feasible Pareto frontier

For multi-objective constrained optimization, the feasible Pareto set is

$$
\mathcal P_X^{\mathrm{feas}}
=
\left\{
 x\in\mathcal X:
 x\text{ feasible and nondominated among feasible points}
\right\}.
$$

Hypervolume should be computed from feasible outcomes.  Under uncertain
constraints, posterior samples can have different feasible Pareto frontiers.
Noisy constrained hypervolume acquisitions integrate over this uncertainty.

---

## 19. Constraints before feasibility is found

If no observed point is feasible, improvement relative to a feasible frontier
is undefined or uninformative.  Strategies include:

- maximize probability of feasibility;
- use feasibility-weighted uncertainty;
- optimize expected constraint violation reduction;
- use a two-stage acquisition;
- specify a safe initial point;
- model slack or violation magnitude rather than only binary labels.

The transition from feasibility search to objective optimization should be
explicit.

---

## 20. Multi-output reduction versus Pareto preservation

An output tensor

```text
sample_shape x batch_shape x q x m
```

can be handled in different ways.

### Preserve outputs

Return

```text
sample_shape x batch_shape x q x M
```

to a multi-objective acquisition.

### Scalarize

Return

```text
sample_shape x batch_shape x q
```

to a scalar acquisition.

### Split into objectives and constraints

Use selected output dimensions for objective vector and others for sample-level
constraint callables.

### Select one output

Optimize a named response while treating others as diagnostics.

These choices belong in an objective layer, not in the raw model.

---

## 21. Heterogeneous outputs

Suppose outputs include:

- regression value;
- binary probability;
- ordinal expected utility;
- multiclass acceptable-set probability.

Each must first be converted to a decision-space scalar.  Only then should the
vector be passed to scalarization or hypervolume.

The current `HybridMultiOutputModel` performs this conversion using
`OutputSpec`.  Its proxy posterior assumes independent normal sampling of
transformed channels unless dependence is already represented by a submodel.
See Chapter 15.

---

## 22. q-batch and pending points

For qEHVI or qNEHVI, the acquisition values the union of outcomes from all
candidates.  Candidate correlation matters because two similar points may
improve the same hypervolume region.

Pending points can affect the future Pareto frontier.  Proper handling uses
`X_pending` or fantasy logic.  A distance penalty prevents duplicate locations
but does not represent their uncertain future outcomes.

---

## 23. Candidate repair and constraints

Suppose acquisition optimization returns `X_raw`, then repair produces

$$
X_{\mathrm{repair}}=R(X_{\mathrm{raw}}).
$$

Because

$$
\alpha(R(X))
e\alpha(X)
$$

in general, repair can change candidate quality.  After repair:

1. re-evaluate deterministic constraints;
2. re-evaluate acquisition value;
3. check duplicate candidates;
4. verify categorical validity;
5. report residual tolerance for approximate constraints.

For q-batches, repair each point independently only if constraints are
pointwise.  Batch-level constraints require joint repair.

---

## 24. Evaluation

### Multi-objective metrics

- hypervolume;
- hypervolume regret;
- epsilon indicator;
- generational distance when a true frontier is known;
- coverage and diversity of recommended trade-offs.

### Constraint metrics

- feasible recommendation rate;
- cumulative violation count;
- violation magnitude;
- false-feasible probability calibration;
- feasible regret;
- time or evaluations until first feasible point.

### Combined sequential metrics

Plot hypervolume or feasible regret against:

- experiment count;
- total cost;
- unsafe evaluations;
- wall-clock time.

---

## 25. `bochan` implementation correspondence

### 25.1 Standard multi-objective acquisitions

The acquisition registry resolves aliases to BoTorch classes:

| Alias | Class |
|---|---|
| `qehvi`, `ehvi` | `qExpectedHypervolumeImprovement` |
| `qnehvi`, `nehvi` | `qNoisyExpectedHypervolumeImprovement` |
| `nparego` | scalar acquisition with randomized scalarization setup in the high-level API |

### 25.2 Objective implementations

```text
src/bochan/acquisition/objective/
```

includes task-specific and hybrid objectives for:

- scalar regression output;
- classification probability;
- ordinal utility;
- multi-output vectors;
- risk aggregation.

### 25.3 Feasibility wrappers

```text
src/bochan/acquisition/feasible/wrapper.py
src/bochan/acquisition/feasible/constraints.py
src/bochan/acquisition/feasible/README.md
```

provide model-based feasibility composition around acquisitions.

### 25.4 Hybrid outputs

```text
src/bochan/models/hybrid/multi_output.py
src/bochan/models/hybrid/specs.py
src/bochan/acquisition/objective/hybrid.py
```

convert heterogeneous submodel outputs into objective-space channels.

### 25.5 Optimizer constraints and repair

```text
src/bochan/optim/
```

contains gradient, torch, evolutionary, mixed, rounding, and repair-related
logic used after acquisition construction.

### 25.6 Task-specific multi-output acquisitions

The registry includes regression, binary, multiclass, ordinal, and
heteroscedastic multi-output EHVI, NEHVI, and NParEGO wrappers under their task
folders.

---

## 26. Configuration checklist

Specify:

1. model outputs;
2. objective outputs;
3. constraint outputs;
4. sign and scale for each objective;
5. probability or utility transformations;
6. correlation assumptions;
7. reference point;
8. baseline points;
9. scalarization and weight distribution if used;
10. deterministic input constraints;
11. probabilistic outcome constraints;
12. feasibility threshold;
13. behavior before any feasible point exists;
14. q-batch and pending logic;
15. rounding or repair;
16. external evaluation metric.

---

## 27. References

- Deb, *Multi-Objective Optimization Using Evolutionary Algorithms*, 2001.
- Emmerich and Deutz, work on hypervolume-based multi-objective optimization.
- Daulton, Balandat, and Bakshy, work on differentiable Monte Carlo EHVI and NEHVI.
- Knowles, *ParEGO: A Hybrid Algorithm with On-line Landscape Approximation for Expensive Multiobjective Optimization Problems*, 2006.
