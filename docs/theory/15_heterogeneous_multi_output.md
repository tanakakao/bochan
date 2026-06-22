# 15. Heterogeneous Multi-output Models

A heterogeneous multi-output problem has several responses with different
sample spaces or likelihoods.  For example, one experiment may return

\[
\mathbf y(x)
=
[y_{\mathrm{strength}},
 y_{\mathrm{pass/fail}},
 y_{\mathrm{grade}},
 y_{\mathrm{defect\ type}}].
\]

These outputs cannot be stacked into one Gaussian target tensor without first
defining what each output means.  This chapter separates three constructions:

1. independent heterogeneous submodels;
2. correlated latent multi-output models with heterogeneous likelihoods;
3. objective-space wrappers used for decision making.

The current `HybridMultiOutputModel` primarily implements construction 3 on top
of independent or separately constructed submodels.  It does not by itself
create cross-output posterior covariance.

---

## 1. Homogeneous versus heterogeneous outputs

### 1.1 Homogeneous multi-output

For `m` continuous outputs,

\[
\mathbf y(x)=[y_1(x),\ldots,y_m(x)]\in\mathbb R^m.
\]

All outputs may use Gaussian likelihoods and share one posterior tensor

```text
batch_shape x q x m
```

Examples are several material properties measured on the same continuous
scale family.

### 1.2 Heterogeneous multi-output

A heterogeneous response may contain

\[
y_r\in\mathbb R,
\qquad
y_b\in\{0,1\},
\qquad
y_o\in\{0,\ldots,K_o-1\},
\qquad
y_c\in\{0,\ldots,K_c-1\}.
\]

Each output needs its own likelihood:

\[
p(y_r\mid f_r),
\quad
p(y_b\mid f_b),
\quad
p(y_o\mid f_o,c),
\quad
p(y_c\mid\mathbf f_c).
\]

A raw class label cannot be averaged with a regression value.  The outputs must
be transformed to probabilities, utilities, constraints, or another common
decision representation.

---

## 2. Independent heterogeneous submodels

The simplest model factorizes the posterior:

\[
p(f_1,\ldots,f_m\mid\mathcal D)
=
\prod_{j=1}^{m}p(f_j\mid\mathcal D_j).
\]

Each output can use a different model family and training procedure.

### Advantages

- supports different likelihoods;
- supports different class counts and cutpoints;
- handles different noise models;
- can use different kernels and transforms;
- missing observations can be represented by different datasets;
- one unstable output model does not invalidate a joint covariance factorization.

### Limitations

- no information transfer between outputs;
- no cross-output covariance;
- posterior samples from different outputs are independent unless dependence is
  added externally;
- multi-output information acquisitions cannot exploit correlations.

The current hybrid wrapper is compatible with this construction.

---

## 3. Correlated latent heterogeneous model

A more general construction introduces `Q` shared latent GPs:

\[
u_q(x)\sim\mathcal{GP}(0,k_q),
\qquad q=1,\ldots,Q.
\]

For output `j`, define

\[
f_j(x)=\sum_{q=1}^{Q}a_{jq}u_q(x).
\]

Then

\[
\operatorname{Cov}[f_j(x),f_l(x')]
=
\sum_{q=1}^{Q}a_{jq}a_{lq}k_q(x,x').
\]

Each latent response is connected to a task-specific likelihood:

\[
p(\mathbf y\mid\mathbf f)
=
\prod_{i,j}p_j(y_{ij}\mid f_j(x_i)).
\]

Examples:

- Gaussian likelihood for regression;
- Bernoulli likelihood for binary classification;
- ordered-logit likelihood for ordinal output;
- categorical likelihood for multiclass output;
- Poisson likelihood for counts.

This is a heterogeneous multi-output GP in the statistical sense.  Its
posterior may transfer information through the shared latent processes.

### Identifiability and negative transfer

The loading matrix `A` and latent kernels are not always identifiable without
constraints.  Moreover, forcing unrelated outputs to share latent functions can
cause negative transfer.  Correlation should therefore be validated using
held-out predictive performance and sequential decision performance.

The current `HybridMultiOutputModel` should not be described as this correlated
model unless its submodels themselves share a joint latent process.

---

## 4. Objective-space representation

For decision making, each output is mapped to a scalar score

\[
t_j(x)=T_j[y_j(x)].
\]

The resulting objective vector is

\[
\mathbf t(x)=[t_1(x),\ldots,t_m(x)].
\]

`bochan.models.hybrid.OutputSpec` records how each output is transformed.

### 4.1 Regression output

For a maximization output,

\[
t_j=y_j.
\]

For minimization,

\[
t_j=-y_j.
\]

With weight `w_j` and sign `s_j`,

\[
t_j=s_jw_jy_j.
\]

For a target value `a_j`, the current objective convention is

\[
t_j=-w_j|y_j-a_j|.
\]

This transformation is nondifferentiable at `y_j=a_j`, although it is
differentiable almost everywhere.

### 4.2 Binary probability

For positive class `1`,

\[
t_j=p_j=P(Y_j=1\mid x).
\]

For positive class `0`,

\[
t_j=1-p_j.
\]

### 4.3 Binary expected utility

With class utilities `u_0,u_1`,

\[
t_j=u_0(1-p_j)+u_1p_j.
\]

The conditional utility variance is

\[
\operatorname{Var}(U_j\mid x)
=(1-p_j)(u_0-\bar u_j)^2
+p_j(u_1-\bar u_j)^2.
\]

### 4.4 Ordinal expected utility

For ordinal probabilities `p_jk` and utilities `u_jk`,

\[
t_j
=
\sum_{k=0}^{K_j-1}u_{jk}p_{jk}.
\]

The class-distribution utility variance is

\[
v_j
=
\sum_k p_{jk}(u_{jk}-t_j)^2.
\]

This variance reflects uncertainty of the discrete class utility conditional on
the predicted probabilities.  It is not automatically the epistemic variance
of the expected utility function.

### 4.5 Multiclass probability or utility

For target class `k*`,

\[
t_j=p_{jk^*}.
\]

For utilities,

\[
t_j=\sum_k u_{jk}p_{jk}.
\]

---

## 5. Current `HybridMultiOutputModel`

The implementation is located in

```text
src/bochan/models/hybrid/multi_output.py
```

It receives a sequence of `OutputSpec` objects with fields including:

```text
name
task_type
model
output_index
sign
weight
eq_target
utility_values
positive_class
transform
```

The wrapper calls task-specific accessors such as

- `posterior` or `latent_posterior` for regression;
- probability posterior accessors for binary outputs;
- `class_probs` and ordinal likelihood helpers for ordinal outputs;
- `class_probs` for multiclass outputs.

It returns a unified posterior in an objective or requested output mode with
shape

```text
batch_shape x q x m
```

The available modes are

```text
objective
mean
latent
probability
expected_utility
```

as defined in

```text
src/bochan/models/hybrid/specs.py
```

---

## 6. Current `HybridPosterior`

`HybridPosterior` is defined in

```text
src/bochan/models/hybrid/posterior.py
```

Its mean and marginal variance have shape

```text
batch_shape x q x m
```

The current posterior has **no cross-output covariance matrix**.  Its
reparameterized sampling uses independent normal proxies:

\[
\tilde t_j
=
\mu_j+\sqrt{v_j}\epsilon_j,
\qquad
\epsilon_j\overset{\mathrm{iid}}{\sim}\mathcal N(0,1).
\]

Consequences:

1. covariance between heterogeneous outputs is zero in the proxy posterior;
2. non-Gaussian output distributions are moment-matched or summarized before
   proxy sampling;
3. class-utility distributions are approximated by normal variables;
4. joint-tail events and joint chance constraints may be inaccurate;
5. Monte Carlo multi-objective acquisitions operate on this proxy distribution,
   not the exact joint heterogeneous posterior.

This design is useful for BoTorch interoperability, but its approximation must
be stated explicitly in theory documents and empirical studies.

---

## 7. Transforming variance

For a linear transformation

\[
t=swy,
\]

variance transforms exactly as

\[
\operatorname{Var}(t)=w^2\operatorname{Var}(y).
\]

For a nonlinear transform `h`, the exact variance is

\[
\operatorname{Var}[h(Y)]
=
\mathbb E[h(Y)^2]-\mathbb E[h(Y)]^2.
\]

A first-order delta approximation is

\[
\operatorname{Var}[h(Y)]
\approx
[h'(\mu)]^2\operatorname{Var}(Y).
\]

The current `OutputSpec.transform` applies an arbitrary callable to the mean,
while variance is primarily adjusted by `weight**2`.  Unless the transform is
linear, this is not an exact transformed variance.  Such outputs should be
considered decision-score approximations.

The target-distance transform

\[
t=-w|Y-a|
\]

also requires either Monte Carlo transformation or an analytic folded-normal
calculation for exact moments.  Applying `-abs(mu-a)` to the mean is a plug-in
approximation.

---

## 8. Scalarization and multi-objective use

### 8.1 Weighted scalarization

A scalar objective is

\[
S(x)=\sum_{j=1}^{m}\lambda_jt_j(x).
\]

Weights encode units and preference.  Standardizing each response before
scalarization changes the interpretation of the weights and must be documented.

For independent output proxies,

\[
\operatorname{Var}[S]
=
\sum_j\lambda_j^2v_j.
\]

For correlated outputs, the correct formula is

\[
\operatorname{Var}[S]
=
\boldsymbol\lambda^\top
\Sigma_t
\boldsymbol\lambda.
\]

The current hybrid posterior cannot represent the cross terms.

### 8.2 Pareto optimization

For vector objective

\[
\mathbf t(x)\in\mathbb R^m,
\]

Pareto dominance is defined after all directions have been transformed to
maximization:

\[
\mathbf a\succeq\mathbf b
\Longleftrightarrow
a_j\ge b_j\ \forall j
\quad\text{and}\quad
\exists j:a_j>b_j.
\]

Probability and expected-utility outputs can be included in EHVI/NEHVI, but
their scales and approximation quality affect hypervolume.  Reference points
must be specified in the same transformed objective space.

---

## 9. Heterogeneous outputs as constraints

Some outputs should not be objectives.  A binary classifier may represent
feasibility:

\[
C(x)=P(Y_{\mathrm{feasible}}=1\mid x).
\]

An ordinal grade can define

\[
C(x)=P(Y\ge g_{\min}\mid x).
\]

A regression constraint may define

\[
C(x)=P(g(x)\le 0\mid x).
\]

A constrained acquisition can use

\[
\alpha_c(x)
=
\alpha_o(x)\prod_{j=1}^{m_c}C_j(x).
\]

This preserves the distinction between objective value and feasibility.
Converting every output into one weighted utility can obscure hard engineering
requirements.

---

## 10. Missing and asynchronously observed outputs

In real experiments, not every output is observed at every input.  Independent
submodels naturally support datasets

\[
\mathcal D_j=\{(x_i,y_{ij}):y_{ij}\text{ observed}\}.
\]

A shared-input tensor interface may incorrectly imply complete observations.
The hybrid wrapper's `train_Y` concatenation assumes aligned rows when exposing
a combined training tensor.  For incomplete data, the submodels and their raw
datasets should remain the source of truth.

For asynchronous measurement, acquisitions may need to value an experiment
according to which outputs will actually be returned and at what cost.

---

## 11. Information acquisition for heterogeneous outputs

A general information objective is

\[
\alpha(x)
=
I(Y_{\mathcal O};\Theta_{\mathcal T}\mid x,\mathcal D),
\]

where `O` is the set of measured outputs and `T` is the scientific target.
Examples include:

- learning all output functions;
- learning only feasibility;
- learning a Pareto frontier;
- learning a joint safe region;
- learning one ordinal boundary while also predicting a continuous property.

Summing output-wise entropies assumes separability:

\[
\alpha(x)=\sum_jw_j\alpha_j(x).
\]

A correlated latent model would permit joint mutual information, but the
current independent proxy does not contain all required dependence.

---

## 12. Recommended terminology

Use the following terms precisely:

| Term | Meaning |
|---|---|
| Multi-output | More than one response channel. |
| Multitask | Outputs/tasks are explicitly modeled and may share statistical structure. |
| ModelList / independent | Separate posterior models combined at the interface level. |
| Heterogeneous likelihood | Different response distributions across outputs. |
| Hybrid objective wrapper | Converts heterogeneous model outputs to a common score tensor. |
| Correlated heterogeneous GP | Joint latent model with cross-output covariance and task-specific likelihoods. |

The current `HybridMultiOutputModel` is principally a hybrid objective wrapper.
Its submodels may themselves be multitask, but the wrapper alone does not add
cross-output covariance.

---

## 13. Validation

A heterogeneous model should be evaluated at three levels.

### Per-output prediction

- regression RMSE/NLPD/coverage;
- binary Brier score, log loss, calibration;
- multiclass log loss and expected calibration error;
- ordinal ranked probability score and boundary calibration.

### Cross-output behavior

- empirical residual dependence;
- held-out performance with and without shared structure;
- calibration of joint events;
- negative transfer.

### Decision performance

- scalar regret;
- hypervolume regret;
- feasible-regret;
- boundary estimation loss;
- experiments required to reach a target.

A joint model should not be preferred solely because it reports a nonzero task
correlation.

---

## 14. Source map

| Component | Implementation |
|---|---|
| Heterogeneous model wrapper | `src/bochan/models/hybrid/multi_output.py` |
| Output metadata and modes | `src/bochan/models/hybrid/specs.py` |
| Independent normal proxy posterior | `src/bochan/models/hybrid/posterior.py` |
| Hybrid prediction helpers | `src/bochan/models/hybrid/prediction.py` |
| Hybrid objective | `src/bochan/acquisition/objective/hybrid.py` |
| API model construction | `src/bochan/api/factory.py` and `src/bochan/api/configs.py` |
| Multi-objective theory | `docs/theory/07_multi_objective_and_constraints.md` |

---

## 15. References

- Álvarez, Rosasco, and Lawrence, *Kernels for Vector-Valued Functions: A Review*, 2012.
- Moreno-Muñoz et al., *Heterogeneous Multi-output Gaussian Process Prediction*, 2018.
- Rasmussen and Williams, *Gaussian Processes for Machine Learning*, 2006.
