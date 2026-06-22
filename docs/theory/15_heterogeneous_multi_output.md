# 15. Heterogeneous Multi-output Models

A heterogeneous multi-output problem contains responses with different sample
spaces or likelihoods.  One experiment may return

```math
\mathbf y(x)
=
[y_{\mathrm{strength}},
 y_{\mathrm{pass/fail}},
 y_{\mathrm{grade}},
 y_{\mathrm{failure\ mode}}].
```

The responses may be continuous, binary, ordered, or multiclass.  They cannot be
stacked into one Gaussian target and interpreted as equivalent outputs.

This chapter focuses on statistical and posterior construction.  Pareto
optimization, scalarization, and constraints are treated in Chapter 07.

---

## 1. Homogeneous and heterogeneous outputs

### Homogeneous outputs

All outputs share the same response type and often the same likelihood family:

```math
\mathbf y(x)
=[y_1(x),\ldots,y_m(x)]
\in\mathbb R^m.
```

Examples:

- several continuous material properties;
- several binary labels;
- several ordinal grades with the same class structure.

### Heterogeneous outputs

Outputs have different supports:

```math
y_r\in\mathbb R,
\qquad
y_b\in\{0,1\},
```

```math
y_o\in\{0,\ldots,K_o-1\},
\qquad
y_c\in\{0,\ldots,K_c-1\}.
```

Each output needs a compatible likelihood and posterior interpretation.

---

## 2. Three modeling levels

Heterogeneous systems can be represented at three distinct levels.

### 2.1 Independent submodels

Fit one model per output:

```math
p(f_1,\ldots,f_m\mid\mathcal D)
=
\prod_{j=1}^{m}
p(f_j\mid\mathcal D_j).
```

Each output may use a different likelihood, kernel, input transform, and fitting
procedure.

### 2.2 Correlated latent heterogeneous model

Introduce shared latent functions and task-specific likelihoods.  Outputs share
information statistically.

### 2.3 Objective-space wrapper

Keep the submodels separate, transform each output into a scalar decision
channel, and expose a common tensor interface for acquisitions.

The current `HybridMultiOutputModel` primarily implements the third level.  It
does not by itself introduce cross-output statistical dependence.

---

## 3. Independent heterogeneous submodels

Let output-specific data be

```math
\mathcal D_j
=
\{(x_{ij},y_{ij})\}_{i=1}^{n_j}.
```

The input sets and observation counts may differ across outputs.

Examples:

- strength measured for every specimen;
- failure label available only after a long test;
- ordinal quality assessed by a subset of experts;
- failure mode recorded only when failure occurs.

### Advantages

- supports different likelihoods;
- supports missing and asynchronous outputs;
- permits output-specific noise models;
- permits different kernels and model families;
- easy to diagnose and fit independently;
- avoids negative transfer.

### Limitations

- no cross-output information transfer;
- no cross-output posterior covariance;
- independent joint-event approximation;
- output relationships are used only in the decision layer, if at all.

---

## 4. Correlated latent heterogeneous model

Introduce shared latent GPs

```math
u_q(x)
\sim
\mathcal{GP}(0,k_q),
\qquad q=1,\ldots,Q.
```

For output `j`, define latent predictor

```math
f_j(x)
=
\sum_{q=1}^{Q}a_{jq}u_q(x).
```

The latent covariance is

```math
\mathrm{Cov}[f_j(x),f_l(x')]
=
\sum_{q=1}^{Q}
a_{jq}a_{lq}k_q(x,x').
```

Each output has a task-specific likelihood:

```math
p(\mathbf y\mid\mathbf f)
=
\prod_{j}
\prod_i
p_j(y_{ij}\mid f_j(x_{ij})).
```

Examples:

- Gaussian likelihood for continuous output;
- Bernoulli likelihood for binary output;
- ordered-logit likelihood for ordinal output;
- categorical likelihood for multiclass output;
- Poisson likelihood for count output.

This is a statistical heterogeneous multi-output GP because the outputs share
latent stochastic structure.

---

## 5. Linear model of coregionalization

The shared-latent construction is a Linear Model of Coregionalization (LMC).
For kernels `k_q`,

```math
K_{jl}(x,x')
=
\sum_qB_{jl}^{(q)}k_q(x,x'),
```

where

```math
B^{(q)}
=\mathbf a_q\mathbf a_q^\top
```

for rank-one coregionalization component, or a higher-rank positive-semidefinite
matrix.

LMC can express:

- positive or negative latent association;
- different smoothness components;
- shared global trend plus task-specific variation.

It also introduces identifiability and computational complexity.

---

## 6. Cross-output dependence and likelihoods

Latent correlation does not imply observed-label Pearson correlation.

For binary or ordinal outputs, the likelihood is nonlinear:

```math
f_j\rightarrow p_j(y\mid f_j).
```

The observed dependence is affected by:

- latent covariance;
- link functions;
- class imbalance;
- cutpoints;
- observation noise;
- missingness;
- input distribution.

Therefore raw label correlation is not an estimator of latent task covariance.

---

## 7. Negative transfer

Shared structure helps only when outputs are related in a way the model can
represent.

Negative transfer occurs when shared parameters degrade one or more outputs.
Causes include:

- unrelated responses;
- dependence changing across input regions;
- different relevant input dimensions;
- incompatible smoothness;
- one poorly calibrated likelihood dominating training;
- large output imbalance;
- missing-not-at-random patterns.

Compare independent and correlated models using held-out predictive and
sequential decision metrics.

---

## 8. Missing outputs

Let observation indicator be

```math
m_{ij}
=
\mathbf1[y_{ij}\text{ observed}].
```

The likelihood is

```math
p(\mathbf y_{\mathrm{obs}}\mid\mathbf f)
=
\prod_{i,j:m_{ij}=1}
p_j(y_{ij}\mid f_j(x_i)).
```

Independent submodels naturally handle different `n_j`.  A dense `n x m`
target tensor with fabricated values or naive imputation changes the likelihood
and can bias the model.

A correlated model must support masked likelihood terms or task-indexed
observations.

---

## 9. Asynchronous outputs

Some outputs arrive at different times.  For example:

- quick sensor result;
- later destructive test;
- delayed quality inspection.

The data state is output specific:

```math
\mathcal D_t
=
(\mathcal D_{1,t},\ldots,\mathcal D_{m,t}).
```

A hybrid wrapper should not assume that every output has the same training rows
or pending status.

Decoupled experimental design may choose both input and output subset:

```math
(x,S),
\qquad S\subseteq\{1,\ldots,m\}.
```

The current wrapper unifies posterior channels but does not by itself solve
output-selection or asynchronous cost-aware acquisition.

---

## 10. Decision-space representation

For decision making, each output is mapped to a scalar channel

```math
t_j(x)=T_j[p_j(y_j\mid x,\mathcal D_j)].
```

The combined vector is

```math
\mathbf t(x)
=[t_1(x),\ldots,t_m(x)].
```

Examples:

### Regression

```math
t_j=s_jw_jy_j.
```

### Binary probability

```math
t_j=P(Y_j=c^*\mid x).
```

### Multiclass acceptable-set probability

```math
t_j
=
\sum_{k\in A_j}P(Y_j=k\mid x).
```

### Ordinal expected utility

```math
t_j
=
\sum_k u_{jk}P(Y_j=k\mid x).
```

The transformation converts heterogeneous outputs to comparable tensor
channels, but it does not create statistical dependence.

---

## 11. OutputSpec

`bochan.models.hybrid.OutputSpec` describes one scalar output channel.

Fields include:

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

### `name`

Provides stable output identification and string-based subsetting.

### `task_type`

Current supported values are:

```text
regression
binary
ordinal
multiclass
```

### `output_index`

Selects one channel when a submodel is itself multi-output.

### `sign` and `weight`

Apply maximization direction and linear scale:

```math
t=swy.
```

### `eq_target`

Creates target-distance score

```math
t=-w|y-a|.
```

### `utility_values`

Maps binary, multiclass, or ordinal probabilities to expected utility.

### `positive_class`

Selects class probability for binary or multiclass probability mode.

### `transform`

Applies a custom callable to the transformed mean.  The current variance
handling does not generally compute exact nonlinear transformed moments.

---

## 12. Posterior modes

The hybrid wrapper supports modes such as:

```text
objective
mean
latent
probability
expected_utility
```

The meaning depends on task type.

### Regression

- `mean`: response posterior mean;
- `latent`: latent accessor when available;
- `objective`: sign, weight, target transformation.

### Binary

- `probability`: selected class probability;
- `expected_utility`: binary class utility;
- `latent`: latent GP accessor when supported.

### Multiclass

- `probability`: selected class or configured target;
- `expected_utility`: class-utility expectation;
- `latent`: class-wise latent representation if requested by supported path.

### Ordinal

- `probability`: selected ordered-class probability or configured probability
  mode;
- `expected_utility`: utility expectation;
- `latent`: scalar ordinal latent posterior.

A request for a mode that is not meaningful for a submodel should fail rather
than silently use an unrelated quantity.

---

## 13. Scalar extraction from submodels

A submodel may return

```text
batch_shape x q x m_sub
```

and `output_index` selects one channel:

```text
batch_shape x q
```

The wrapper then appends a hybrid output axis and stacks values:

```text
batch_shape x q x m_hybrid
```

Extra sample-like dimensions from DeepGP or ensemble models may be averaged by
wrapper helpers.  This is an approximation unless the full distribution is
preserved.

---

## 14. Binary probability statistics

For binary class-1 probability `p`, selected class probability is

```math
t=
\begin{cases}
p,&c^*=1,\\1-p,&c^*=0.
\end{cases}
```

If the source posterior exposes Bernoulli variance,

```math
v=p(1-p),
```

that is observation variance of a label, not necessarily posterior uncertainty
of the probability function.

When mapped into `HybridPosterior`, the variance channel should be interpreted
according to the source accessor and wrapper calculation.

---

## 15. Class-utility moments

For class probabilities `p_k` and utility values `u_k`,

```math
\mu_U
=
\sum_kp_ku_k,
```

```math
\sigma_U^2
=
\sum_kp_k(u_k-\mu_U)^2.
```

These are moments of the realized discrete utility conditional on the given
class probabilities.

They do not include posterior epistemic uncertainty in `p_k` unless that
uncertainty is integrated separately.

A normal proxy with these moments approximates a discrete utility distribution.
It does not preserve its finite support or multimodality.

---

## 16. Nonlinear transforms

For nonlinear `h`, exact transformed mean and variance are

```math
\mathbb E[h(Y)],
```

```math
\mathrm{Var}[h(Y)]
=
\mathbb E[h(Y)^2]-\mathbb E[h(Y)]^2.
```

In general,

```math
h(\mathbb E[Y])
\ne
\mathbb E[h(Y)].
```

A first-order delta approximation is

```math
\mathrm{Var}[h(Y)]
\approx
[h'(\mu)]^2\mathrm{Var}(Y).
```

The current custom `OutputSpec.transform` acts on the mean and does not
generally implement exact transformed variance.  Treat such channels as
objective-score proxies unless the transform is linear.

---

## 17. HybridPosterior

`HybridPosterior` stores

```text
mean:     batch_shape x q x m
variance: batch_shape x q x m
```

and returns reparameterized proxy samples

```math
T_j^{(s)}
=
\mu_j+\sqrt{v_j}\epsilon_j^{(s)},
\qquad
\epsilon_j^{(s)}\sim\mathcal N(0,1).
```

### Important limitations

The current posterior does not contain a full covariance matrix across:

- q candidates;
- heterogeneous output channels.

Therefore:

```math
\mathrm{Cov}(T_j,T_l)=0
```

in proxy sampling for distinct elements unless dependence is represented before
conversion in a way retained by the wrapper.

Consequences include approximate:

- joint tail probabilities;
- scalarization variance;
- joint chance constraints;
- q-batch redundancy;
- multi-output information gain;
- hypervolume distributions.

The interface is useful for BoTorch interoperability, but it is not a full
correlated heterogeneous posterior.

---

## 18. Why a normal proxy is useful

A normal proxy provides:

- differentiable `rsample()`;
- compatibility with MC acquisitions;
- common `[...,q,m]` shape;
- sampler dispatcher integration;
- simple marginal uncertainty representation.

It is reasonable when:

- acquisitions mainly use means and marginal scales;
- transformed output distributions are approximately unimodal;
- cross-output correlation is weak or not essential;
- the wrapper is used as a practical decision interface.

It is risky when decisions depend on discrete support, extreme tails, or joint
dependence.

---

## 19. Alternative posterior constructions

### 19.1 Exact sample transformation

Draw samples from each submodel in its native posterior space and apply
`T_j` sample by sample.  This preserves nonlinearity and discrete utility more
faithfully.

### 19.2 Copula coupling

Estimate marginal transformed distributions and connect them with a copula.
This requires dependence estimation and differentiable sampling.

### 19.3 Shared latent heterogeneous GP

Build one joint variational model with task-specific likelihoods.  This is the
most coherent but most complex approach.

### 19.4 Empirical joint residual model

Fit independent submodels and model dependence among calibrated residual or
latent variables.  This is a two-stage approximation.

---

## 20. Input transforms across submodels

A hybrid wrapper may expose one common `input_transform` only when all submodels
share the same transform object.  Otherwise the wrapper delegates transformation
to each submodel.

Potential problems:

- one submodel expects raw `X`, another transformed `X`;
- categorical dimensions differ;
- one model uses PCA or VAE projection;
- one model expands q through input perturbation;
- distance penalties assume one shared transformed space.

Hybrid acquisitions should generally pass raw-space candidates to submodels and
let each model apply its own transform.

---

## 21. Training-data properties

`HybridMultiOutputModel.train_Y` concatenates submodel targets when possible.
This is mainly an interface convenience.

It is statistically meaningful only when:

- rows refer to the same inputs;
- observation counts match;
- target alignment is known.

For missing or asynchronous outputs, submodel training data remain authoritative.

---

## 22. Subsetting outputs

A hybrid model should support selection by:

- integer output index;
- output name;
- list of names or indices.

Subsetting must also subset:

- specifications;
- posterior mean and variance channels;
- output metadata;
- objective transformations.

Stable names reduce mistakes when the output order changes.

---

## 23. Conditioning and fantasies

Look-ahead acquisitions require conditioning on hypothetical observations.
For a hybrid model, this requires output-specific fantasy handling:

- Gaussian regression can use exact conditioning;
- variational classification may rebuild and refit approximately;
- ordinal models may require variational optimization;
- different outputs may be observed asynchronously.

A wrapper cannot provide exact joint fantasies if its submodels do not support
compatible conditioning.

Hybrid look-ahead should therefore document whether conditioning is exact,
approximate, or unsupported.

---

## 24. Calibration across outputs

A common objective vector can hide output-specific calibration problems.
Evaluate each submodel with suitable metrics:

### Regression

- RMSE/MAE;
- NLPD;
- interval coverage.

### Binary

- Brier score;
- log loss;
- calibration curve.

### Multiclass

- multiclass log loss;
- classwise calibration;
- confusion matrix.

### Ordinal

- ranked probability score;
- cumulative calibration;
- mean absolute grade error.

Then evaluate the transformed decision channels themselves.

---

## 25. Independent versus correlated model selection

Prefer independent submodels when:

- output relationships are weak or uncertain;
- missingness patterns differ;
- likelihoods and scales are highly different;
- data are sufficient per output;
- implementation reliability is the priority.

Consider a correlated heterogeneous model when:

- some outputs are data-poor;
- shared physical mechanisms are credible;
- joint probabilities matter;
- cross-output information transfer is validated;
- the added inference complexity is justified.

Always compare against independent baselines.

---

## 26. `bochan` implementation correspondence

### Model wrapper

```text
src/bochan/models/hybrid/multi_output.py
```

contains `HybridMultiOutputModel`.

### Output metadata

```text
src/bochan/models/hybrid/specs.py
```

contains `OutputSpec`, `TaskType`, and `PosteriorMode`.

### Proxy posterior

```text
src/bochan/models/hybrid/posterior.py
```

contains `HybridPosterior` and sampler registration.

### Prediction helpers

```text
src/bochan/models/hybrid/prediction.py
```

contains hybrid prediction utilities.

### Objective support

```text
src/bochan/acquisition/objective/hybrid.py
```

contains objective transformations for hybrid outputs.

### High-level construction

```text
src/bochan/api/factory.py
src/bochan/api/configs.py
```

construct models and specifications from high-level configuration.

### Main current contract

`HybridMultiOutputModel`:

1. receives single-output or selected submodel channels;
2. converts each according to `OutputSpec` and requested mode;
3. stacks scalar means and variances into `[...,q,m]`;
4. returns `HybridPosterior`;
5. does not add cross-output covariance.

---

## 27. Extension directions

Potential future improvements include:

- sample-wise native posterior transformations;
- covariance-aware hybrid posterior;
- heterogeneous shared-latent variational model;
- masked and asynchronous training-data interface;
- decoupled output-selection acquisitions;
- exact nonlinear transformed moments where available;
- calibrated copula dependence;
- output-specific costs and delays;
- joint fantasy support.

---

## 28. New heterogeneous-model checklist

1. What is each output support?
2. Which likelihood is used per output?
3. Are submodels independent or correlated?
4. How are missing outputs represented?
5. What posterior mode is exposed?
6. How is each output transformed to a scalar channel?
7. What does each variance channel mean?
8. Is nonlinear transformation moment-matched or sample based?
9. Is cross-output covariance represented?
10. Is q-point covariance represented?
11. Are input transforms compatible?
12. Is conditioning supported?
13. Which predictive metrics validate each output?
14. Which decision metrics validate the combined model?

---

## 29. References

- Álvarez, Rosasco, and Lawrence, *Kernels for Vector-Valued Functions: A Review*, 2012.
- Moreno-Muñoz et al., *Heterogeneous Multi-output Gaussian Process Prediction*, 2018.
- Rasmussen and Williams, *Gaussian Processes for Machine Learning*, 2006.
