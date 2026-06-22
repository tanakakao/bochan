# 09. Tensor Shapes and Interface Contracts

Tensor shape is part of the mathematics of a BoTorch-compatible component.
Many apparent model or acquisition errors are actually axis errors: a class axis
is mistaken for an output axis, a q-batch is reduced too early, or an
input-perturbation axis is interpreted as additional candidates.

This chapter is the canonical shape reference for `bochan`.

---

## 1. Axis symbols

| Symbol | Meaning |
|---|---|
| `n` | number of training observations |
| `d` | raw input dimension |
| `d_internal` | transformed or projected input dimension |
| `q` | number of nominal candidates selected jointly |
| `m` | number of model-output channels |
| `m_obj` | number of objective channels after transformation |
| `K` | number of classes |
| `n_w` | number of perturbations per nominal candidate |
| `S` | Monte Carlo posterior sample count |
| `F` | fantasy sample count |
| `H` | hyperparameter / fully Bayesian sample count |
| `L` | DeepGP likelihood or hidden sample count, depending on context |
| `batch_shape` | leading t-batch dimensions evaluated independently by the acquisition optimizer |
| `model_batch_shape` | leading dimensions representing batched model parameters or tasks |

The same integer size can appear on different semantic axes.  Shape equality
alone is not enough; axis meaning must be documented.

---

## 2. Training-data shapes

### 2.1 Single-output Gaussian regression

```text
train_X: n x d
train_Y: n x 1
train_Yvar: n x 1        optional known variances
```

Some GPyTorch internals store targets as `n`, but public BoTorch-style wrappers
generally accept `n x 1`.

### 2.2 Multi-output regression

```text
train_X: n x d
train_Y: n x m
train_Yvar: n x m        when supported
```

This may represent:

- independent batched outputs;
- a multitask posterior;
- a ModelList assembled from single-output models.

The same data shape does not imply the same covariance model.

### 2.3 Binary classification

```text
train_X: n x d
train_Y: n               labels 0 or 1
```

Wrappers may accept `n x 1` and squeeze the final singleton axis internally.
Targets are labels, not probabilities unless the model explicitly documents
soft labels.

### 2.4 Multiclass classification

```text
train_X: n x d
train_Y: n               integer labels 0, ..., K-1
```

One-hot targets are not interchangeable with integer labels for every
likelihood.

### 2.5 Ordinal regression

```text
train_X: n x d
train_Y: n               consecutive ordered labels 0, ..., K-1
```

The base ordinal implementation infers class count only when labels are
consecutive and start at zero.

### 2.6 ModelList or hybrid data

Each submodel may own a separate dataset:

```text
train_X_j: n_j x d
train_Y_j: n_j x output_shape_j
```

A combined `train_Y` property is meaningful only when rows are aligned across
outputs.  Missing or asynchronous outputs should remain represented in their
submodels.

---

## 3. Raw and transformed inputs

A model may expose:

```text
train_inputs_raw: original search-space inputs
train_inputs:     model training inputs
```

The exact convention differs by wrapper:

- some wrappers keep `train_inputs` in raw space and transform inside
  `forward()`;
- the base ordinal wrapper stores transformed inputs in `train_inputs` and raw
  inputs in `train_inputs_raw`;
- DKL can keep raw inputs on the wrapper and transformed inputs in the inner GP;
- PCA, REMBO, or VAE wrappers can maintain both raw and latent representations.

Never infer raw versus transformed space only from the attribute name without
checking the model chapter and class implementation.

---

## 4. Candidate input shape

BoTorch acquisition functions normally receive

```text
X: batch_shape x q x d
```

Examples:

### One candidate, no explicit t-batch

```text
X: 1 x d
```

`t_batch_mode_transform` commonly interprets this as one q-batch.

### q candidates, no explicit t-batch

```text
X: q x d
```

### Many restart or raw-sample batches

```text
X: num_restarts x q x d
```

or

```text
X: raw_samples x q x d
```

The leading axis is a t-batch of independent acquisition evaluations, not a
joint candidate axis.

---

## 5. t-batch versus q-batch

For

```text
X.shape = B x q x d
```

- `B` candidate batches are evaluated independently;
- each candidate batch contains `q` jointly valued points.

The acquisition output is

```text
B
```

not

```text
B x q
```

unless the function is a pointwise diagnostic rather than an acquisition ready
for `optimize_acqf`.

A common bug is to average over `B` and preserve `q`, reversing their meanings.

---

## 6. Standard posterior shapes

### 6.1 Single-output posterior

```text
posterior.mean:     batch_shape x q x 1
posterior.variance: batch_shape x q x 1
```

Some custom posteriors expose a squeezed pointwise form

```text
batch_shape x q
```

and acquisition bases normalize it.

### 6.2 Multi-output posterior

```text
posterior.mean:     batch_shape x q x m
posterior.variance: batch_shape x q x m
```

Marginal variance does not contain cross-output covariance.  A full posterior
may internally represent covariance over `q*m` events.

### 6.3 Multiclass probability posterior

```text
class_probs: batch_shape x q x K
```

Here the final axis is class, not generic independent output.  Selecting a class
or computing utility should happen before passing a scalar objective to an
ordinary single-objective acquisition.

### 6.4 Ordinal posterior

The current base ordinal model exposes:

```text
model.posterior(X).mean:       batch_shape x q x 1   latent f
model.class_probs(X):          batch_shape x q x K
model.expected_utility(X, u):  batch_shape x q
```

This differs from binary and multiclass wrappers, whose `posterior()` methods
return probability-space custom posteriors.

---

## 7. Posterior event and batch shapes

A posterior distinguishes:

- batch shape: independent distributions;
- event shape: jointly distributed values.

A multivariate Gaussian over q points may have

```text
batch_shape = B
 event size = q
```

For multi-output models, internal event flattening may represent

```text
q * m
```

while public mean uses

```text
B x q x m
```

Custom posteriors must define:

```python
batch_shape
event_shape
base_sample_shape
batch_range
```

consistently so that BoTorch samplers generate correct base-sample shapes.

---

## 8. Posterior sample shapes

For sample shape

```text
S
```

and multi-output posterior, typical samples are

```text
S x batch_shape x q x m
```

A scalar MC objective returns

```text
S x batch_shape x q
```

A multi-objective MC objective returns

```text
S x batch_shape x q x m_obj
```

The acquisition then reduces sample and q axes according to its value function.

Sample axes should not be identified only by their size.  Use the sampler's
`sample_shape` and known trailing q/output axes.

---

## 9. Fully Bayesian model axes

Suppose there are `H` posterior hyperparameter samples.  A posterior may expose

```text
H x batch_shape x q x m
```

before MC sampling, and

```text
S x H x batch_shape x q x m
```

after MC sampling.

BoTorch ensemble-aware acquisitions may average over the model-batch axis.
Custom code that repeatedly averages leading axes until a desired rank can
silently collapse a real t-batch.  Axis reduction should use explicit expected
shapes.

---

## 10. DeepGP sample axes

DeepGP prediction may introduce a leading likelihood-sample or hidden-function
sample axis:

```text
L x batch_shape x q x m
```

A wrapper may:

- retain the sample mixture;
- moment-match it to one posterior;
- average means and include between-sample variance;
- incorrectly average only variances.

The reduction policy must be documented.  Acquisition helpers that reduce extra
leading dimensions should preserve t-batch, q, output, and class axes.

---

## 11. ModelList shapes

A ModelList returns a posterior that logically combines submodel outputs.  If
all submodels are scalar:

```text
posterior.mean: batch_shape x q x m
```

where `m` is number of submodels.

Submodels may have different internal likelihoods, but the combined posterior
must present compatible output dimensions to the objective.

For heterogeneous submodels, raw outputs should not be blindly concatenated.
`HybridMultiOutputModel` transforms each output to a scalar mode before stacking.

---

## 12. HybridPosterior shapes

The current `HybridPosterior` stores

```text
mean:     batch_shape x q x m
variance: batch_shape x q x m
```

and samples

```text
S x batch_shape x q x m
```

using independent normal proxy noise per element.

It does not store a full covariance matrix across:

- q points;
- heterogeneous output channels.

Therefore shape compatibility does not imply exact joint dependence.

---

## 13. Objective shapes

### Scalar MC objective

Input:

```text
S x batch_shape x q x m
```

Output:

```text
S x batch_shape x q
```

### Multi-objective MC objective

Output:

```text
S x batch_shape x q x m_obj
```

### Pointwise deterministic score objective

Input:

```text
batch_shape x q_like
```

Output:

```text
batch_shape x q
```

when it aggregates perturbations.

### Already aggregated score

Some score objectives receive

```text
batch_shape
```

or

```text
batch_shape x q
```

and should not attempt a second `n_w` reduction.  Implementations use shape
checks and an `aggregated_risk_mode` or equivalent policy.

---

## 14. Acquisition output shape

For acquisition input

```text
batch_shape x q x d
```

expected output is

```text
batch_shape
```

Examples:

```text
X: 32 x 1 x d   -> acq: 32
X: 10 x 3 x d   -> acq: 10
X: 1 x q x d    -> acq: 1
X: q x d        -> scalar or length-1 after t-batch normalization
```

Returning `batch_shape x q` causes `optimize_acqf` to interpret pointwise values
incorrectly.

---

## 15. Input perturbation shapes

Original candidate:

```text
X: batch_shape x q x d
```

Expanded input:

```text
X_tilde: batch_shape x (q * n_w) x d
```

Pointwise posterior mean:

```text
batch_shape x (q * n_w) x m
```

After reshaping:

```text
batch_shape x q x n_w x m
```

Posterior sample form:

```text
S x batch_shape x q x n_w x m
```

Possible reduction order:

```text
output transform
    -> perturbation risk reduction over n_w
    -> q-batch value reduction
    -> posterior sample mean
```

The correct order depends on the acquisition definition.

---

## 16. InputPerturbation raw-space alignment

For mixed models, code may compare transformed categorical columns with raw
categories.  When evaluation transforms expand q to `q*n_w`, raw `X` must be
repeated for comparison:

```text
raw X:       batch_shape x q x d
aligned raw: batch_shape x (q * n_w) x d
```

The relevant operation is typically

```python
X.repeat_interleave(n_w, dim=-2)
```

This alignment is for validation or distance calculation.  It does not mean
categories themselves are perturbed.

---

## 17. Joint covariance shapes

For single-output q-batch posterior covariance:

```text
batch_shape x q x q
```

For an expanded input with perturbations:

```text
batch_shape x (q * n_w) x (q * n_w)
```

To aggregate covariance of perturbation means, reshape into blocks:

```text
batch_shape x q x n_w x q x n_w
```

and average both perturbation axes:

$$
[\Sigma_q]_{ij}
=
\frac1{n_w^2}
\sum_{r=1}^{n_w}
\sum_{s=1}^{n_w}
\Sigma_{(i,r),(j,s)}.
$$

A diagonal-only approximation instead averages marginal variances and discards
cross-candidate and cross-perturbation covariance.

---

## 18. Multitask covariance layouts

GPyTorch may use:

- `MultitaskMultivariateNormal`;
- interleaved task/event layouts;
- batch-of-independent-task GPs;
- Kronecker linear operators.

Public mean commonly uses

```text
batch_shape x q x m
```

but covariance may flatten to

```text
batch_shape x (q * m) x (q * m)
```

The flattening order must be known before selecting output blocks.  Do not
reshape covariance based only on total element count unless the interleaving
convention is verified.

---

## 19. Multiclass latent batch axis

The multiclass latent SVGP uses class-wise batch shape

```text
K
```

for inducing points and kernels.  Acquisition input may be

```text
batch_shape x q x d
```

The wrapper inserts a singleton class-batch axis:

```text
batch_shape x 1 x q x d
```

which broadcasts internally to

```text
batch_shape x K x q x d
```

The probability posterior then returns

```text
batch_shape x q x K
```

This class-batch axis is model structure, not a t-batch to average blindly.

---

## 20. Ordinal boundary shapes

For `K` ordinal classes there are

```text
K - 1
```

cutpoints.  Boundary-wise scores may have

```text
batch_shape x q x (K - 1)
```

before boundary reduction.

`target_boundary_idx` selects one cutpoint.  It does not select a class-probability
column directly.

After boundary reduction:

```text
batch_shape x q
```

then q reduction produces

```text
batch_shape
```

---

## 21. Multi-output reductions

Suppose score is

```text
batch_shape x q x m
```

There are two independent reductions.

### Output reduction

```text
mean / sum / max / min over m
```

produces

```text
batch_shape x q
```

### q reduction

```text
mean / sum / max / min over q
```

produces

```text
batch_shape
```

Reversing the order can change the result for nonlinear reductions:

$$
\max_q\operatorname{mean}_m a_{qm}
\ne
\operatorname{mean}_m\max_q a_{qm}.
$$

The acquisition definition must state the order.

---

## 22. Constraints and candidate tensors

Linear constraints are usually defined over feature indices of each candidate.
For q-batches, check whether a constraint applies:

- independently to each point;
- to the sum across q points;
- across selected feature groups;
- after rounding or repair;
- in raw or normalized space.

A candidate repair function should preserve

```text
batch_shape x q x d
```

and not collapse t-batch or q axes.

---

## 23. Pending and observed reference shapes

Reference points may be supplied as

```text
n_ref x d
```

or with compatible batch shape.  Distance penalties commonly flatten reference
points to

```text
N_total x d_internal
```

after model-consistent input transformation.

Pending and observed tensors should be detached from the candidate gradient
graph.  They are constants during acquisition optimization.

---

## 24. Common shape failures

### Size mismatch between output scales and length scales

A task scale vector of shape `m` cannot multiply an unprocessed length-scale
tensor such as

```text
num_terms x batch_shape x 1 x d
```

until kernel-component and task axes are identified.

### Constant posterior from wrong input transform

Applying a transform twice or mixing raw and transformed training inputs can
make all candidates appear identical.

### Class axis averaged as DeepGP sample axis

Repeatedly averaging leading dimensions can destroy multiclass structure.

### q and output axis swapped

A tensor `B x m x q` reshaped as `B x q x m` without permuting changes data
association.

### `q*n_w` mistaken for q

The acquisition may return too many pointwise scores or treat perturbations as
independent candidates.

### Covariance reshaped by element count alone

Interleaved multitask covariance can be semantically corrupted even when the
reshape has the same number of elements.

### Objective collapses q

An MC objective that returns `S x B` instead of `S x B x q` prevents the
acquisition from computing its own batch utility.

---

## 25. Debugging procedure

Print or assert shapes in this order:

```text
raw X
transformed X
posterior.mean
posterior.variance
posterior covariance or distribution batch/event shape
posterior samples
objective(samples)
pointwise acquisition score
score after perturbation reduction
score after output/class/boundary reduction
final acquisition value
repaired candidates
```

Also record semantic labels for each axis.  A useful debugging note is:

```text
samples: [S, H, B, q, m]
```

rather than only

```text
samples.shape == (128, 8, 32, 3, 2)
```

---

## 26. Runtime assertion patterns

Useful checks include:

```python
assert X.shape[-1] == input_dim
assert posterior.mean.shape[-2] == X.shape[-2]
assert objective_values.shape[-1] == X.shape[-2]
assert acq_value.shape == X.shape[:-2]
```

For multi-output posterior:

```python
assert posterior.mean.shape[-1] == num_outputs
```

For multiclass probability:

```python
assert probs.shape[-1] == num_classes
assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs[..., 0]))
```

For perturbation:

```python
assert q_expanded == q * n_w
```

Assertions should be placed near the interface where axis meaning is known.

---

## 27. `bochan` implementation correspondence

### 27.1 Model wrappers

Model-specific conventions are implemented under

```text
src/bochan/models/
```

Important examples:

- binary base model normalizes label and probability shapes;
- multiclass base model inserts a class-batch axis;
- ordinal base model tracks raw and transformed training inputs;
- DeepGP wrappers reduce or moment-match extra sample axes;
- high-dimensional wrappers map raw to latent dimensions;
- hybrid wrappers stack scalar transformed outputs.

### 27.2 Shape helpers

Relevant helper modules include:

```text
src/bochan/models/components/heteroscedastic.py
src/bochan/models/components/multiclass.py
src/bochan/models/hybrid/posterior.py
```

Task-specific acquisition bases also contain shape-alignment functions for
pointwise scores, extra leading sample dimensions, covariance extraction, and
q-batch finalization.

### 27.3 BoTorch decorators

Many acquisitions use

```python
@t_batch_mode_transform()
```

which normalizes candidate inputs to t-batch form and checks expected q where
specified.  It does not resolve custom output, class, or perturbation axes.

### 27.4 Posterior samplers

Custom posteriors may require registration with BoTorch's sampler dispatcher.
`HybridPosterior` registers a `SobolQMCNormalSampler` path because baseline
pruning and MC acquisitions call automatic sampler resolution.

### 27.5 Registry and high-level API

```text
src/bochan/api/model_registry.py
src/bochan/api/acquisition_registry.py
```

resolve names, but they do not validate every semantic shape assumption.  The
resolved model and acquisition classes remain responsible for their contracts.

---

## 28. New-component shape template

Every model should document:

```text
train_X:
train_Y:
raw input X:
internal input X:
posterior mean:
posterior variance:
posterior samples:
model batch axes:
output or class axes:
observation_noise behavior:
```

Every objective should document:

```text
input sample/score shape:
output shape:
reduced axes and order:
q*n_w behavior:
```

Every acquisition should document:

```text
X shape:
posterior accessor:
pointwise score shape:
class/output/boundary reductions:
q reduction:
final shape:
```

---

## 29. Minimal test matrix

Test at least:

1. `q=1`, no t-batch;
2. `q>1`, no t-batch;
3. multiple t-batches;
4. single and multiple outputs;
5. input perturbation with `n_w>1`;
6. mixed input;
7. pending and observed points;
8. DeepGP or extra sample axis;
9. fully Bayesian model batch where supported;
10. objective and posterior transform;
11. gradient backpropagation to `X`;
12. final acquisition shape expected by `optimize_acqf`.

Shape tests are part of theoretical correctness because they determine which
random variables and candidate sets are being reduced.
