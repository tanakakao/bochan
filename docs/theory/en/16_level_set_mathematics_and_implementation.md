# 16. Level-set Acquisition Formulas and Implementation Reference

Chapter 05 defines Level-set Estimation problems, losses, confidence sets, and
evaluation.  This chapter is the implementation reference: it records the
actual score formulas, posterior spaces, reductions, and source classes used by
current `bochan` LSE acquisitions.

The class implementation is authoritative when a literature name has several
variants.

---

## 1. Common notation

For candidate batch

```math
X=[x_1,\ldots,x_q],
```

let

```math
\mu_i=\mu(x_i),
\qquad
v_i=\mathrm{Var}[f(x_i)],
\qquad
\sigma_i=\sqrt{v_i}.
```

Let threshold be `h`.  For classification probability, write

```math
p_i=P(Y\in A\mid x_i,\mathcal D)
```

for a target class or acceptable class set.  For ordinal models, cutpoints are

```math
c_0<\cdots<c_{K-2}.
```

A pointwise acquisition constructs

```text
batch_shape x q_like
```

then applies:

```text
pointwise score
    -> duplicate / pending / observed penalties
    -> optional score objective
    -> q*n_w to q perturbation reduction
    -> q reduction
    -> batch_shape acquisition value
```

A joint acquisition constructs one value from the q-dimensional posterior
covariance before applying batch-level penalties.

---

## 2. Regression LSE implementation family

Main source:

```text
src/bochan/acquisition/regression/levelset_estimation/single_output.py
```

Multi-output and heteroscedastic variants are in the same package.

### 2.1 Posterior accessor

Regression LSE uses

```python
posterior = model.posterior(X)
```

and extracts response or latent mean and variance according to the regression
model's posterior contract.  Multiple output channels can be reduced through
`output_reduction`.

---

## 3. `qRegressionStraddle`

Implemented pointwise score:

```math
s_i
=
\beta\sigma_i-|\mu_i-h|.
```

Code correspondence:

```python
mean, var, Xt = self._posterior_mean_variance(X)
std = var.sqrt()
score = beta * std - (mean - threshold).abs()
```

Interpretation:

- `beta * sigma`: uncertainty reward;
- `-|mu-h|`: boundary proximity;
- the implementation parameter `beta` multiplies standard deviation directly;
- `h` can also be supplied through the alias argument used by the base.

Final pointwise scores are penalized, optionally risk aggregated, and reduced
over q.

---

## 4. `qRegressionJointStraddle`

Let posterior covariance across the q-batch be

```math
\Sigma_X\in\mathbb R^{q\times q}.
```

Boundary-proximity term is

```math
D(X)
=
\frac1q
\sum_{i=1}^{q}|\mu_i-h|.
```

Implemented score:

```math
s(X)
=-D(X)+\beta U(\Sigma_X).
```

Supported uncertainty functions are:

### Trace

```math
U_{\mathrm{trace}}(\Sigma)
=
\mathrm{tr}(\Sigma).
```

### Log determinant

```math
U_{\log\det}(\Sigma)
=
\log\det(\Sigma+\epsilon I).
```

### Log determinant of identity plus covariance

```math
U_{\log\det1p}(\Sigma)
=
\log\det(I+\Sigma+\epsilon I).
```

The log-determinant variants reward joint uncertainty volume and penalize
redundancy through covariance.

---

## 5. `qRegressionICU`

The current ICU-style local score is

```math
s_i
=
\exp\left[
-\frac12
\left(
\frac{\mu_i-h}{b_i}
\right)^2
\right]
\sigma_i.
```

If no bandwidth is supplied,

```math
b_i=\sigma_i.
```

With fixed `bandwidth`,

```math
b_i=b.
```

This is a local contour-weighted uncertainty proxy.  It is not a full numerical
integration of expected global contour-loss reduction.

---

## 6. `qRegressionBoundaryVariance`

Boundary weight:

```math
w_i
=
\exp\left[
-\frac12
\left(
\frac{\mu_i-h}{\tau}
\right)^2
\right].
```

Score:

```math
s_i=v_iw_i.
```

`tau` controls the latent-response width of the boundary neighborhood.

---

## 7. `qRegressionProbabilityOfExceedance`

For Gaussian posterior and `mode="above"`, the exact normal-CDF path is

```math
s_i
=
P(f_i\ge h)
=
\Phi\left(
\frac{\mu_i-h}{\sigma_i}
\right).
```

For `mode="below"`,

```math
s_i
=
\Phi\left(
\frac{h-\mu_i}{\sigma_i}
\right).
```

For interval `[l,u]`,

```math
s_i
=
\Phi\left(
\frac{u-\mu_i}{\sigma_i}
\right)
-
\Phi\left(
\frac{l-\mu_i}{\sigma_i}
\right).
```

When `temperature` is supplied, the implementation uses smooth sigmoid
approximations in mean space rather than Gaussian posterior CDFs.

Probability of exceedance is a membership score.  It tends to reward points
already likely to be above the threshold and is not automatically a
boundary-learning criterion.

---

## 8. Regression score objective

`RegressionLevelSetScoreObjective` supports perturbation aggregation.

Input score:

```text
... x (q * n_w)
```

Reshape:

```text
... x q x n_w
```

### Mean

```math
\bar s_i
=
\frac1{n_w}
\sum_{r=1}^{n_w}s_{ir}.
```

### VaR path

Scores are sorted according to `maximize`; the selected tail size is

```math
k=\lceil\alpha n_w\rceil.
```

The boundary tail element is returned.

### CVaR path

The selected tail mean is returned.

This operates on the already constructed LSE score.  It is score-level risk
aggregation, not necessarily LSE of a robust latent response.

---

## 9. Regression penalties

The regression base supports:

- same-batch soft penalty;
- pending-point soft penalty;
- observed-point soft penalty;
- hard duplicate penalty.

For transformed points `z_i`, a same-batch term is based on

```math
\exp[-\eta\|z_i-z_j\|^2].
```

Reference penalties use distance to the nearest transformed reference point.
The implementation prefers `model.transform_inputs()` so wrapper-specific raw
to internal mappings are respected.

---

## 10. Binary LSE implementation family

Main source:

```text
src/bochan/acquisition/binary/levelset_estimation/single_output.py
```

The binary acquisition base obtains a latent posterior through
`latent_posterior()` or a supported fallback.  Probability-space criteria use
`probability_posterior()` when available or `posterior()`.

---

## 11. `qBinaryLatentStraddleAcquisition`

Latent posterior:

```math
f_i\mid\mathcal D
\sim
\mathcal N(\mu_i,v_i).
```

Implemented smoothed score:

```math
s_i
=
\beta\sigma_i
-
\sqrt{(\mu_i-h_f)^2+10^{-8}}.
```

Default latent threshold is

```math
h_f=0.
```

This targets the latent decision boundary, not a probability threshold unless
the link and threshold correspondence are specified.

---

## 12. `qBinaryJointLatentStraddleAcquisition`

Joint score:

```math
s(X)
=
\beta U(\Sigma_f)
-D(\boldsymbol\mu_f,h_f).
```

Supported uncertainty modes:

### `logdet1p`

```math
U(\Sigma)
=
\frac12
\log\det
\left(
I+\frac{\Sigma}{\tau^2}
\right).
```

### `logdet`

```math
U(\Sigma)
=
\frac12\log\det(\Sigma).
```

### `sqrt_trace`

```math
U(\Sigma)
=
\sqrt{\mathrm{tr}(\Sigma)}.
```

Supported boundary distances:

```math
D_{\mathrm{mean\ abs}}
=
\frac1q\sum_i|\mu_i-h_f|,
```

```math
D_{\mathrm{l2\ mean}}
=
\sqrt{
\frac1q\sum_i(\mu_i-h_f)^2
},
```

```math
D_{\mathrm{max\ abs}}
=
\max_i|\mu_i-h_f|.
```

If `marginalize_pending=True`, the implementation evaluates the incremental
joint score:

```math
s(X_{\mathrm{pending}}\cup X)
-
s(X_{\mathrm{pending}}).
```

This is a joint-score difference, not full fantasy conditioning on pending
outcomes.

---

## 13. `qBinaryICUAcquisition`

The probability posterior mean is converted to

```math
p_i\in(0,1).
```

Implemented score:

```math
s_i
=4p_i(1-p_i).
```

This score is one at `p=0.5` and zero at `p=0` or `1`.  It is a normalized
binary boundary ambiguity score.

Despite the ICU name, this class does not integrate global contour-loss
reduction.

---

## 14. `qBinaryBoundaryVarianceAcquisition`

Uses latent mean and variance.  Boundary kernel weight is a Gaussian function
centered at latent threshold:

```math
w_i
=
\exp\left[
-\frac12
\left(
\frac{\mu_i-h_f}{\tau}
\right)^2
\right].
```

Score:

```math
s_i=v_iw_i.
```

---

## 15. `qBinaryClassEntropyAcquisition`

For binary probability `p_i`, score is Bernoulli entropy:

```math
s_i
=-p_i\log p_i
-(1-p_i)\log(1-p_i).
```

This is predictive class ambiguity.  It is not the same as latent posterior
variance or BALD.

---

## 16. Multiclass LSE implementation family

Main source:

```text
src/bochan/acquisition/multiclass/levelset_estimation/single_output.py
```

The current family reduces multiclass output to a target probability

```math
p_T(x)
=
\mathrm{class\_reduce}
\{p_k(x):k\in T\}.
```

The level set is defined in probability space:

```math
p_T(x)=h_p.
```

The historical class name contains `LatentStraddle`, but the implemented
single-output score is target-probability Straddle.

---

## 17. Multiclass target-class reduction

`target_class` can select one class or a sequence.  `class_reduction` determines
how selected probabilities are combined.

For an acceptable union of mutually exclusive classes, the probabilistically
meaningful quantity is

```math
p_T=\sum_{k\in T}p_k.
```

Mean or max reductions are scores with different interpretations.

---

## 18. Multiclass uncertainty modes

Let posterior target-probability samples be

```math
p_i^{(s)}.
```

### `bernoulli`

```math
u_i
=
\sqrt{p_i(1-p_i)}.
```

This treats target-set membership as a binary observation.

### `posterior`

```math
u_i
=
\mathrm{Std}_s[p_i^{(s)}].
```

This measures posterior variation of the target probability.

### `combined`

```math
u_i
=
\sqrt{
\mathrm{Var}_s[p_i^{(s)}]
+p_i(1-p_i)
}.
```

This combines probability-function uncertainty with Bernoulli observation
ambiguity.

---

## 19. `qMulticlassLatentStraddleAcquisition`

Implemented probability-space score:

```math
s_i
=
\beta u_i-|p_i-h_p|.
```

`u_i` is selected by `uncertainty_mode`.

The class name is retained for API compatibility, but theory documentation
should call it target-probability Straddle.

---

## 20. `qMulticlassJointLatentStraddleAcquisition`

Target-probability samples across q are used to estimate sample covariance:

```math
\widehat\Sigma_p
=
\frac1{S-1}
\sum_s
(\mathbf p^{(s)}-\bar{\mathbf p})
(\mathbf p^{(s)}-\bar{\mathbf p})^\top
+\epsilon I.
```

Implemented joint score:

```math
s(X)
=
\beta U(\widehat\Sigma_p)
-D(\bar{\mathbf p},h_p).
```

Uncertainty modes:

- `logdet1p`;
- `logdet`;
- `sqrt_trace`;
- `trace`.

Boundary-distance modes:

- `mean_abs`;
- `l2_mean`;
- `max_abs`.

Pending marginalization uses the same incremental joint-score difference as the
binary joint class.

---

## 21. `qMulticlassICUAcquisition`

Contour weight:

```math
w_i
=
\exp\left[
-\frac12
\left(
\frac{p_i-h_p}{b}
\right)^2
\right].
```

Score:

```math
s_i
=u_i^2w_i.
```

The uncertainty `u_i` can use Bernoulli, posterior, or combined mode.

---

## 22. `qMulticlassBoundaryVarianceAcquisition`

Boundary weight uses exponential absolute distance:

```math
w_i
=
\exp\left(
-\frac{|p_i-h_p|}{b}
\right).
```

Score:

```math
s_i
=u_i^2w_i.
```

This differs from the Gaussian squared-distance boundary weight used in ICU.
The actual helper `_boundary_weight` defines the formula.

---

## 23. `qMulticlassClassEntropyAcquisition`

Without target-class selection, score is full categorical entropy:

```math
s_i
=-\sum_{k=0}^{K-1}p_{ik}\log p_{ik}.
```

With selected target classes, the current code selects/reduces probabilities and
computes

```math
s_i=-p_T\log p_T.
```

This selected-class path is not full binary entropy because it omits

```math
-(1-p_T)\log(1-p_T).
```

That distinction should be considered when interpreting target-specific class
entropy.

---

## 24. `qMulticlassProbabilityOfExceedance`

Implemented smooth score:

```math
s_i
=
\sigma\left(
\frac{p_i-h_p}{\tau}
\right).
```

This is a smooth probability-space membership score, not posterior probability
that an uncertain probability function exceeds the threshold.

`qMulticlassLevelSetUncertainty` is an alias subclass of the multiclass ICU
implementation.

---

## 25. Ordinal LSE implementation family

Main source:

```text
src/bochan/acquisition/ordinal/levelset_estimation/single_output.py
```

The base obtains:

- latent posterior through `model.posterior(X)`;
- cutpoints from `ordinal_likelihood` or `likelihood`;
- class probabilities by sampling latent `f` and applying ordered-logit
  probabilities.

Cutpoints are detached and stored as buffers during acquisition optimization,
so only candidate `X` is optimized.

---

## 26. Ordinal boundary indexing

For `K` classes there are `K-1` cutpoints.  Index `j` is the boundary between
class `j` and class `j+1`.

For classes `0/1/2`:

```text
target_boundary_idx = 0 -> boundary 0/1
target_boundary_idx = 1 -> boundary 1/2
```

Boundary-wise tensor shape is

```text
batch_shape x q_like x (K - 1)
```

before reduction.

---

## 27. Ordinal boundary reduction

Given boundary scores `s_ij`, the implementation supports:

### Specific boundary

```math
s_i=s_{i,j^*}.
```

### Weighted sum

```math
s_i
=
\sum_jw_js_{ij}.
```

### Mean, max, or min

```math
s_i
=
\mathrm{reduce}_j(s_{ij}).
```

If `target_boundary_idx` is supplied, boundary weights and reduction are
ignored.

---

## 28. `qOrdinalLatentStraddleAcquisition`

Boundary-specific distance:

```math
d_{ij}=|\mu_i-c_j|.
```

Boundary score:

```math
s_{ij}
=
\beta\sigma_i-d_{ij}.
```

After boundary selection or reduction, pointwise penalties and optional score
objective are applied, then q is reduced by sum or mean.

The single-point private variant uses nearest-cutpoint distance:

```math
\min_j|\mu_i-c_j|.
```

The public q variant supports explicit boundary control.

---

## 29. `qOrdinalJointLatentStraddleAcquisition`

Input perturbation may first reduce expanded mean and covariance to nominal q.

Joint uncertainty is:

### Trace

```math
U(\Sigma)=\mathrm{tr}(\Sigma).
```

### Logdet mode

The current implementation computes

```math
U(\Sigma)
=
\frac12
\log\det
\left(
I+rac{\Sigma}{\tau^2}
\right).
```

Boundary contribution uses negative cutpoint distance, reduced over boundaries
and averaged over q:

```math
B(X)
=
\frac1q
\sum_i
\mathrm{boundary\_reduce}_j[-|\mu_i-c_j|].
```

Final score:

```math
s(X)=\beta U(\Sigma)+B(X)-P(X).
```

`P(X)` includes configured same-batch, pending, and observed reference
penalties.

---

## 30. `qOrdinalICUAcquisition`

From predictive class probabilities, define cumulative upper-class probability
for boundary `j`:

```math
g_{ij}
=P(Y_i\ge j+1)
=
\sum_{k=j+1}^{K-1}p_{ik}.
```

Boundary ambiguity is

```math
u_{ij}
=4g_{ij}(1-g_{ij}).
```

The acquisition selects or reduces `u_ij` across boundaries, subtracts
pointwise repulsion, applies the score objective, and reduces q.

This is an ordinal boundary-ambiguity score; it is not a global integrated
contour-loss calculation.

---

## 31. `qOrdinalBoundaryVarianceAcquisition`

For cutpoint `c_j`, Gaussian boundary kernel weight is

```math
w_{ij}
=
\exp\left[
-\frac12
\left(
\frac{\mu_i-c_j}{\tau}
\right)^2
\right].
```

Boundary score:

```math
s_{ij}=v_iw_{ij}.
```

Scores are selected or reduced across boundaries.

The legacy `reduce` argument maps to boundary reduction `sum` or `max`.

---

## 32. `qOrdinalClassEntropyAcquisition`

Predictive class probabilities are estimated by latent posterior sampling and
ordered-logit conversion.  Score is

```math
s_i
=-\sum_{k=0}^{K-1}p_{ik}\log p_{ik}.
```

This measures full grade ambiguity.  It does not target a specific cutpoint.

---

## 33. Ordinal score objective

The ordinal LSE score objective supports:

- multiplicative `sign` and `weight`;
- `q*n_w -> q` mean aggregation;
- VaR or CVaR tail reduction;
- `maximize` direction;
- `alpha` tail size.

As with regression, risk is applied to the computed acquisition score unless
the acquisition itself constructs a robust latent target.

---

## 34. Multi-output LSE wrappers

Task packages provide multi-output classes, including registered names such as:

```text
qMultiOutputRegressionStraddle
qMultiOutputRegressionJointStraddle
qMultiOutputBinaryLatentStraddleAcquisition
qMultiOutputBinaryJointLatentStraddleAcquisition
qMultiOutputMulticlassLatentStraddleAcquisition
qMultiOutputOrdinalLatentStraddleAcquisition
```

A multi-output implementation typically:

1. obtains output-wise posterior or probability scores;
2. retains shape `... x q x m`;
3. reduces outputs with configured rule or objective;
4. reduces q;
5. returns t-batch shape.

Output mean, sum, max, and min are score reductions.  They do not automatically
represent intersection or union membership probabilities.

---

## 35. Heteroscedastic LSE wrappers

Registered heteroscedastic families include regression, binary, multiclass, and
ordinal single- and multi-output classes.

Examples:

```text
qHeteroRegressionStraddle
qHeteroBinaryLatentStraddleAcquisition
qHeteroMulticlassICUAcquisition
qHeteroOrdinalBoundaryVarianceAcquisition
qHeteroMultiOutputOrdinalLevelSetUncertainty
```

The heteroscedastic wrapper can combine base score with a predicted noise or
reliability term by configured weighting or combination modes.

Interpretation depends on the concrete module:

- noise penalty can avoid irreducibly noisy measurements;
- noise uncertainty can value learning the noise process;
- adding noise to probability variance is an engineering convention, not a
  fully specified noisy-label likelihood;
- a noise-aware score is not automatically LSE of future-observation
  reliability.

Chapter 13 defines these distinctions.

---

## 36. Input-perturbation covariance reduction

Joint ordinal LSE contains explicit reduction from expanded covariance

```text
... x (q*n_w) x (q*n_w)
```

to nominal covariance

```text
... x q x q
```

### `block_mean`

Reshape covariance as

```text
... x q x n_w x q x n_w
```

and average both perturbation axes.  This approximates covariance of perturbation
means.

### `diagonal_mean`

Average marginal variances inside each perturbation group and build a diagonal
q covariance.  This discards cross-candidate covariance.

A jitter term is added after symmetrization.

---

## 37. Distance penalties and perturbation groups

Ordinal pointwise same-batch penalty excludes pairs belonging to the same
nominal candidate when q has been expanded by `n_w`.  Otherwise perturbation
replicas of one candidate would repel each other.

Group index is

```math
g(r)=\left\lfloor\frac{r}{n_w}\right\rfloor.
```

Pairs with the same group are masked from duplicate penalty.

This logic is required only when the distance calculation sees expanded
perturbation points.

---

## 38. Pending and observed reference handling

Reference inputs are detached and transformed into the same model distance
space as candidates.

The code supports:

- Tensor;
- list or tuple of tensors;
- flattening multiple reference batches;
- first-submodel transform fallback for some wrappers.

Pending penalties discourage already launched conditions.  Observed penalties
discourage remeasurement.  Both should be disabled when replication is
scientifically valuable.

---

## 39. q reduction

Pointwise classes support q reduction such as:

```math
\mathrm{mean}_i s_i,
\qquad
\sum_i s_i,
\qquad
\max_i s_i,
\qquad
\min_i s_i
```

according to family-specific accepted modes.

`sum` makes acquisition magnitude grow with q.  `mean` is more comparable
across batch sizes but can undervalue adding one highly useful point.  `max`
can ignore diversity.  Joint classes should be preferred when covariance-aware
batch value is required.

---

## 40. Class, boundary, output, and q reduction order

A common pointwise ordinal order is:

```text
latent or probability values
    -> boundary-wise score
    -> boundary reduction
    -> pointwise penalties
    -> perturbation objective
    -> q reduction
```

A common multi-output order is:

```text
output-wise point score
    -> output reduction
    -> perturbation reduction
    -> q reduction
```

Nonlinear reductions do not commute.  The implementation order must be used
when reproducing results.

---

## 41. Ensemble and extra sample dimensions

Some public methods use

```python
@average_over_ensemble_models
```

which averages acquisition values across model ensemble batches according to
BoTorch conventions.

DeepGP and probability-sampling paths can leave extra leading sample dimensions.
Helper functions reduce those dimensions while attempting to preserve t-batch,
q, class, and output axes.

Shape tests should include:

- ordinary exact GP;
- variational classifier;
- DeepGP;
- fully Bayesian or ensemble batch;
- `q>1`;
- `n_w>1`.

---

## 42. Registry inventory

High-level public registration is in

```text
src/bochan/api/registry/acquisition.py
```

### Regression

- Straddle;
- Joint Straddle;
- ICU;
- Boundary Variance;
- Probability of Exceedance;
- multi-output and heteroscedastic variants.

### Binary

- Latent Straddle;
- Joint Latent Straddle;
- ICU;
- Boundary Variance;
- Class Entropy;
- multi-output and heteroscedastic variants.

### Multiclass

- target-probability Straddle;
- joint target-probability Straddle;
- ICU;
- Boundary Variance;
- Class Entropy;
- Probability of Exceedance;
- Level-set Uncertainty alias;
- multi-output and heteroscedastic variants.

### Ordinal

- Latent Straddle;
- Joint Latent Straddle;
- ICU;
- Boundary Variance;
- Class Entropy;
- multi-output and heteroscedastic variants.

---

## 43. Source map

| Family | Source |
|---|---|
| Regression single-output | `src/bochan/acquisition/regression/levelset_estimation/single_output.py` |
| Regression multi-output | `src/bochan/acquisition/regression/levelset_estimation/multi_output.py` |
| Regression heteroscedastic | `src/bochan/acquisition/regression/levelset_estimation/hetero_*.py` |
| Binary single-output | `src/bochan/acquisition/binary/levelset_estimation/single_output.py` |
| Binary multi-output | `src/bochan/acquisition/binary/levelset_estimation/multi_output.py` |
| Binary heteroscedastic | `src/bochan/acquisition/binary/levelset_estimation/hetero_*.py` |
| Multiclass single-output | `src/bochan/acquisition/multiclass/levelset_estimation/single_output.py` |
| Multiclass multi-output | `src/bochan/acquisition/multiclass/levelset_estimation/multi_output.py` |
| Multiclass heteroscedastic | `src/bochan/acquisition/multiclass/levelset_estimation/hetero_*.py` |
| Ordinal single-output | `src/bochan/acquisition/ordinal/levelset_estimation/single_output.py` |
| Ordinal multi-output | `src/bochan/acquisition/ordinal/levelset_estimation/multi_output.py` |
| Ordinal heteroscedastic | `src/bochan/acquisition/ordinal/levelset_estimation/hetero_*.py` |
| Non-Gaussian regression | `src/bochan/acquisition/non_gaussian/levelset_estimation/` |
| Public registry | `src/bochan/api/registry/acquisition.py` |

---

## 44. Formula-to-class summary

| Class | Space | Core score |
|---|---|---|
| `qRegressionStraddle` | regression posterior | $\beta\sigma-|\mu-h|$ |
| `qRegressionJointStraddle` | regression joint covariance | $-\mathrm{mean}|\mu-h|+\beta U(\Sigma)$ |
| `qRegressionICU` | regression posterior | Gaussian contour weight times $\sigma$ |
| `qRegressionBoundaryVariance` | regression posterior | boundary weight times variance |
| `qRegressionProbabilityOfExceedance` | response membership | Gaussian CDF or sigmoid mode |
| `qBinaryLatentStraddleAcquisition` | binary latent | smoothed $\beta\sigma-|\mu-h_f|$ |
| `qBinaryICUAcquisition` | binary probability | $4p(1-p)$ |
| `qBinaryBoundaryVarianceAcquisition` | binary latent | boundary weight times latent variance |
| `qBinaryClassEntropyAcquisition` | binary probability | Bernoulli entropy |
| `qMulticlassLatentStraddleAcquisition` | target probability | $\beta u-|p_T-h_p|$ |
| `qMulticlassICUAcquisition` | target probability | $u^2$ times Gaussian contour weight |
| `qMulticlassBoundaryVarianceAcquisition` | target probability | $u^2$ times exponential boundary weight |
| `qMulticlassClassEntropyAcquisition` | class probabilities | categorical or selected-class entropy score |
| `qMulticlassProbabilityOfExceedance` | target probability | sigmoid threshold score |
| `qOrdinalLatentStraddleAcquisition` | ordinal latent/cutpoints | $\beta\sigma-|\mu-c_j|$ |
| `qOrdinalICUAcquisition` | cumulative ordinal probability | $4g_j(1-g_j)$ |
| `qOrdinalBoundaryVarianceAcquisition` | ordinal latent/cutpoints | variance times cutpoint kernel weight |
| `qOrdinalClassEntropyAcquisition` | ordinal class probability | categorical entropy |

---

## 45. Validation checklist for an LSE implementation

1. Verify posterior accessor and threshold space.
2. Verify formula against class `forward()`.
3. Test `q=1` and `q>1`.
4. Test t-batch output shape.
5. Test single and multiple outputs.
6. Test class or boundary indexing.
7. Test `X_pending` updates.
8. Test observed-point penalty.

## Non-Gaussian response and observation level sets

Response-mean Straddle uses $\beta\sigma_\mu-|\bar\mu-t|$ and excludes
observation noise. JointStraddle replaces pointwise uncertainty by a covariance
trace or log determinant. BoundaryVariance and ICUProxy are local contour
scores; ICUProxy is not fantasy-based integrated contour reduction. Response
PoE uses fixed MC samples (smooth MC by default), whereas ObservationPoE
integrates a family CDF and includes heteroscedastic variance through moment
matching where it cannot remain in the original family. For counts,
$P(Y\ge k)=1-P(Y\le k-1)$ and $P(Y\le t)=F(\lfloor t\rfloor)$.
LevelSetUncertainty scores Bernoulli variance, binary entropy, or margin and is
maximal at exceedance probability one half.
9. Test exact duplicates.
10. Test mixed input transform.
11. Test `InputPerturbation` with `n_w>1`.
12. Test DeepGP or ensemble extra axes.
13. Compare pointwise and joint batch behavior.
14. Compare acquisition against external LSE loss from Chapter 05.
15. Record whether the score is a published criterion or an implementation
    proxy.

---

## 46. Interpretation rules

- `ICU` in this repository denotes family-specific contour-uncertainty proxies;
  it does not always mean exact integrated expected contour-loss reduction.
- `LatentStraddle` in the multiclass class name currently operates on selected
  target-class probability.
- Bernoulli variance and class entropy include observation ambiguity.
- Posterior probability variance is a different uncertainty source.
- Output reductions are score aggregations unless a joint event is explicitly
  computed.
- Distance penalties encourage diversity but are not Bayesian conditioning.
- Risk score objectives aggregate acquisition scores unless the class defines a
  robust latent target before scoring.

These rules should be used when comparing acquisitions across task families.
