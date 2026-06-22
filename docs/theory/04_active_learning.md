# 04. Active Learning

Active Learning chooses observations to improve a predictive model or reduce
uncertainty about a specified scientific target.  It is not defined by one
acquisition function.  It is defined by what is being learned and how learning
quality is measured.

This chapter develops regression, classification, ordinal, multi-output,
heteroscedastic, and q-batch Active Learning.  Level-set Estimation is treated
separately in Chapters 05 and 16 because it has a boundary-specific loss.

---

## 1. Problem formulation

Let the current data be

\[
\mathcal D_t=\{(x_i,y_i)\}_{i=1}^{n_t}.
\]

An Active Learning policy chooses

\[
x_{t+1}
\in
\arg\max_{x\in\mathcal X}
\alpha_{\mathrm{AL}}(x;\mathcal D_t).
\]

The acquisition is valuable only relative to a learning target.  Examples are:

- the latent function over the entire domain;
- predictive response distribution;
- class labels;
- class probabilities;
- model parameters;
- a selected output or set of outputs;
- a low-dimensional region of interest;
- expected utility;
- a task covariance or noise function.

A point can have high objective value but low learning value, or low objective
value but high learning value.

---

## 2. Pool-based and continuous Active Learning

### 2.1 Pool-based setting

Candidates come from a finite set

\[
\mathcal P=\{x^{(1)},\ldots,x^{(N)}\}.
\]

The next point is

\[
x_{t+1}
\in
\arg\max_{x\in\mathcal P\setminus X_{\mathrm{observed}}}
\alpha(x).
\]

This is common when unlabeled specimens or simulation cases already exist.

### 2.2 Continuous setting

Candidates are optimized over a domain:

\[
x_{t+1}
\in
\arg\max_{x\in\mathcal X}
\alpha(x).
\]

This uses the same acquisition-optimization machinery as BO, including bounds,
mixed variables, q-batches, and constraints.

### 2.3 Experimental-design setting

The candidate determines how a measurement is made rather than which existing
sample is labeled.  Costs, replicates, sensor choices, or output subsets may be
part of the decision.

---

## 3. Local uncertainty sampling

The simplest strategy samples where the model is uncertain.

For regression latent posterior

\[
f(x)\mid\mathcal D
\sim
\mathcal N(\mu(x),\sigma_f^2(x)),
\]

posterior-variance sampling uses

\[
\alpha_{\mathrm{var}}(x)=\sigma_f^2(x).
\]

This is a local criterion.  It does not directly measure how observing `x`
reduces uncertainty elsewhere.

Advantages:

- inexpensive;
- easy to optimize;
- appropriate baseline;
- directly targets epistemic uncertainty when latent variance is used.

Limitations:

- may oversample domain boundaries;
- may focus on scientifically irrelevant regions;
- may confuse observation noise with learnable uncertainty;
- may produce clustered q-batches;
- may not reduce integrated predictive error efficiently.

---

## 4. Predictive entropy

For a random future observation `Y`, predictive entropy is

\[
H(Y\mid x,\mathcal D)
=
-\mathbb E
\left[
\log p(Y\mid x,\mathcal D)
\right].
\]

### 4.1 Gaussian regression

If

\[
Y\mid x,\mathcal D
\sim
\mathcal N(\mu_Y(x),\sigma_Y^2(x)),
\]

then

\[
H(Y\mid x,\mathcal D)
=
\frac12
\log\left(2\pi e\sigma_Y^2(x)\right).
\]

Maximizing Gaussian entropy is equivalent to maximizing predictive variance.
If `sigma_Y^2` includes observation noise, entropy may prefer irreducibly noisy
regions.

### 4.2 Binary classification

For probability

\[
p=P(Y=1\mid x,\mathcal D),
\]

\[
H(Y\mid x,\mathcal D)
=
-p\log p-(1-p)\log(1-p).
\]

The maximum occurs at `p=0.5`.

### 4.3 Multiclass and ordinal outputs

For class probabilities

\[
\mathbf p=(p_0,\ldots,p_{K-1}),
\]

\[
H(Y\mid x,\mathcal D)
=
-\sum_{k=0}^{K-1}p_k\log p_k.
\]

Entropy treats classes as labels.  For ordinal models it does not encode that
confusion between adjacent grades is less severe than confusion between distant
grades.  Utility variance or ranked losses can better reflect order.

---

## 5. Aleatoric and epistemic uncertainty

Predictive uncertainty can be decomposed conceptually into:

- aleatoric uncertainty: randomness in observations given the underlying
  function;
- epistemic uncertainty: uncertainty about the function or parameters.

For Gaussian regression with known noise,

\[
\operatorname{Var}(Y\mid x,\mathcal D)
=
\underbrace{\operatorname{Var}(f(x)\mid\mathcal D)}_{\text{epistemic}}
+
\underbrace{\sigma_n^2(x)}_{\text{aleatoric}}.
\]

For classification, Bernoulli variance

\[
p(1-p)
\]

can be high even if `p` is known exactly.  It is observation ambiguity, not by
itself epistemic uncertainty.

A learning acquisition should state whether it values:

- total predictive uncertainty;
- latent posterior uncertainty;
- uncertainty in the class-probability function;
- uncertainty in likelihood or noise parameters.

---

## 6. BALD and mutual information

Bayesian Active Learning by Disagreement uses mutual information between a
future observation and latent model uncertainty.

Let `Theta` denote the latent function, model parameters, or another learning
target.  Then

\[
\operatorname{BALD}(x)
=
I(Y;\Theta\mid x,\mathcal D).
\]

Using the entropy identity,

\[
I(Y;\Theta\mid x,\mathcal D)
=
H(Y\mid x,\mathcal D)
-
\mathbb E_{\Theta\mid\mathcal D}
[H(Y\mid x,\Theta)].
\]

The first term is total predictive uncertainty.  The second is expected
irreducible uncertainty after the latent state is known.  Their difference
isolates disagreement due to posterior uncertainty.

### 6.1 Binary Monte Carlo estimator

Draw latent samples

\[
f^{(s)}(x)
\sim
p(f(x)\mid\mathcal D),
\]

convert to probabilities

\[
p_s=\pi(f^{(s)}(x)),
\]

and define

\[
\bar p=\frac1S\sum_{s=1}^Sp_s.
\]

Then

\[
\widehat{\operatorname{BALD}}
=
H(\bar p)
-
\frac1S\sum_{s=1}^SH(p_s).
\]

### 6.2 Multiclass estimator

For sampled class-probability vectors

\[
\mathbf p_s,
\]

\[
\widehat{\operatorname{BALD}}
=
H\left(\frac1S\sum_s\mathbf p_s\right)
-
\frac1S\sum_sH(\mathbf p_s).
\]

### 6.3 Interpretation

BALD is high when posterior samples disagree about predictions but each sample
is relatively confident.  Predictive entropy is high both for disagreement and
for consistent aleatoric ambiguity.

---

## 7. Margin uncertainty

### 7.1 Binary margin

A simple score is

\[
\alpha_{\mathrm{margin}}(x)
=1-|2p(x)-1|.
\]

It is maximal at `p=0.5`.

### 7.2 Multiclass margin

Let

\[
p_{(1)}\ge p_{(2)}\ge\cdots
\]

be sorted class probabilities.  A margin score is

\[
\alpha_{\mathrm{margin}}(x)
=1-[p_{(1)}-p_{(2)}].
\]

A small top-two gap indicates ambiguous classification.

Margin uncertainty is cheap but does not separate aleatoric and epistemic
uncertainty.

---

## 8. Probability variance

Suppose posterior latent samples produce probabilities

\[
p^{(s)}(x).
\]

Probability-function uncertainty is

\[
\operatorname{Var}_s[p^{(s)}(x)].
\]

This differs from Bernoulli observation variance

\[
\bar p(1-\bar p).
\]

For multiclass probabilities, one can compute per-class variance and reduce by:

- sum;
- mean;
- maximum;
- target-class selection;
- utility-weighted covariance.

The reduction changes the learning target.

---

## 9. Integrated posterior variance

Local uncertainty asks where the model is uncertain.  Integrated Posterior
Variance asks how much an observation reduces uncertainty over a reference
region.

For reference measure `nu`, current integrated variance is

\[
\operatorname{IPV}_t
=
\int_{\mathcal X}
\sigma_t^2(z)
\,d\nu(z).
\]

The ideal one-step reduction from observing at `x` is

\[
\Delta\operatorname{IPV}(x)
=
\operatorname{IPV}_t
-
\mathbb E_{y_x}
[\operatorname{IPV}_{t+1}\mid x,y_x].
\]

For an exact noiseless GP, posterior covariance update does not depend on the
observed value, and variance reduction can be computed from covariance:

\[
\sigma_{t+1}^2(z)
=
\sigma_t^2(z)
-
\frac{k_t(z,x)^2}
{k_t(x,x)+\sigma_n^2}.
\]

Therefore

\[
\Delta\operatorname{IPV}(x)
=
\int
\frac{k_t(z,x)^2}
{k_t(x,x)+\sigma_n^2}
\,d\nu(z).
\]

In non-Gaussian or approximate models, fantasy conditioning or proxy criteria
may be required.

### Negative Integrated Posterior Variance

Some APIs maximize

\[
-\operatorname{IPV}_{t+1}
\]

rather than directly maximizing variance reduction.  The sign and baseline
must be documented.

---

## 10. Expected model change

Another family values the expected change in model parameters or predictions.
For parameter vector `theta`, a gradient-based proxy is

\[
\alpha(x)
=
\mathbb E_{Y\mid x,\mathcal D}
\left[
\|\nabla_\theta\ell(Y,x;\theta)\|^2
\right].
\]

For GP models, information gain and posterior variance reduction are often more
natural because the model is explicitly probabilistic.  Expected model change
can still be useful for deep feature extractors or approximate likelihoods.

---

## 11. Regression Active Learning

Useful criteria include:

### Latent posterior variance

\[
\alpha(x)=\operatorname{Var}[f(x)\mid\mathcal D].
\]

### Predictive entropy

Includes observation noise when the observation distribution is used.

### BALD-style Gaussian information

For Gaussian latent value and additive Gaussian noise, mutual information
between noisy observation and latent function at the same point is

\[
I(Y;f(x)\mid\mathcal D)
=
\frac12
\log\left(
1+
\frac{\sigma_f^2(x)}{\sigma_n^2(x)}
\right).
\]

This favors large epistemic variance relative to noise.

### Integrated variance reduction

Values global covariance reduction over a reference set.

### Region-weighted learning

For scientific weight `w(z)`,

\[
\operatorname{IPV}_w
=
\int w(z)\sigma^2(z)d\nu(z).
\]

This avoids spending budget in irrelevant parts of the domain.

---

## 12. Binary and multiclass Active Learning

### Binary

Possible learning targets are:

- latent boundary function;
- class probability;
- future label;
- classifier parameters;
- feasibility probability.

Corresponding acquisitions include:

- latent variance;
- predictive entropy;
- margin;
- probability variance;
- BALD;
- integrated probability variance.

### Multiclass

The class dimension must be retained until a meaningful reduction is applied.
Possible reductions are:

\[
\sum_k\operatorname{Var}[p_k],
\]

\[
\max_k\operatorname{Var}[p_k],
\]

or uncertainty in a target set of classes.

A class-average reduction can hide a highly uncertain safety-critical class.

---

## 13. Ordinal Active Learning

Ordinal labels have an order and cutpoints.  Learning targets include:

- full class-probability vector;
- latent quality function;
- one or all cutpoints;
- expected utility;
- probability of exceeding a grade;
- class-transition boundaries.

### Predictive entropy

Treats classes as categorical labels.

### Utility variance

For class utilities `u_k`,

\[
\bar u(x)=\sum_kp_k(x)u_k,
\]

\[
\operatorname{Var}(U\mid x)
=
\sum_kp_k(x)[u_k-\bar u(x)]^2.
\]

This respects utility spacing but is conditional class-distribution variance,
not necessarily epistemic uncertainty in expected utility.

### Ordinal BALD

\[
I(Y;f\mid x,\mathcal D)
=
H\left(\mathbb E_f[p(Y\mid f)]\right)
-
\mathbb E_f[H(p(Y\mid f))].
\]

### Boundary-specific learning

When one grade transition matters, reduce the ordinal prediction to binary event

\[
Z_j=\mathbf1[Y\ge j+1]
\]

and learn its probability or latent cutpoint boundary.

---

## 14. Multi-output Active Learning

For outputs `j=1,...,m`, let `a_j(x)` be an output-wise information score.
Common reductions include

\[
\alpha(x)=\sum_jw_ja_j(x),
\]

\[
\alpha(x)=\max_ja_j(x),
\]

\[
\alpha(x)=\min_ja_j(x).
\]

Interpretation:

- weighted sum: average learning priority;
- maximum: focus on the currently least-known output;
- minimum: choose points informative for all outputs, often restrictive.

If outputs are correlated, the joint information is not generally the sum of
marginal information:

\[
I(Y_1,\ldots,Y_m;\Theta)
\ne
\sum_jI(Y_j;\Theta).
\]

Independent ModelList and `HybridPosterior` approximations do not provide full
cross-output information.

---

## 15. q-batch Active Learning

Selecting the top `q` pointwise scores can be redundant.  Joint batch
information is

\[
I(\mathbf Y_X;\Theta\mid\mathcal D),
\qquad
\mathbf Y_X=[Y(x_1),\ldots,Y(x_q)].
\]

### 15.1 Gaussian log-determinant criterion

For Gaussian observations with covariance `Sigma_X` and independent noise
variance `sigma_n^2`, information about latent values has form

\[
I(\mathbf Y_X;\mathbf f_X)
=
\frac12
\log\det
\left(
I+\sigma_n^{-2}\Sigma_X
\right).
\]

The determinant rewards uncertain and nonredundant batches.

### 15.2 Joint BALD

For classification, joint BALD estimates

\[
I(\mathbf Y_X;\Theta\mid\mathcal D).
\]

Exact enumeration scales as `K^q`.  Greedy or Monte Carlo approximations are
needed for larger class count or batch size.

### 15.3 Distance penalties

A practical proxy subtracts same-batch similarity.  It is cheaper but not equal
to joint information gain.

---

## 16. Heteroscedastic Active Learning

If observation noise depends on input,

\[
Y=f(x)+\varepsilon(x),
\qquad
\varepsilon(x)\sim\mathcal N(0,\sigma_n^2(x)),
\]

maximum predictive variance can select points with large irreducible noise.

Possible goals differ:

### Learn the mean function

Prefer epistemic variance relative to noise, for example

\[
\frac{\sigma_f^2(x)}{\sigma_n^2(x)+\epsilon}
\]

or Gaussian mutual information.

### Learn the noise function

Sample where the noise-model posterior is uncertain.

### Learn both

Use

\[
\alpha(x)=\alpha_f(x)+\lambda\alpha_g(x).
\]

### Improve prediction of future observations

Total predictive uncertainty may be relevant even if it is irreducible.

The goal must be declared before applying inverse-noise weighting.

---

## 17. Input perturbation and robust learning

For nominal point `x`, execution is

\[
\tilde x=x+\delta.
\]

A robust learning score may average pointwise information:

\[
\alpha_{\mathrm{mean}}(x)
=
\mathbb E_\delta[\alpha_0(x+\delta)].
\]

Alternatively, define information about a neighborhood-level functional, such
as

\[
\rho[f(x+\delta)].
\]

These are not the same.  Averaging acquisition scores and computing acquisition
about an aggregated robust target involve different orders of expectation and
nonlinear transformation.

Chapter 08 defines risk measures and Chapter 09 defines `q*n_w` shapes.

---

## 18. Cost-sensitive and output-selective learning

If observation cost depends on `x` or measured output set `S`, use information
per cost:

\[
\alpha_{\mathrm{cost}}(x,S)
=
\frac{I(Y_S;\Theta\mid x,\mathcal D)}{c(x,S)}.
\]

In heterogeneous experiments, the decision may include which response to
measure.  A wrapper that always observes all outputs cannot represent decoupled
measurement cost without an additional action variable.

---

## 19. Stopping criteria

Possible stopping rules include:

- integrated variance below a threshold;
- maximum local uncertainty below a threshold;
- predictive loss on a validation set stabilizes;
- BALD or expected information gain becomes negligible;
- class calibration reaches a target;
- scientific region of interest is sufficiently certain;
- labeling or experiment budget exhausted.

Predictive entropy may not approach zero when labels are inherently noisy.
Epistemic criteria are more appropriate for stopping based on learnability.

---

## 20. Evaluation

### Regression

- held-out RMSE or MAE;
- negative log predictive density;
- interval coverage;
- integrated posterior variance;
- error within a region of interest.

### Classification

- log loss;
- Brier score;
- calibration error;
- accuracy or macro-F1;
- class-specific recall;
- posterior probability error when ground truth is available.

### Ordinal

- negative log likelihood;
- mean absolute class error;
- ranked probability score;
- boundary calibration;
- expected-utility error.

### Sequential comparison

Plot each metric against:

- number of labels or experiments;
- total cost;
- wall-clock time.

Use identical initial data and multiple random seeds.

---

## 21. `bochan` implementation correspondence

The acquisition registry maps names to task-specific Active Learning classes.

### Regression

```text
src/bochan/acquisition/regression/active_learning/
```

Representative registered classes:

- `qRegressionPredictiveEntropy`;
- `qRegressionBALD`;
- `qRegressionPosteriorVariance`;
- `qRegressionNegIntegratedPosteriorVariance`;
- `qRegressionIntegratedPosteriorVarianceProxy`;
- multi-output and heteroscedastic variants.

### Binary classification

```text
src/bochan/acquisition/binary/active_learning/
```

Representative classes:

- `qBinaryPredictiveEntropy`;
- `qBinaryBALD`;
- `qBinaryJointBALD`;
- `qBinaryGreedyJointBALD`;
- `qBinaryProbabilityVariance`;
- `qBinaryMarginUncertainty`;
- fantasy NIPV and heteroscedastic variants.

### Multiclass classification

```text
src/bochan/acquisition/multiclass/active_learning/
```

Representative classes:

- `qMulticlassPredictiveEntropy`;
- `qMulticlassBALD`;
- `qMulticlassJointBALD`;
- `qMulticlassGreedyJointBALD`;
- probability variance, margin, IPV proxy, multi-output, and heteroscedastic
  variants.

### Ordinal

```text
src/bochan/acquisition/ordinal/active_learning/
```

Representative classes:

- `qOrdinalPredictiveEntropy`;
- `qOrdinalBALD`;
- `qOrdinalUtilityVariance`;
- `qOrdinalMarginUncertainty`;
- fantasy NIPV, multi-output, and heteroscedastic variants.

### High-level resolution

```text
src/bochan/api/acquisition_registry.py
```

contains the current public-name mappings.  The class implementation determines
whether the criterion is exact, fantasy-based, Monte Carlo, or a proxy.

---

## 22. New Active Learning component checklist

Document:

1. learning target;
2. uncertainty decomposition;
3. local or global criterion;
4. pointwise or joint q-batch formulation;
5. posterior space consumed;
6. class/output reduction;
7. heteroscedastic interpretation;
8. input perturbation order;
9. pending and observed-point handling;
10. cost model;
11. stopping metric;
12. external evaluation loss;
13. implementation approximation.

---

## 23. References

- MacKay, *Information-Based Objective Functions for Active Data Selection*, 1992.
- Cohn, Ghahramani, and Jordan, *Active Learning with Statistical Models*, 1996.
- Houlsby et al., *Bayesian Active Learning for Classification and Preference Learning*, 2011.
- Settles, *Active Learning Literature Survey*, 2009.
