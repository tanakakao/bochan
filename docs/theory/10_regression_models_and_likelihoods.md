# 10. Regression Models and Likelihoods

This chapter describes regression models according to the support and
observation distribution of the response.  Gaussian-process foundations,
kernels, posterior conditioning, and variational inference are defined once in
Chapter 01 and are not repeated here.

The central modeling question is:

> What observation distribution is appropriate for the measured response, and
> what quantity does the model posterior represent?

---

## 1. Regression task definition

A regression observation is associated with input

$$
x_i\in\mathcal X
$$

and response

$$
y_i\in\mathcal Y.
$$

The response support determines plausible likelihoods.

| Response support | Typical model |
|---|---|
| \(\mathbb R\) | Gaussian regression |
| \((0,1)\) | Beta regression |
| \((0,\infty)\) continuous | Gamma regression |
| \(\{0,1,2,\ldots\}\) | Poisson or Negative-Binomial regression |
| bounded interval \((a,b)\) | transformed Beta or bounded likelihood |
| heavy-tailed real response | Student-t or robust likelihood |

A Gaussian likelihood can still be a useful approximation after transformation,
but the transformation and inverse interpretation must be explicit.

---

## 2. Generalized latent-function regression

Let

$$
f(x)
$$

be a latent GP function.  A likelihood parameter is defined through a link:

$$
\eta(x)=f(x),
\qquad
\vartheta(x)=g^{-1}(\eta(x)).
$$

The observation model is

$$
y\mid x
\sim
p(y\mid\vartheta(x)).
$$

Three spaces must be distinguished:

1. latent space `f(x)`;
2. likelihood-parameter space `vartheta(x)`;
3. response space `Y`.

For a nonlinear inverse link,

$$
g^{-1}(\mathbb E[f])
\ne
\mathbb E[g^{-1}(f)].
$$

Therefore a plug-in response mean based only on latent posterior mean can differ
from the correctly marginalized predictive mean.

---

## 3. Gaussian regression

### 3.1 Homoscedastic model

The standard model is

$$
y_i=f(x_i)+\varepsilon_i,
\qquad
\varepsilon_i\sim\mathcal N(0,\sigma_n^2).
$$

Equivalently,

$$
y_i\mid f_i
\sim
\mathcal N(f_i,\sigma_n^2).
$$

The observation covariance is

$$
K_y=K_f+\sigma_n^2I.
$$

The model estimates one global noise variance unless a fixed-noise or
heteroscedastic construction is used.

### 3.2 Known observation variance

If measurement variance `s_i^2` is known,

$$
y_i\mid f_i
\sim
\mathcal N(f_i,s_i^2),
$$

$$
K_y
=
K_f+\operatorname{diag}(s_1^2,\ldots,s_n^2).
$$

In implementation, `train_Yvar` contains variances, not standard deviations.
It should have the same output scaling as `train_Y` after any outcome transform.

Known per-observation variance is different from learning a noise function.
Chapter 13 treats learned heteroscedastic models.

### 3.3 Latent and noisy prediction

The latent posterior predicts `f(x)`.  The predictive observation distribution
adds likelihood noise:

$$
\operatorname{Var}(Y_*\mid\mathcal D)
=
\operatorname{Var}(f_*\mid\mathcal D)
+\sigma_n^2.
$$

Use cases:

- latent posterior: optimization of the underlying mean process;
- noisy predictive posterior: interval for a future measurement;
- reliability: probability a future observation satisfies a specification.

The `observation_noise` argument controls this distinction where supported.

---

## 4. Gaussian posterior quantities used in decisions

For scalar posterior

$$
f(x)\mid\mathcal D
\sim
\mathcal N(\mu(x),\sigma^2(x)),
$$

important summaries include:

### Mean

$$
\mathbb E[f(x)\mid\mathcal D]=\mu(x).
$$

### Quantile

$$
q_\alpha(x)
=
\mu(x)+\Phi^{-1}(\alpha)\sigma(x).
$$

### Exceedance probability

$$
P(f(x)\ge h\mid\mathcal D)
=
\Phi\left(
\frac{\mu(x)-h}{\sigma(x)}
\right).
$$

### Posterior sample

$$
f^{(s)}(X)
\sim
p(f(X)\mid\mathcal D).
$$

The choice between latent and noisy prediction must be made before computing
these quantities.

---

## 5. Independent multi-output regression

For outputs

$$
\mathbf y(x)
=[y_1(x),\ldots,y_m(x)],
$$

independent models assume

$$
p(f_1,\ldots,f_m\mid\mathcal D)
=
\prod_{j=1}^m
p(f_j\mid\mathcal D_j).
$$

Advantages:

- different kernels and noise per output;
- different training input sets;
- easy handling of missing outputs;
- robust when relationships between outputs are weak.

Limitations:

- no information transfer;
- no cross-output posterior covariance;
- joint constraints or scalarization variance use independence unless another
  dependence model is added.

BoTorch ModelList-style constructions implement this pattern.

---

## 6. Correlated multi-output regression

A separable multitask covariance is

$$
\operatorname{Cov}[f_a(x),f_b(x')]
=
B_{ab}k_X(x,x').
$$

`B` is a positive-semidefinite task covariance matrix.  If all tasks share the
same input grid, the full covariance may use

$$
K_{\mathrm{full}}
=B\otimes K_X.
$$

### Interpretation

The learned task covariance describes residual association after accounting for
input covariance, transforms, and noise.  It is not equal to the raw empirical
correlation of target columns.

### Benefits

- information transfer between outputs;
- improved prediction for sparse tasks;
- coherent joint posterior samples.

### Risks

- negative transfer when task relationships are misspecified;
- identifiability between task and input covariance scales;
- stronger assumptions about aligned data;
- larger and more structured covariance computations.

`KroneckerMultiTaskGP`-style models are most natural when outputs are observed at
common inputs.

---

## 7. Mixed continuous and categorical regression

Let

$$
x=(x_c,x_g)
$$

contain continuous and categorical variables.

A mixed kernel can be written schematically as

$$
k(x,x')
=
k_c(x_c,x_c')
+k_g(x_g,x_g')
+k_{cg}(x,x').
$$

A common implementation uses

$$
k_{cg}=k_c'k_g'.
$$

Important rules:

- category codes are identifiers;
- normalize continuous columns only;
- use a categorical kernel or fixed-feature enumeration;
- preserve category values during input perturbation unless a categorical error
  model is defined;
- acquisition optimization must respect the discrete search space.

---

## 8. Beta regression

Beta regression is suitable for continuous responses

$$
y\in(0,1).
$$

Parameterize

$$
y\sim\operatorname{Beta}(\alpha,\beta).
$$

Using mean and precision,

$$
\mu
=
\frac{\alpha}{\alpha+\beta},
\qquad
\phi
=
\alpha+\beta,
$$

$$
\alpha=\mu\phi,
\qquad
\beta=(1-\mu)\phi.
$$

The conditional variance is

$$
\operatorname{Var}(Y\mid x)
=
\frac{\mu(x)[1-\mu(x)]}{\phi(x)+1}.
$$

### Link functions

A latent mean function can use

$$
\mu(x)=\sigma(f_\mu(x)).
$$

Precision must be positive:

$$
\phi(x)
=
\operatorname{softplus}(f_\phi(x))+\epsilon.
$$

### Modeling choices

- one latent GP for mean and constant precision;
- one latent GP for mean and learned global precision;
- two latent functions for mean and input-dependent precision.

Only the last represents full input-dependent dispersion.

### Boundary responses

Exact values `0` and `1` are outside the ordinary Beta support.  Options include:

- zero/one-inflated Beta model;
- physically justified clipping;
- transformation such as `(y*(n-1)+0.5)/n`;
- alternative likelihood.

The chosen treatment affects tail predictions.

---

## 9. Gamma regression

Gamma regression models positive continuous responses:

$$
y>0.
$$

Using shape `a` and rate `b`,

$$
y\sim\operatorname{Gamma}(a,b),
$$

$$
\mathbb E[Y]=\frac{a}{b},
\qquad
\operatorname{Var}(Y)=\frac{a}{b^2}.
$$

Using shape and scale `theta`,

$$
\mathbb E[Y]=a\theta,
\qquad
\operatorname{Var}(Y)=a\theta^2.
$$

Documentation and code must state whether the second parameter is rate or
scale.

Positive parameters can be generated by

$$
\operatorname{softplus}(f)+\epsilon
$$

or exponentiation.

Gamma regression is useful for:

- durations;
- positive intensities;
- skewed continuous measurements;
- rates with variance increasing with mean.

It is inappropriate when zeros occur unless a hurdle or zero-inflated model is
specified.

---

## 10. Poisson regression

For count response

$$
y\in\{0,1,2,\ldots\},
$$

Poisson regression uses

$$
y\mid x
\sim
\operatorname{Poisson}(\lambda(x)),
$$

$$
\lambda(x)>0.
$$

A log link is

$$
\lambda(x)=\exp(f(x)).
$$

Then

$$
\mathbb E[Y\mid x]
=
\lambda(x),
$$

$$
\operatorname{Var}(Y\mid x)
=
\lambda(x).
$$

The equal mean-variance assumption is restrictive.  Exposure or observation
window `e(x)` can be included as offset:

$$
\lambda(x)=e(x)\exp(f(x)).
$$

Ignoring exposure differences can create spurious input effects.

---

## 11. Negative-Binomial regression

Negative-Binomial regression allows overdispersed counts.  A common
mean-dispersion parameterization is

$$
\mathbb E[Y\mid x]=\mu(x),
$$

$$
\operatorname{Var}(Y\mid x)
=
\mu(x)+\frac{\mu(x)^2}{r(x)}.
$$

Here `r` is a positive dispersion or size parameter.  As

$$
r\rightarrow\infty,
$$

the variance approaches the Poisson variance.

Use Negative Binomial when count variance substantially exceeds the mean and
that overdispersion is not explained by missing covariates or zero inflation.

A model may use:

- GP for `mu(x)` and constant `r`;
- GP for `mu(x)` and input-dependent `r(x)`;
- separate latent functions for mean and dispersion.

---

## 12. Poisson versus Negative Binomial

| Property | Poisson | Negative Binomial |
|---|---|---|
| Support | nonnegative integers | nonnegative integers |
| Mean | \(\lambda\) | \(\mu\) |
| Variance | \(\lambda\) | \(\mu+\mu^2/r\) |
| Overdispersion | not represented | represented |
| Complexity | lower | higher |
| Dispersion identifiability | not applicable | can be weak with few data |

Model comparison should use predictive log likelihood and calibration of count
quantiles, not only RMSE.

---

## 13. Response transformations versus non-Gaussian likelihoods

A positive response can be transformed:

$$
z=\log y
$$

and modeled with Gaussian regression.  This differs from Gamma regression.

### Log-Gaussian model

$$
\log Y=f(x)+\varepsilon.
$$

The response is approximately lognormal.  The predictive mean in original
space is not simply `exp(mu)`:

$$
\mathbb E[Y]
=
\exp\left(\mu+\frac12\sigma^2\right)
$$

for a Gaussian log response.

### Direct non-Gaussian model

A Gamma or count likelihood encodes response support and mean-variance relation
directly.

Transformation can be simpler and more stable; direct likelihood can be more
interpretable.  Compare them empirically using predictive distributions.

---

## 14. Exact and variational fitting by model family

### Gaussian exact GP

Use exact marginal likelihood when data size and covariance structure permit.

### Non-Gaussian GP

Use variational ELBO or another approximate inference method.

### Deep non-Gaussian GP

Use deep variational objectives and likelihood sampling.

### Heteroscedastic approximation

A mean model plus auxiliary noise/dispersion model may use separate fit stages
or a joint custom optimizer.

The fit function must match the model.  A VAE-GP or DeepGP should not be passed
to a generic exact-GP fitting routine unless explicitly supported.

---

## 15. Posterior-space requirements

A regression wrapper should state whether `posterior()` represents:

1. latent GP value;
2. likelihood mean;
3. predictive observation;
4. transformed response;
5. a normal proxy to a non-Gaussian response.

For non-Gaussian likelihoods, `posterior.mean` may be:

- latent mean;
- response-parameter mean;
- response expectation;
- Monte Carlo approximation.

A BO threshold or objective must use the same interpretation.

---

## 16. Acquisition implications

### Gaussian regression

Standard EI, NEI, UCB, KG, and LSE acquisitions can use the GP posterior
directly.

### Beta regression

A bounded response objective should operate in response or utility space.  A
latent Gaussian improvement threshold may not correspond to a bounded-response
threshold.

### Count regression

Expected count, probability of exceeding a count, or count utility may be used.
The predictive distribution is discrete and asymmetric.

### Positive continuous response

Gamma-response thresholds should use response-space probabilities or samples.

For all non-Gaussian models, an acquisition that assumes normality of the final
response must document the proxy approximation.

---

## 17. Model checking

### Residual checks for Gaussian regression

Inspect:

- residual mean;
- residual versus fitted value;
- heteroscedasticity;
- heavy tails;
- outliers;
- dependence on omitted variables.

### Probability integral transform

For continuous predictive CDF `F_i`, compute

$$
u_i=F_i(y_i).
$$

A calibrated model gives approximately uniform `u_i`.

### Count calibration

Compare observed and predicted:

- zero frequency;
- mean and variance by input region;
- upper-tail counts;
- predictive intervals;
- deviance residuals.

### Bounded-response calibration

Check mass near boundaries and predictive quantiles, not only mean error.

---

## 18. Evaluation metrics

### Point prediction

- RMSE;
- MAE;
- Poisson or Gamma deviance;
- mean absolute percentage error only when zeros and scale permit.

### Probabilistic prediction

- negative log predictive density;
- continuous ranked probability score;
- interval coverage;
- calibration curves;
- tail exceedance calibration.

### Sequential decisions

- regret;
- probability of meeting target;
- feasible regret;
- LSE set error;
- robust performance under repeats.

A model with slightly worse RMSE can produce better BO decisions if its
uncertainty is better calibrated.

---

## 19. `bochan` implementation correspondence

### 19.1 Gaussian regression registry

`src/bochan/api/model_registry.py` resolves base Gaussian regression to:

| Configuration | Model |
|---|---|
| continuous base | BoTorch `SingleTaskGP` |
| mixed base | BoTorch `MixedSingleTaskGP` |
| mixed Kronecker multitask | `MixedKroneckerMultiTaskGP` |
| high-dimensional variants | task-specific SAAS, PCA, REMBO, or VAE wrappers |
| deep variants | task-specific DeepGP or Deep Kernel wrappers |
| robust variants | heteroscedastic or relevance-pursuit wrappers |

### 19.2 Gaussian source tree

```text
src/bochan/models/regression/gaussian/base/
src/bochan/models/regression/gaussian/deep/
src/bochan/models/regression/gaussian/high_dim/
src/bochan/models/regression/gaussian/robust/
```

### 19.3 Non-Gaussian source tree

```text
src/bochan/models/regression/non_gaussian/beta/
src/bochan/models/regression/non_gaussian/gamma/
src/bochan/models/regression/non_gaussian/poisson/
src/bochan/models/regression/non_gaussian/negative_binomial/
```

Each family may contain:

- base model;
- mixed model;
- DeepGP or Deep Kernel variant;
- high-dimensional variant;
- robust or heteroscedastic variant;
- task-specific fitting helper.

### 19.4 Fit functions

```text
src/bochan/fit/
```

contains exact, variational, deep, robust, and VAE-specific fitting paths.  The
high-level API selects a fit procedure based on model type.

### 19.5 Acquisition source tree

```text
src/bochan/acquisition/regression/
src/bochan/acquisition/non_gaussian/
```

contains BO, Active Learning, and LSE acquisitions.  The acquisition must match
the posterior space returned by the model family.

---

## 20. Regression model-selection checklist

1. What is the response support?
2. Are zeros or boundary values present?
3. Is the response continuous or count-valued?
4. Does variance change with the mean or input?
5. Is overdispersion present?
6. Are there outliers or heavy tails?
7. Are outputs correlated?
8. Are all outputs observed at common inputs?
9. Is a response transformation scientifically interpretable?
10. Does the wrapper posterior return latent or response values?
11. Which fit function is required?
12. Which acquisition assumptions are valid?
13. Which predictive and sequential metrics will compare models?

---

## 21. References

- Rasmussen and Williams, *Gaussian Processes for Machine Learning*, 2006.
- McCullagh and Nelder, *Generalized Linear Models*.
- Ferrari and Cribari-Neto, *Beta Regression for Modelling Rates and Proportions*, 2004.
- Hilbe, *Negative Binomial Regression*.
