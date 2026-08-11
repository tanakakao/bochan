# Poisson regression models

## Public model keys

The high-level API accepts `poisson_base`, `poisson_deepgp`,
`poisson_deepkernel`, `poisson_saas`, `poisson_pca`, `poisson_rembo`,
`poisson_hetero`, `poisson_rrp`, and `poisson_multitask`. The first eight
families existed before the multitask integration; this package now exposes all
of them through the default registry and factory.

```python
from bochan.api import BayesianOptimizer, FitConfig, ModelConfig

optimizer = BayesianOptimizer(
    model_config=ModelConfig(model_type="poisson_multitask"),
    fit_config=FitConfig(num_epochs=300, lr=0.01),
    bounds=bounds,
)
optimizer.fit(train_X, train_Y)
```

`PoissonMultiTaskGPModel` accepts complete `[n, m]` targets and
`WidePoissonMultiTaskGPModel` additionally accepts partial `NaN` cells. A zero
is an observed count, never a missing-value marker. Every task must contain at
least one observation. The model is one correlated variational GP with an ICM
data-by-task kernel; task correlations are learned between latent log-rates. It
does not replace the count data with a Gaussian multitask model or independent
`ModelList` members.

## Statistical and posterior contract

Targets must be finite, non-negative, and integer-like (absolute tolerance
`1e-6`); only `NaN` is accepted as a wide-format missing value. Ordinary
outcome transforms such as `Standardize` are rejected because transformed
values are no longer Poisson counts.

* `latent_posterior(X)` represents the joint latent log-rate process.
* `rate_posterior(X)` represents positive, differentiably sampled expected
  rates and is the posterior used by gradient-based MC acquisition functions.
* `posterior(X)` has the same mean but includes conditional Poisson variance,
  so its marginal predictive variance approximates `E[rate] + Var(rate)`.
* `sample_observations(X)` draws non-reparameterized integer counts and is only
  intended for prediction, visualization, and posterior predictive checks.

Multitask posterior tensors have shape `[..., q, m]`, and rate samples have
shape `[S, ..., q, m]`. The underlying joint latent covariance and learned task
covariance remain available through `posterior.latent_posterior` and
`model.task_covar_matrix`.

## Current limitations

The models currently assume equal exposure. Exposure/log-offset metadata,
mixed-input multitask regression, and exact Poisson fantasy conditioning are
not implemented in this change. `fantasize()` rebuilds a variational model from
response-scale fantasy draws and is explicitly a local approximation; batched
fantasy training data are rejected rather than silently losing task
covariance. For substantially overdispersed data, use Negative Binomial
regression. Structural excess zeros require a dedicated zero-inflated model;
Poisson models never switch distribution families automatically.
