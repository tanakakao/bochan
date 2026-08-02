# Beta regression model contract and migration note

Bochan provides the public registry keys `beta_base`, `beta_deepgp`,
`beta_deepkernel`, `beta_saas`, `beta_pca`, `beta_rembo`, `beta_rrp`,
`beta_hetero`, and `beta_multitask`. The first eight implementations predated
this integration; `BetaMultiTaskGPModel` and `WideBetaMultiTaskGPModel` are the
new correlated wide-response models. Mixed variants remain available for the
single-output families.

Beta regression models continuous responses strictly inside `(0, 1)`. Binary
`0/1` labels belong to binary classification, while proportions derived from
varying trial counts may require Binomial or Beta-Binomial regression. Standard
Beta regression does not model structural zero/one inflation.

The canonical parameterization is `mean = sigmoid(f)`, precision
`concentration = phi > 0`, `concentration1 = mean * phi`, and
`concentration0 = (1 - mean) * phi`. Thus the conditional observation variance
is `mean * (1 - mean) / (phi + 1)`. Concentration is owned by the likelihood,
is positive by construction, is part of `state_dict`, and may be fixed or
learned. `FitConfig.beta` is unrelated: it remains the Variational ELBO KL
weight and is never interpreted as a Beta shape or precision.

The default `boundary_policy="error"` rejects zero and one. The explicit
`boundary_policy="clip"` uses fixed `boundary_epsilon` while preserving
`train_targets_raw` separately from `train_targets_model`; clipping does not
model zero/one inflation. Generic `Standardize` outcome transforms are rejected
because they invalidate the Beta likelihood.

`latent_posterior()` is on the correlated logit-mean scale. `mean_posterior()`
and mean samples are on the conditional response-mean scale and are intended
for ordinary BO. `predictive_posterior()` additionally includes Beta aleatoric
variance. Response samples are available through `sample_observations()`.

The multitask implementation converts only finite observed wide cells to long
form and fits one variational GP with an ICM product kernel. It retains the
latent task covariance, returns `[..., q, task]`, and uses one learnable
concentration per task. NaN alone denotes a missing cell and every task must
have at least one observation. Its `fantasize()` path is a local variational
rebuild rather than exact Beta conditioning; batched fantasy training remains
unsupported. Bounded outcome transforms, mixed-input multitask Beta, and a
fully input-dependent concentration process are intentionally deferred.
