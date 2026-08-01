# Gamma regression models

Bochan exposes the existing Gamma families through `gamma_base`,
`gamma_deepgp`, `gamma_deepkernel`, `gamma_saas`, `gamma_pca`, `gamma_rembo`,
`gamma_hetero`, and `gamma_rrp`. Correlated multi-output regression is selected
with `gamma_multitask`.

```python
from bochan.api import BayesianOptimizer, FitConfig, ModelConfig

bo = BayesianOptimizer(
    model_config=ModelConfig(
        task_type="regression",
        model_type="gamma_multitask",
        model_kwargs={
            "rank": 2,
            "num_latents": 2,
            "num_inducing_points": 64,
        },
    ),
    fit_config=FitConfig(num_epochs=300, lr=0.01),
    bounds=bounds,
)
bo.fit(train_X, train_Y)
```

Partially observed wide targets use `float("nan")` only for missing cells. The
wide model converts observed cells to long form; it never fills missing cells
with a mean or zero. Every task must have at least one observation, and rows in
which every task is missing contribute no training observations.

Gamma targets must be strictly positive. Mean-centering transforms such as
`Standardize` can violate this assumption, so Gamma registry entries default to
`PositiveScaleOutcomeTransform`. The multi-task implementation is a sparse
variational non-Gaussian GP with a learned low-rank task covariance, not an
exact Gaussian multi-task GP and not a list of independent models. Its posterior
has shape `[..., q, m]`; samples have shape `[S, ..., q, m]` and preserve the
joint point/task latent covariance.

Single-draw `fantasize` uses local conditioning without re-optimizing the
variational posterior. Batched fantasy training data are currently rejected
explicitly; MC acquisition functions should consume joint posterior samples
directly. Mixed-input Gamma multitask is also not exposed because task-aware
categorical inducing-point handling is not yet stable. PCA and REMBO apply input
perturbations in raw space before their internal projection.
