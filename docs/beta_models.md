# Beta regression models

Bochan's Beta family models continuous proportions on their original scale. It is
not a binary classifier and does not convert percentages in the range 0–100.
Targets must be strictly inside `(0, 1)`. With `clip_targets=True`, exact zero and
one are accepted with a warning and clipped to `eps` and `1 - eps`; values outside
`[0, 1]`, NaN, and infinity remain errors. Generic outcome transforms such as
`Standardize` are not supported.

The public normal and mixed-input model types are `beta_base`, `beta_deepgp`,
`beta_deepkernel`, `beta_saas`, `beta_pca`, `beta_rembo`, `beta_rrp`, and
`beta_hetero`. The latter exposes auxiliary input-dependent variance through
`predict_noise_var`. Its posterior variance includes that additional variance,
while `rsample` delegates to the base Beta posterior and does not promise an
auxiliary-noise sample.

```python
import torch
from bochan.api import BayesianOptimizer, FitConfig, ModelConfig

train_X = torch.rand(30, 3, dtype=torch.double)
train_Y = torch.sigmoid(1.5 * train_X[:, 0] - 0.8 * train_X[:, 1]).unsqueeze(-1)
optimizer = BayesianOptimizer(
    model_config=ModelConfig(
        task_type="regression",
        model_type="beta_base",
        outcome_transform=False,
        model_kwargs={
            "link": "sigmoid",
            "init_concentration": 20.0,
            "clip_targets": True,
        },
    ),
    fit_config=FitConfig(fit_method="non_gaussian", num_epochs=100, lr=0.01),
)
optimizer.fit(train_X, train_Y)
```

Multiple Beta outputs use independent `OutputConfig` models. There is no public
native `beta_multitask` model. Concentration describes conditional dispersion; it
is not feature importance. RRP relevance is an observation diagnostic, and
permutation importance is predictive association rather than a causal effect.
