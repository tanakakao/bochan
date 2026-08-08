# Neural regression surrogates

This package contains PyTorch-native neural surrogate models that expose the
standard BoTorch `Model` / `Posterior` contracts used throughout bochan.

## Deep Ensemble

Use the high-level API with:

```python
ModelConfig(
    task_type="regression",
    model_type="deep_ensemble",
    model_kwargs={
        "ensemble_size": 5,
        "hidden_dims": (64, 64),
        "bootstrap": True,
        "random_state": 0,
    },
)
```

Adding `cat_dims=[...]` selects `DeepEnsembleMixedRegressorModel` automatically.
Categorical columns remain in the public mixed search space and are one-hot
encoded only inside the neural surrogate. Continuous columns stay differentiable
with respect to candidate `X`.

### Posterior semantics

`DeepEnsembleRegressorModel` subclasses BoTorch `EnsembleModel`. Each independently
initialized neural network contributes one finite posterior member. The inherited
`EnsemblePosterior` therefore represents:

- `posterior.mean`: mean prediction across neural members;
- `posterior.variance`: member-to-member predictive disagreement, interpreted as
  epistemic uncertainty;
- `posterior.rsample()`: differentiable samples selected from the finite ensemble.

The default regression loss is mean squared error. Observation / aleatoric noise is
not folded into the ensemble posterior. This is intentional: the BO-facing
posterior represents latent-response uncertainty, matching bochan's Random Forest
and bootstrap NGBoost ensemble convention.

### Fitting

Deep Ensemble does not use a GPyTorch marginal log likelihood, so `make_mll()`
returns `None`. The common bochan factory then delegates to the model's bound
`fit()` method. `FitConfig` settings such as `num_epochs`, `lr`, `batch_size`,
`shuffle`, `clip_grad_norm`, and `optimizer_kwargs` are supported directly.

By default each member is trained on a bootstrap resample. A custom `member_factory`
or explicit `members` list can replace the default MLP architecture while retaining
the BoTorch ensemble interface.

### Input and outcome transforms

BoTorch input and outcome transforms are supported through the inherited model
contract. In particular, `InputPerturbation` retains its standard behavior: it is
not applied during training by default and expands candidate inputs during
evaluation. Outcome transforms such as bochan's default standardization are
applied during training and undone by `EnsembleModel.posterior()`.

### Current scope

The initial implementation covers single-output continuous regression for normal
and mixed inputs. Classification, ordinal, explicit aleatoric-output heads, and a
native correlated multi-output Deep Ensemble are intentionally left for separate
extensions rather than overloading the first regression implementation.
