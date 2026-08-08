# Deep Ensemble ordinal regression

`model_type="deep_ensemble"` supports ordinal regression with independently initialized neural latent-score models and a shared ordered-logit likelihood.

- Each member predicts one latent ordinal score.
- Shared monotone cutpoints are learned through `OrdinalLogitLikelihood`.
- `OrdinalEnsemblePosterior` retains finite latent member samples for epistemic uncertainty.
- A moment-matched Gaussian `posterior.distribution` bridge keeps existing ordinal likelihood quadrature and acquisition APIs compatible.
- `probability_posterior()` exposes member-wise ordered class probabilities.
- Mixed models use Torch-native one-hot encoding and retain gradients for continuous candidate variables.

Example:

```python
ModelConfig(
    task_type="ordinal",
    model_type="deep_ensemble",
    model_kwargs={
        "ensemble_size": 5,
        "hidden_dims": (64, 64),
        "bootstrap": True,
    },
)
```

Ordinal labels must be contiguous integers `0..K-1` with at least three classes.
