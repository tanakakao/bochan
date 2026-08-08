# Deep Ensemble classification

`model_type="deep_ensemble"` supports binary and multiclass classification with a finite probability ensemble that follows bochan's existing classification posterior contract.

- Each member is an independently initialized PyTorch MLP.
- Binary members output one logit and use sigmoid probabilities.
- Multiclass members output one logit per class and use softmax probabilities.
- `ClassificationEnsemblePosterior.epistemic_variance` represents member disagreement.
- The existing `latent_posterior()` compatibility path remains available to classification acquisitions.
- Mixed models one-hot encode categorical inputs internally with Torch operations, preserving continuous-input gradients.

Example:

```python
ModelConfig(
    task_type="binary",
    model_type="deep_ensemble",
    model_kwargs={
        "ensemble_size": 5,
        "hidden_dims": (64, 64),
        "bootstrap": True,
    },
)
```

For mixed inputs, set `cat_dims` on `ModelConfig`.
