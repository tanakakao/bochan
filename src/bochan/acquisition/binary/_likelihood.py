"""Likelihood-aware probability transforms for binary classification."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def _resolve_binary_likelihood(model: Any) -> Any:
    """Resolve the likelihood that defines ``p(y=1 | f)`` for one model."""
    likelihood = getattr(model, "likelihood", None)
    if likelihood is not None:
        return likelihood

    inner = getattr(model, "model", None)
    likelihood = getattr(inner, "likelihood", None) if inner is not None else None
    if likelihood is not None:
        return likelihood

    raise AttributeError(
        f"Binary likelihood was not found for {type(model).__name__}. "
        "Expected model.likelihood or model.model.likelihood."
    )


def _conditional_probability(
    model: Any,
    latent_samples: Tensor,
    *,
    eps: float,
    name: str,
) -> Tensor:
    likelihood = _resolve_binary_likelihood(model)
    conditional = likelihood.forward(latent_samples)

    probs = getattr(conditional, "probs", None)
    if probs is None:
        probs = getattr(conditional, "mean", None)
    if probs is None or not torch.is_tensor(probs):
        raise TypeError(
            f"{name}: {type(likelihood).__name__}.forward(...) did not return a distribution with Tensor probs or mean."
        )
    if not torch.isfinite(probs).all():
        raise RuntimeError(f"{name}: binary likelihood returned NaN or inf probabilities.")

    pmin = probs.detach().min().item()
    pmax = probs.detach().max().item()
    tolerance = 1e-6
    if pmin < -tolerance or pmax > 1.0 + tolerance:
        raise RuntimeError(f"{name}: binary likelihood returned values outside [0,1] (min={pmin:.4g}, max={pmax:.4g}).")
    return probs.clamp(eps, 1.0 - eps)


def latent_samples_to_binary_probabilities(
    model: Any,
    latent_samples: Tensor,
    *,
    eps: float = 1e-6,
    name: str = "latent samples",
    output_dim: int = -1,
) -> Tensor:
    """Map latent samples to probabilities with each model's own likelihood.

    For a single-output classifier this calls ``model.likelihood.forward``.
    For ModelList-like wrappers, the last output dimension is split and each
    submodel's likelihood is applied independently.  Consequently the default
    GPyTorch ``BernoulliLikelihood`` uses its probit link, while a custom
    logistic likelihood continues to use sigmoid without acquisition-side
    hard-coding.
    """
    submodels = getattr(model, "models", None)
    if submodels is None:
        return _conditional_probability(
            model,
            latent_samples,
            eps=eps,
            name=name,
        )

    submodels = list(submodels)
    if len(submodels) == 0:
        raise ValueError(f"{name}: model.models is empty.")

    dim = output_dim if output_dim >= 0 else latent_samples.ndim + output_dim
    if not 0 <= dim < latent_samples.ndim:
        raise IndexError(f"{name}: output_dim={output_dim} is invalid for shape {tuple(latent_samples.shape)}.")
    if latent_samples.shape[dim] != len(submodels):
        if len(submodels) == 1:
            return _conditional_probability(
                submodels[0],
                latent_samples,
                eps=eps,
                name=name,
            )
        raise RuntimeError(
            f"{name}: latent output dimension {latent_samples.shape[dim]} does not "
            f"match number of submodels {len(submodels)}. "
            f"shape={tuple(latent_samples.shape)}, output_dim={output_dim}."
        )

    outputs = []
    for index, (submodel, samples_i) in enumerate(zip(submodels, latent_samples.unbind(dim=dim))):
        outputs.append(
            _conditional_probability(
                submodel,
                samples_i,
                eps=eps,
                name=f"{name}[output={index}]",
            )
        )
    return torch.stack(outputs, dim=dim)


def values_to_binary_probabilities(
    model: Any,
    values: Tensor,
    *,
    eps: float = 1e-6,
    name: str = "binary values",
    output_dim: int = -1,
    values_are_probabilities: bool | None = None,
) -> Tensor:
    """Validate probabilities or transform latent values via likelihood.

    ``values_are_probabilities=False`` forces likelihood conversion even when
    every latent value happens to lie inside ``[0, 1]``.  ``None`` preserves the
    support behavior that infers the value space from the numeric range.
    """
    if not torch.isfinite(values).all():
        raise RuntimeError(f"{name}: values contain NaN or inf.")

    vmin = values.detach().min().item()
    vmax = values.detach().max().item()
    if values_are_probabilities is not False and vmin >= 0.0 and vmax <= 1.0:
        return values.clamp(eps, 1.0 - eps)

    return latent_samples_to_binary_probabilities(
        model,
        values,
        eps=eps,
        name=name,
        output_dim=output_dim,
    )


__all__ = [
    "latent_samples_to_binary_probabilities",
    "values_to_binary_probabilities",
]
