from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor


def squeeze_output_dim(t: Tensor) -> Tensor:
    """Drop a singleton output dimension while preserving q=1."""
    if torch.is_tensor(t) and t.ndim >= 1 and t.shape[-1] == 1:
        return t.squeeze(-1)
    return t


def ensure_q_batch(X: Tensor) -> Tensor:
    if not torch.is_tensor(X):
        raise TypeError(f"X must be a Tensor. Got {type(X)}.")
    if X.ndim == 1:
        return X.view(1, 1, -1)
    if X.ndim == 2:
        return X.unsqueeze(0)
    return X


def safe_normal_cdf(z: Tensor) -> Tensor:
    return 0.5 * (1.0 + torch.erf(z / torch.sqrt(torch.as_tensor(2.0, device=z.device, dtype=z.dtype))))


def safe_logdet(covar: Tensor, jitter: float = 1e-6) -> Tensor:
    q = covar.shape[-1]
    eye = torch.eye(q, device=covar.device, dtype=covar.dtype)
    while eye.ndim < covar.ndim:
        eye = eye.unsqueeze(0)
    covar = 0.5 * (covar + covar.transpose(-1, -2))
    return torch.linalg.slogdet(covar + jitter * eye).logabsdet


def _latent_posterior(model: Any, X: Tensor):
    latent_fn = getattr(model, "latent_posterior", None)
    if callable(latent_fn):
        return latent_fn(X)
    return model.posterior(X, observation_noise=False)


def _distribution_from_latent(model: Any, function_samples: Tensor):
    likelihood = getattr(model, "likelihood", None)
    if likelihood is None or not callable(likelihood):
        return None
    return likelihood(function_samples)


def non_gaussian_response_stats(
    model: Any,
    X: Tensor,
    *,
    num_samples: int = 64,
    eps: float = 1e-12,
) -> dict[str, Tensor]:
    """Estimate response-scale statistics for non-Gaussian GP wrappers.

    The current Poisson / Beta / Gamma / Negative Binomial wrappers expose
    ``latent_posterior(X)`` and a likelihood whose ``forward`` method maps
    latent samples to a torch distribution.  This helper samples latent f,
    converts samples to observation distributions, and returns pointwise
    response-scale quantities.

    Returned tensors are intentionally pointwise, typically shaped
    ``batch_shape x q`` or ``batch_shape x q x 1``.  Acquisition base classes
    perform final alignment and q reduction.
    """
    if int(num_samples) <= 1:
        raise ValueError("num_samples must be greater than 1 for sample-based non-Gaussian acquisitions.")

    Xq = ensure_q_batch(X)
    latent_post = _latent_posterior(model, Xq)
    f_samples = latent_post.rsample(sample_shape=torch.Size([int(num_samples)]))
    dist = _distribution_from_latent(model, f_samples)

    if dist is None:
        response_mean_samples = squeeze_output_dim(f_samples)
        conditional_variance_samples = torch.zeros_like(response_mean_samples)
        conditional_entropy_samples = torch.zeros_like(response_mean_samples)
    else:
        response_mean_samples = squeeze_output_dim(dist.mean)
        try:
            conditional_variance_samples = squeeze_output_dim(dist.variance).clamp_min(0.0)
        except Exception:
            conditional_variance_samples = torch.zeros_like(response_mean_samples)
        try:
            conditional_entropy_samples = squeeze_output_dim(dist.entropy())
        except Exception:
            conditional_entropy_samples = 0.5 * torch.log(
                torch.as_tensor(2.0 * torch.pi * torch.e, device=response_mean_samples.device, dtype=response_mean_samples.dtype)
                * conditional_variance_samples.clamp_min(eps)
            )

    response_mean = response_mean_samples.mean(dim=0)
    response_mean_variance = response_mean_samples.var(dim=0, unbiased=False).clamp_min(0.0)
    expected_observation_variance = conditional_variance_samples.mean(dim=0).clamp_min(0.0)
    total_observation_variance = (response_mean_variance + expected_observation_variance).clamp_min(0.0)
    expected_observation_entropy = conditional_entropy_samples.mean(dim=0)
    predictive_entropy_proxy = 0.5 * torch.log(
        torch.as_tensor(2.0 * torch.pi * torch.e, device=response_mean_samples.device, dtype=response_mean_samples.dtype)
        * total_observation_variance.clamp_min(eps)
    )
    bald_proxy = 0.5 * torch.log(
        total_observation_variance.clamp_min(eps) / expected_observation_variance.clamp_min(eps)
    )

    return {
        "response_mean_samples": response_mean_samples,
        "response_mean": response_mean,
        "response_mean_variance": response_mean_variance,
        "expected_observation_variance": expected_observation_variance,
        "total_observation_variance": total_observation_variance,
        "expected_observation_entropy": expected_observation_entropy,
        "predictive_entropy_proxy": predictive_entropy_proxy,
        "bald_proxy": bald_proxy,
    }
