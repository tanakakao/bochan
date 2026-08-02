"""Response-scale statistics shared by non-Gaussian acquisitions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from botorch.sampling.get_sampler import get_sampler
from torch import Tensor


@dataclass(frozen=True)
class NonGaussianResponseStats:
    """Epistemic and aleatoric response statistics (the last axis is output)."""

    response_mean_samples: Tensor
    response_mean: Tensor
    response_mean_variance: Tensor
    base_expected_observation_variance: Tensor
    extra_heteroscedastic_variance: Tensor
    expected_observation_variance: Tensor
    total_observation_variance: Tensor
    expected_observation_entropy: Tensor
    predictive_entropy_proxy: Tensor
    bald_variance_ratio_proxy: Tensor
    bald_entropy_difference_proxy: Tensor
    entropy_is_proxy: bool = False


def ensure_q_batch(X: Tensor) -> Tensor:
    """Return ``X`` with explicit t-batch and q dimensions."""
    if not torch.is_tensor(X):
        raise TypeError(f"X must be a Tensor. Got {type(X)}.")
    if X.ndim == 1:
        return X.view(1, 1, -1)
    return X.unsqueeze(0) if X.ndim == 2 else X


def safe_normal_cdf(z: Tensor) -> Tensor:
    """Evaluate a device/dtype preserving standard-normal CDF."""
    return torch.special.ndtr(z)


def safe_logdet(covar: Tensor, jitter: float = 1e-6) -> Tensor:
    """Compute a checked log determinant after symmetrisation and jitter."""
    covar = 0.5 * (covar + covar.transpose(-1, -2))
    eye = torch.eye(covar.shape[-1], device=covar.device, dtype=covar.dtype)
    sign, value = torch.linalg.slogdet(covar + jitter * eye)
    if torch.any(sign <= 0):
        raise RuntimeError("Covariance remained non-positive definite after jitter.")
    return value


def _as_output(t: Tensor, X: Tensor) -> Tensor:
    """Normalize a pointwise posterior tensor to ``batch x q x m``."""
    q = X.shape[-2]
    if t.shape[-1] == q:
        return t.unsqueeze(-1)
    if t.ndim >= 2 and t.shape[-2] == q:
        return t
    if t.shape[-1] % q == 0:
        return t.reshape(*t.shape[:-1], q, t.shape[-1] // q)
    raise RuntimeError(f"Cannot normalize posterior tensor shape {tuple(t.shape)} for q={q}.")


def _likelihood(model: Any) -> Any:
    likelihood = getattr(model, "likelihood", None)
    if likelihood is None:
        raise TypeError(f"{type(model).__name__} does not expose a non-Gaussian likelihood.")
    return likelihood


def observation_variance_from_mean(model: Any, mean: Tensor) -> Tensor:
    """Evaluate the family observation variance from response means."""
    like = _likelihood(model)
    name = type(like).__name__.lower()
    if "beta" in name:
        phi = like.concentration.to(mean).clamp_min(1e-12)
        mu = mean.clamp(1e-12, 1 - 1e-12)
        return mu * (1 - mu) / (phi + 1)
    if "gamma" in name:
        k = like.concentration.to(mean).clamp_min(1e-12)
        return mean.clamp_min(1e-12).square() / k
    if "negativebinomial" in name or "negative_binomial" in name:
        r = like.total_count.to(mean).clamp_min(1e-12)
        mu = mean.clamp_min(1e-12)
        return mu + mu.square() / r
    if "poisson" in name:
        return mean.clamp_min(1e-12)
    raise TypeError(f"Unsupported likelihood {type(like).__name__} on {type(model).__name__}.")


def _entropy(model: Any, mean: Tensor, variance: Tensor) -> tuple[Tensor, bool]:
    """Return conditional entropy, using only the documented Gaussian proxy fallback."""
    like = _likelihood(model)
    try:
        if "poisson" in type(like).__name__.lower():
            dist = torch.distributions.Poisson(mean.clamp_min(1e-12))
        elif "negative" in type(like).__name__.lower():
            r = like.total_count.to(mean).clamp_min(1e-12)
            dist = torch.distributions.NegativeBinomial(r.expand_as(mean), logits=(mean / r).log())
        elif "beta" in type(like).__name__.lower():
            phi = like.concentration.to(mean)
            mu = mean.clamp(1e-12, 1 - 1e-12)
            dist = torch.distributions.Beta(mu * phi, (1 - mu) * phi)
        else:
            k = like.concentration.to(mean)
            dist = torch.distributions.Gamma(k.expand_as(mean), k / mean.clamp_min(1e-12))
        return dist.entropy(), False
    except NotImplementedError:
        c = mean.new_tensor(2 * torch.pi * torch.e)
        return 0.5 * torch.log(c * variance.clamp_min(1e-12)), True


def _extra_variance(model: Any, X: Tensor, like: Tensor) -> Tensor:
    """Get explicitly modelled heteroscedastic variance or exact zeros."""
    fn = getattr(model, "extra_observation_variance", None)
    if callable(fn):
        return fn(X, like=like).to(like)
    fn = getattr(model, "predict_noise_var", None)
    if callable(fn):
        return fn(X, ref_like=like).to(like)
    return torch.zeros_like(like)


def _single_stats(model: Any, X: Tensor, sampler: Any, eps: float) -> NonGaussianResponseStats:
    posterior = model.posterior(X, observation_noise=False)
    samples = _as_output(sampler(posterior), X)
    mean = samples.mean(0)
    epistemic = samples.var(0, unbiased=False).clamp_min(eps)
    conditional = observation_variance_from_mean(model, samples).clamp_min(eps)
    base = conditional.mean(0)
    extra = _as_output(_extra_variance(model, X, mean), X).clamp_min(0)
    expected = (base + extra).clamp_min(eps)
    total = (epistemic + expected).clamp_min(eps)
    entropy_samples, proxy = _entropy(model, samples, conditional)
    expected_entropy = entropy_samples.mean(0)
    predictive = 0.5 * torch.log(mean.new_tensor(2 * torch.pi * torch.e) * total)
    ratio = 0.5 * torch.log(total / expected)
    difference = (predictive - expected_entropy).clamp_min(0)
    return NonGaussianResponseStats(samples, mean, epistemic, base, extra, expected, total,
                                    expected_entropy, predictive, ratio, difference, proxy)


def non_gaussian_response_stats(model: Any, X: Tensor, *, sampler: Any | None = None,
                                sample_shape: torch.Size = torch.Size([128]), seed: int | None = None,
                                num_samples: int | None = None, eps: float = 1e-12) -> NonGaussianResponseStats:
    """Compute raw response-scale statistics using public posteriors and fixed samplers.

    Args:
        model: A supported non-Gaussian model or ``NonGaussianModelList``.
        X: Raw-space candidates.
        sampler: Persistent BoTorch MC sampler. If omitted one is constructed.
        sample_shape: MC sample shape.
        seed: Sobol sampler seed.
        num_samples: Deprecated compatible spelling for a one-dimensional sample shape.
        eps: Positive numerical floor.
    """
    X = ensure_q_batch(X)
    if num_samples is not None:
        sample_shape = torch.Size([num_samples])
    if hasattr(model, "models") and getattr(model, "is_non_gaussian_model_list", False):
        parts = [non_gaussian_response_stats(m, X, sample_shape=sample_shape, seed=seed, eps=eps) for m in model.models]
        fields = [f.name for f in NonGaussianResponseStats.__dataclass_fields__.values() if f.name != "entropy_is_proxy"]
        values = [torch.cat([getattr(p, name) for p in parts], dim=-1) for name in fields]
        return NonGaussianResponseStats(*values, any(p.entropy_is_proxy for p in parts))
    posterior = model.posterior(X, observation_noise=False)
    sampler = sampler or get_sampler(posterior, sample_shape=sample_shape, seed=seed)
    return _single_stats(model, X, sampler, eps)


__all__ = ["NonGaussianResponseStats", "ensure_q_batch", "non_gaussian_response_stats",
           "observation_variance_from_mean", "safe_logdet", "safe_normal_cdf"]
