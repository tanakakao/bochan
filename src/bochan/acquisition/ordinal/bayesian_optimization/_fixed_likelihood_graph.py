"""Keep fitted ordinal likelihood parameters out of acquisition autograd graphs."""

from __future__ import annotations

from functools import wraps
from typing import Any, Literal

import torch
from gpytorch.utils.memoize import clear_cache_hook
from torch import Tensor

LinkType = Literal["auto", "probit", "logit"]


def _resolve_ordered_link(likelihood: Any, link: LinkType) -> Literal["probit", "logit"] | None:
    """Resolve a standard ordered-link family without guessing custom likelihoods."""

    if link in {"probit", "logit"}:
        return link

    declared = getattr(likelihood, "link", None)
    if callable(declared):
        declared = declared()
    marker = f"{declared or ''} {type(likelihood).__name__}".lower()
    if "logit" in marker or "logistic" in marker:
        return "logit"
    if "probit" in marker or "normal" in marker:
        return "probit"
    return None


def _ordered_probs_from_detached_cutpoints(
    *,
    latent: Tensor,
    cutpoints: Tensor,
    link: Literal["probit", "logit"],
    eps: float,
) -> Tensor:
    """Compute ordered probabilities while treating fitted cutpoints as constants."""

    cuts = cutpoints.detach().to(device=latent.device, dtype=latent.dtype).reshape(-1)
    z = cuts - latent.unsqueeze(-1)
    cdf = (
        torch.sigmoid(z)
        if link == "logit"
        else 0.5 * (1.0 + torch.erf(z / (2.0**0.5)))
    )

    probs = torch.cat(
        [cdf[..., :1], cdf[..., 1:] - cdf[..., :-1], 1.0 - cdf[..., -1:]],
        dim=-1,
    )
    probs = probs.clamp_min(float(eps))
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(float(eps))


def _patch_utility_objective_forward(module: Any) -> None:
    """Detach static baseline utilities while keeping candidate input gradients."""

    objective_cls = module.qMultiOutputOrdinalUtilityObjective
    current = objective_cls.forward
    if getattr(current, "_bochan_detaches_static_baseline", False):
        return

    original = current

    def forward(
        self,
        samples: Tensor,
        X: Tensor | None = None,
    ) -> Tensor:
        track_candidate_grad = torch.is_grad_enabled() and (
            X is None or bool(X.requires_grad)
        )
        with torch.set_grad_enabled(track_candidate_grad):
            return original(self, samples=samples, X=X)

    forward._bochan_detaches_static_baseline = True  # type: ignore[attr-defined]
    forward._bochan_original = original  # type: ignore[attr-defined]
    objective_cls.forward = forward


def _as_t_batch(X: Tensor) -> Tensor:
    """Match BoTorch's implicit singleton t-batch for two-dimensional inputs."""

    return X.unsqueeze(0) if X.ndim == 2 else X


def _variational_cache_key(X: Tensor) -> tuple[Any, ...]:
    """Return the input properties that determine variational cache batch shape."""

    return (
        tuple(X.shape[:-2]),
        X.device.type,
        X.device.index,
        X.dtype,
    )


def _prime_fixed_variational_caches(model: Any, X: Tensor) -> None:
    """Populate model-only prediction caches without an autograd graph.

    GPyTorch's ``VariationalStrategy`` memoizes the inducing covariance Cholesky
    factor while ignoring call arguments. Its shape nevertheless depends on the
    candidate t-batch through broadcasting. The factor depends only on the fitted
    model and inducing locations, not on candidate values, so candidate
    optimization should treat it as constant.
    """

    apply = getattr(model, "apply", None)
    if callable(apply):
        apply(clear_cache_hook)
    model.posterior(X)


def _patch_nehvi_forward(module: Any) -> None:
    """Prime detached variational caches for the actual candidate t-batch shape."""

    acquisition_cls = module.qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement
    current = acquisition_cls.forward
    if getattr(current, "_bochan_primes_variational_cache", False):
        return

    @wraps(current)
    def forward(self, X: Tensor) -> Tensor:
        X_t = _as_t_batch(X)
        cache_key = _variational_cache_key(X_t)
        if getattr(self, "_bochan_variational_cache_key", None) != cache_key:
            model = getattr(self, "model", None)
            if model is not None and hasattr(model, "posterior"):
                with torch.no_grad():
                    _prime_fixed_variational_caches(model, X_t)
                self._bochan_variational_cache_key = cache_key
        return current(self, X)

    forward._bochan_primes_variational_cache = True  # type: ignore[attr-defined]
    forward._bochan_original = current  # type: ignore[attr-defined]
    acquisition_cls.forward = forward


def apply_fixed_ordinal_likelihood_graph_support(module: Any) -> None:
    """Patch ordinal qNEHVI for stable repeated acquisition backward.

    Standard ordered-link cutpoints, transformed baseline utilities, and
    variational inducing-point factorizations are fixed after fitting. Candidate
    latent samples remain differentiable with respect to candidate inputs.
    """

    current = module.ordinal_probs_from_latent
    if not getattr(current, "_bochan_detaches_fitted_cutpoints", False):
        original = current

        def ordinal_probs_from_latent(
            latent: Tensor,
            likelihood: Any,
            *,
            num_classes: int,
            link: LinkType = "auto",
            eps: float = 1e-12,
        ) -> Tensor:
            resolved_link = _resolve_ordered_link(likelihood, link)
            if resolved_link is not None:
                try:
                    cutpoints = module._get_cutpoints(likelihood)
                except (AttributeError, ValueError):
                    cutpoints = None
                if cutpoints is not None and int(cutpoints.numel()) + 1 == int(num_classes):
                    return _ordered_probs_from_detached_cutpoints(
                        latent=latent,
                        cutpoints=cutpoints,
                        link=resolved_link,
                        eps=eps,
                    )

            return original(
                latent,
                likelihood,
                num_classes=num_classes,
                link=link,
                eps=eps,
            )

        ordinal_probs_from_latent._bochan_detaches_fitted_cutpoints = True  # type: ignore[attr-defined]
        ordinal_probs_from_latent._bochan_original = original  # type: ignore[attr-defined]
        module.ordinal_probs_from_latent = ordinal_probs_from_latent

    _patch_utility_objective_forward(module)
    _patch_nehvi_forward(module)


__all__ = ["apply_fixed_ordinal_likelihood_graph_support"]
