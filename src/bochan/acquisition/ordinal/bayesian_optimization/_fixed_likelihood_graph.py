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


def _positional_argument(args: tuple[Any, ...], index: int) -> Any:
    """Return a positional constructor argument after ``self`` when available."""

    return args[index] if len(args) > index else None


def _prime_fixed_variational_caches(model: Any, X_baseline: Any) -> None:
    """Populate model-only prediction caches without an autograd graph.

    GPyTorch's variational strategy memoizes the inducing covariance Cholesky
    factor. It depends only on the fitted model and inducing locations, not on the
    candidate input. During acquisition optimization it must therefore behave as
    a constant. Clearing potentially graph-bearing caches and evaluating one
    baseline posterior under ``no_grad`` creates a reusable detached factor while
    preserving gradients through candidate-to-inducing covariances later.
    """

    if model is None or X_baseline is None or not hasattr(model, "posterior"):
        return

    apply = getattr(model, "apply", None)
    if callable(apply):
        apply(clear_cache_hook)

    model.posterior(X_baseline)


def _patch_nehvi_baseline_initialization(module: Any) -> None:
    """Build qNEHVI baseline state and variational caches as fitted constants."""

    acquisition_cls = module.qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement
    current = acquisition_cls.__init__
    if getattr(current, "_bochan_detaches_baseline_initialization", False):
        return

    @wraps(current)
    def init(self, *args, **kwargs) -> None:
        model = kwargs.get("model", _positional_argument(args, 0))
        X_baseline = kwargs.get("X_baseline", _positional_argument(args, 2))
        with torch.no_grad():
            current(self, *args, **kwargs)
            _prime_fixed_variational_caches(model, X_baseline)

    init._bochan_detaches_baseline_initialization = True  # type: ignore[attr-defined]
    init._bochan_original = current  # type: ignore[attr-defined]
    acquisition_cls.__init__ = init


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
    _patch_nehvi_baseline_initialization(module)


__all__ = ["apply_fixed_ordinal_likelihood_graph_support"]
