"""Keep fitted ordinal likelihood parameters out of acquisition autograd graphs."""

from __future__ import annotations

from typing import Any, Literal

import torch
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


def apply_fixed_ordinal_likelihood_graph_compat(module: Any) -> None:
    """Patch ordinal probability conversion for stable repeated acquisition backward.

    qNEHVI caches transformed baseline values. When the transformation includes
    trainable ordinal cutpoints, that cache retains a graph through the fitted
    likelihood parameters and the second optimizer closure attempts to traverse
    an already-freed graph. Candidate optimization must differentiate only with
    respect to candidate inputs, so standard ordered-link cutpoints are detached.
    """

    current = module.ordinal_probs_from_latent
    if getattr(current, "_bochan_detaches_fitted_cutpoints", False):
        return

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


__all__ = ["apply_fixed_ordinal_likelihood_graph_compat"]
