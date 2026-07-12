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


def _patch_utility_objective_forward(module: Any) -> None:
    """Detach static baseline utilities while keeping candidate input gradients.

    BoTorch constructs qNEHVI by evaluating the objective on ``X_baseline`` and
    caching the transformed values. Variational and correlated multi-task models
    can make those baseline posterior samples depend on model parameters and
    inducing locations. Reusing that cache after an optimizer step then traverses
    a stale LinearOperator graph and may raise an in-place version error.

    Acquisition optimization only needs gradients with respect to candidate
    inputs. A baseline tensor does not require gradients, whereas the candidate
    tensor supplied by gradient optimizers does. Use that distinction to evaluate
    static baselines under ``no_grad`` and preserve normal candidate gradients.
    """

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


def apply_fixed_ordinal_likelihood_graph_support(module: Any) -> None:
    """Patch ordinal utility conversion for stable repeated acquisition backward.

    qNEHVI caches transformed baseline values. Standard ordered-link cutpoints and
    the baseline posterior itself must be treated as fitted constants, while the
    latent candidate samples remain differentiable with respect to candidate
    inputs.
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


__all__ = ["apply_fixed_ordinal_likelihood_graph_support"]
