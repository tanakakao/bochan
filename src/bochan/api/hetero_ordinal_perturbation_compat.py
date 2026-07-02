"""InputPerturbation compatibility for heteroscedastic ordinal objectives."""

from __future__ import annotations

import math
from functools import wraps
from typing import Any

import torch
from torch import Tensor


_PATCHED = False


def _objective_n_w(objective: Any) -> int | None:
    """Read the perturbation count from a preprocessing objective."""

    if objective is None:
        return None
    for name in ("input_perturbation_n_w", "n_w"):
        value = getattr(objective, name, None)
        if value is not None:
            return int(value)
    return None


def _objective_utility_values(objective: Any):
    """Recover per-output utility vectors from the standard ordinal objective."""

    if objective is None:
        return None
    utility_table = getattr(objective, "utility_table", None)
    num_classes = getattr(objective, "num_classes", None)
    if not torch.is_tensor(utility_table) or not torch.is_tensor(num_classes):
        return None
    return [
        utility_table[i, : int(num_classes[i].item())]
        for i in range(int(num_classes.numel()))
    ]


def _aggregate_perturbations(
    values: Tensor,
    *,
    q: int,
    n_w: int,
    risk_type: str | None,
    alpha: float,
) -> Tensor:
    """Aggregate ``q * n_w`` utility values for a maximization objective."""

    m = int(values.shape[-1])
    values_w = values.reshape(*values.shape[:-2], q, n_w, m)
    if risk_type is None:
        return values_w.mean(dim=-2)
    if risk_type not in {"var", "cvar"}:
        raise ValueError(f"Unknown risk_type: {risk_type!r}.")
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("alpha must be in (0, 1].")

    k = max(1, int(math.ceil(n_w * float(alpha))))
    lower_tail = torch.sort(values_w, dim=-2, descending=False).values[..., :k, :]
    if risk_type == "var":
        return lower_tail[..., -1, :]
    return lower_tail.mean(dim=-2)


def _is_standard_ordinal_preprocessor(objective: Any) -> bool:
    """Return whether ``objective`` already performs latent-to-utility mapping."""

    if objective is None:
        return False
    return objective.__class__.__name__ in {
        "qMultiOutputOrdinalUtilityObjective",
        "MultiOutputOrdinalInputPerturbationObjective",
    }


def _align_summary_to_shape(
    module: Any,
    value: Tensor,
    *,
    X_raw: Tensor,
    X_shape: Tensor,
    m: int,
    name: str,
) -> Tensor:
    """Accept hetero summaries returned at either raw-q or expanded-q shape."""

    raw_q = int(X_raw.shape[-2])
    expanded_q = int(X_shape.shape[-2])
    if (
        expanded_q != raw_q
        and value.ndim >= 2
        and int(value.shape[-2]) == raw_q
    ):
        value = value.repeat_interleave(expanded_q // raw_q, dim=-2)
    return module._align_pointwise_to_X_q_m(
        value,
        X_shape,
        m=m,
        name=name,
    )


def _patch_utility_forward() -> None:
    """Patch hetero utility conversion so expanded q is reduced after adjustment."""

    from bochan.acquisition.ordinal.bayesian_optimization import (
        hetero_multi_output as module,
    )

    cls = module.qHeteroMultiOutputOrdinalUtilityObjective
    if getattr(cls, "_bochan_input_perturbation_patched", False):
        return

    def compatible_forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        if X is None:
            raise ValueError(
                "X must be provided for qHeteroMultiOutputOrdinalUtilityObjective."
            )

        Xq = module._ensure_q_batch(X)
        base_objective = self.base_objective
        n_w = _objective_n_w(base_objective)

        ordinal_likelihoods = self.ordinal_likelihoods
        if ordinal_likelihoods is None:
            ordinal_likelihoods = getattr(
                base_objective,
                "ordinal_likelihoods",
                None,
            )
        likes = module._extract_ordinal_likelihoods(
            self.model,
            ordinal_likelihoods,
        )
        m = len(likes)

        utility_values = self.utility_values_list
        if utility_values is None:
            utility_values = _objective_utility_values(base_objective)

        objective_signs = self.objective_signs
        if objective_signs is None:
            objective_signs = getattr(base_objective, "objective_signs", None)

        utilities = module.ordinal_latent_samples_to_expected_utility(
            samples,
            model=self.model,
            utility_values_list=utility_values,
            ordinal_likelihoods=likes,
            latent_to_probs_list=self.latent_to_probs_list,
            objective_signs=objective_signs,
            eps=self.eps,
        )

        raw_q = int(Xq.shape[-2])
        q_like = int(utilities.shape[-2])
        if q_like != raw_q:
            if raw_q <= 0 or q_like % raw_q != 0:
                raise RuntimeError(
                    "Hetero ordinal InputPerturbation could not align q dimensions: "
                    f"raw_q={raw_q}, utility_q={q_like}."
                )
            inferred_n_w = q_like // raw_q
            if n_w is None:
                n_w = inferred_n_w
            elif int(n_w) != inferred_n_w:
                raise RuntimeError(
                    "Configured n_w does not match hetero ordinal posterior shape: "
                    f"configured={n_w}, inferred={inferred_n_w}."
                )
            X_shape = Xq.repeat_interleave(inferred_n_w, dim=-2)
        else:
            X_shape = Xq

        utilities = module._align_mo_samples_to_X(
            utilities,
            X_shape,
            m=m,
            name="qHeteroMultiOutputOrdinalUtilityObjective.utilities",
        )

        with torch.no_grad():
            summary = module.stack_multi_summaries(
                self.model,
                Xq,
                utility_values_list=utility_values,
                noise_penalties=0.0,
                variance_scales=self.variance_scale,
                taus=self.tau,
                default_sigmas=self.default_sigma,
                eps=self.eps,
            )
            robust_mean = _align_summary_to_shape(
                module,
                summary["robust_mean"],
                X_raw=Xq,
                X_shape=X_shape,
                m=m,
                name="qHeteroMultiOutputOrdinalUtilityObjective.robust_mean",
            )
            sigma = summary.get("sigma", summary.get("total_std", None))
            if sigma is not None:
                sigma = _align_summary_to_shape(
                    module,
                    sigma,
                    X_raw=Xq,
                    X_shape=X_shape,
                    m=m,
                    name="qHeteroMultiOutputOrdinalUtilityObjective.sigma",
                )

            if objective_signs is not None:
                signs = torch.as_tensor(
                    objective_signs,
                    device=robust_mean.device,
                    dtype=robust_mean.dtype,
                ).reshape(-1)
                if signs.numel() != m:
                    raise ValueError(
                        f"objective_signs must have length {m}. Got {signs.numel()}."
                    )
                robust_mean = robust_mean * signs.view(
                    *((1,) * (robust_mean.ndim - 1)),
                    m,
                )

        adjusted = robust_mean.unsqueeze(0) + self.beta * (
            utilities - robust_mean.unsqueeze(0)
        )
        if sigma is not None:
            penalties = module._expand_scalar_or_list(
                self.noise_penalty,
                m,
                "noise_penalty",
            )
            penalties = torch.as_tensor(
                penalties,
                device=utilities.device,
                dtype=utilities.dtype,
            ).reshape(-1)
            adjusted = adjusted - sigma.unsqueeze(0) * penalties.view(
                *((1,) * (adjusted.ndim - 1)),
                m,
            )

        adjusted = module._align_mo_samples_to_X(
            adjusted,
            X_shape,
            m=m,
            name="qHeteroMultiOutputOrdinalUtilityObjective.adjusted",
        )

        if n_w is not None and int(n_w) > 1 and adjusted.shape[-2] != raw_q:
            risk_type = getattr(base_objective, "risk_type", None)
            risk_alpha = getattr(
                base_objective,
                "risk_alpha",
                getattr(base_objective, "alpha", 0.5),
            )
            adjusted = _aggregate_perturbations(
                adjusted,
                q=raw_q,
                n_w=int(n_w),
                risk_type=risk_type,
                alpha=float(risk_alpha),
            )

        if base_objective is None or _is_standard_ordinal_preprocessor(
            base_objective
        ):
            return adjusted
        return base_objective(adjusted, X=Xq)

    cls.forward = compatible_forward
    cls._bochan_input_perturbation_patched = True


def _patch_hetero_nparego_wrapper() -> None:
    """Preserve the inferred perturbation objective in hetero ordinal NParEGO."""

    import bochan.acquisition.ordinal.bayesian_optimization as package

    original = package.qHeteroMultiOutputOrdinalNParEGO
    if getattr(original, "_bochan_input_perturbation_patched", False):
        return

    @wraps(original)
    def compatible(*args: Any, objective=None, **kwargs: Any):
        acquisition = original(*args, objective=None, **kwargs)
        if objective is not None:
            acquisition.utility_objective.base_objective = objective
            acquisition.base_objective = objective
        return acquisition

    compatible._bochan_input_perturbation_patched = True
    compatible._bochan_original = original
    package.qHeteroMultiOutputOrdinalNParEGO = compatible


def apply_hetero_ordinal_perturbation_compat() -> None:
    """Install hetero ordinal InputPerturbation compatibility once."""

    global _PATCHED
    if _PATCHED:
        return
    _patch_utility_forward()
    _patch_hetero_nparego_wrapper()
    _PATCHED = True


__all__ = ["apply_hetero_ordinal_perturbation_compat"]
