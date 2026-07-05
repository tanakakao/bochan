"""Compatibility helpers for wide correlated ordinal acquisitions."""

from __future__ import annotations

from typing import Any

_APPLIED = False


def apply_ordinal_multitask_compat() -> None:
    """Install wide posterior and shared ordinal-likelihood compatibility.

    The long-format multi-task ordinal model learns one shared ordinal likelihood
    and a task covariance. Its wide public interface exposes one objective per
    task. Multi-output utility objectives require one likelihood entry per public
    output, so repeat the shared likelihood without copying its parameters.
    """

    from bochan.acquisition.wide_posterior_event_compat import (
        apply_wide_posterior_event_compat,
    )

    apply_wide_posterior_event_compat()

    global _APPLIED
    if _APPLIED:
        return

    from bochan.acquisition.ordinal.bayesian_optimization import multi_output as module

    original = module._extract_ordinal_likelihoods
    if getattr(original, "_bochan_wide_multitask_compatible", False):
        _APPLIED = True
        return

    def compatible_extract(
        model: Any,
        ordinal_likelihoods: Any = None,
    ) -> list[Any]:
        likelihoods = list(original(model, ordinal_likelihoods))
        try:
            num_outputs = int(getattr(model, "num_outputs", 1))
        except (TypeError, ValueError):
            num_outputs = 1
        if len(likelihoods) == 1 and num_outputs > 1:
            return likelihoods * num_outputs
        return likelihoods

    compatible_extract._bochan_wide_multitask_compatible = True  # type: ignore[attr-defined]
    compatible_extract._bochan_original = original  # type: ignore[attr-defined]
    module._extract_ordinal_likelihoods = compatible_extract
    _APPLIED = True


__all__ = ["apply_ordinal_multitask_compat"]
