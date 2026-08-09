"""Candidate-repair constraint semantics used by the core optimizer factory."""

from __future__ import annotations

from dataclasses import replace
from typing import Any


def _with_botorch_fallback_inequality_sense(config: Any) -> Any:
    """Use ``ge`` when repair falls back to top-level BoTorch inequalities.

    ``OptimizeConfig.inequality_constraints`` follows BoTorch's canonical
    ``a^T x >= rhs`` convention. ``CandidateRepairConfig`` also supports its own
    constraints whose sense may be either ``le`` or ``ge``. When no repair-local
    inequalities are supplied, the post-processing factory falls back to the
    top-level optimizer constraints, so those constraints must always be
    interpreted with ``ge`` semantics regardless of the repair-local default.
    """

    repair = getattr(config, "repair_config", None)
    if repair is None:
        return config
    if getattr(repair, "inequality_constraints", None) is not None:
        return config
    if getattr(config, "inequality_constraints", None) is None:
        return config
    if str(getattr(repair, "inequality_sense", "le")).lower() == "ge":
        return config
    return replace(
        config,
        repair_config=replace(repair, inequality_sense="ge"),
    )


def apply_repair_constraint_defaults() -> None:
    """Compatibility no-op; the factory resolves this default directly."""

    return None


__all__ = [
    "_with_botorch_fallback_inequality_sense",
    "apply_repair_constraint_defaults",
]
