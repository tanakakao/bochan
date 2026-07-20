"""Compatibility defaults for candidate repair constraint semantics."""

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
    """Install top-level inequality fallback semantics on the factory helper."""

    from . import factory as factory_module

    current = factory_module._build_post_processing_func
    if getattr(current, "_bochan_repair_constraint_defaults", False):
        return

    def build_post_processing_func(config: Any, bounds: Any):
        resolved = _with_botorch_fallback_inequality_sense(config)
        return current(resolved, bounds)

    build_post_processing_func._bochan_repair_constraint_defaults = True
    factory_module._build_post_processing_func = build_post_processing_func


__all__ = [
    "_with_botorch_fallback_inequality_sense",
    "apply_repair_constraint_defaults",
]
