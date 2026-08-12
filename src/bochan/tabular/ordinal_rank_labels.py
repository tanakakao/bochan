"""Resolve string labels used by tabular ordinal-rank constraints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_ORDINAL_RANK_KINDS = {"ordinal", "ordinal_rank", "ordinalrank", "rank"}


def _constraint_kind(value: Mapping[str, Any]) -> str:
    """Return the normalized constraint kind used by serializable configs."""

    return str(
        value.get("kind")
        or value.get("type")
        or value.get("constraint_type")
        or "feasibility"
    ).lower()


def resolve_ordinal_rank_constraint(
    value: Any,
    *,
    target_names: list[Any],
    target_category_maps: Mapping[Any, Mapping[Any, int]] | None,
) -> Any:
    """Resolve one string ordinal rank to the encoded integer class index."""

    if not isinstance(value, Mapping):
        return value
    if _constraint_kind(value) not in _ORDINAL_RANK_KINDS:
        return value

    rank = value.get("rank")
    output = value.get("output")
    if not isinstance(rank, str) or output is None:
        return value

    from .optimizer_api import _resolve_target_class_value

    resolved = dict(value)
    resolved["rank"] = _resolve_target_class_value(
        rank,
        output=output,
        target_names=target_names,
        target_category_maps=target_category_maps,
    )
    return resolved


def resolve_ordinal_rank_config(
    value: Any,
    *,
    target_names: list[Any],
    target_category_maps: Mapping[Any, Mapping[Any, int]] | None,
) -> Any:
    """Resolve ordinal ranks nested in an outcome-constraint configuration."""

    if not isinstance(value, Mapping):
        return value
    resolved = dict(value)
    constraints = resolved.get("constraints")
    if constraints is None:
        return resolved
    constraint_values = (
        [constraints]
        if isinstance(constraints, Mapping)
        else list(constraints)
    )
    resolved["constraints"] = [
        resolve_ordinal_rank_constraint(
            item,
            target_names=target_names,
            target_category_maps=target_category_maps,
        )
        for item in constraint_values
    ]
    return resolved


def resolve_acquisition_ordinal_ranks(
    value: Any,
    *,
    target_names: list[Any],
    target_category_maps: Mapping[Any, Mapping[Any, int]] | None,
) -> Any:
    """Resolve ordinal ranks nested in a mapping-style acquisition config."""

    if not isinstance(value, Mapping):
        return value
    if "outcome_constraint_config" not in value:
        return value
    resolved = dict(value)
    resolved["outcome_constraint_config"] = resolve_ordinal_rank_config(
        resolved["outcome_constraint_config"],
        target_names=target_names,
        target_category_maps=target_category_maps,
    )
    return resolved


__all__ = [
    "resolve_acquisition_ordinal_ranks",
    "resolve_ordinal_rank_config",
    "resolve_ordinal_rank_constraint",
]
