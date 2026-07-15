'''Resolve string labels used by tabular ordinal-rank constraints.'''

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_APPLIED = False
_ORDINAL_RANK_KINDS = {"ordinal", "ordinal_rank", "ordinalrank", "rank"}


def _constraint_kind(value: Mapping[str, Any]) -> str:
    '''Return the normalized constraint kind used by serializable configs.'''

    return str(
        value.get("kind")
        or value.get("type")
        or value.get("constraint_type")
        or "feasibility"
    ).lower()


def apply_tabular_ordinal_rank_labels() -> None:
    '''Resolve string ordinal ranks through fitted target category mappings.

    The core ordinal constraint spec stores an integer rank.  The tabular API,
    however, also retains mappings from original target labels to encoded class
    indices.  This extension applies the same label-resolution rules already
    used for ``target_class`` / ``target_classes`` to ordinal ``rank`` values,
    allowing configurations such as ``{"rank": "medium"}``.
    '''

    global _APPLIED
    if _APPLIED:
        return

    from . import optimizer_api

    original_resolver = optimizer_api._resolve_constraint_target_classes
    if getattr(original_resolver, "_bochan_resolves_string_ordinal_rank", False):
        _APPLIED = True
        return

    def resolve_constraint_labels(
        value: Any,
        *,
        target_names: list[Any],
        target_category_maps: Mapping[Any, Mapping[Any, int]] | None,
    ) -> Any:
        resolved = original_resolver(
            value,
            target_names=target_names,
            target_category_maps=target_category_maps,
        )
        if not isinstance(resolved, Mapping):
            return resolved
        if _constraint_kind(resolved) not in _ORDINAL_RANK_KINDS:
            return resolved

        rank = resolved.get("rank")
        output = resolved.get("output")
        if not isinstance(rank, str) or output is None:
            return resolved

        updated = dict(resolved)
        updated["rank"] = optimizer_api._resolve_target_class_value(
            rank,
            output=output,
            target_names=target_names,
            target_category_maps=target_category_maps,
        )
        return updated

    setattr(resolve_constraint_labels, "_bochan_resolves_string_ordinal_rank", True)
    optimizer_api._resolve_constraint_target_classes = resolve_constraint_labels
    _APPLIED = True


__all__ = ["apply_tabular_ordinal_rank_labels"]
