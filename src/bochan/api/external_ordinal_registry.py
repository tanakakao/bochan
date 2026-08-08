"""Register cumulative external ordinal models with the default API registry."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


_NORMAL_ORDINAL_MODELS = {
    "ngboost": (
        "bochan.models.ordinal.external",
        "NGBoostOrdinalModel",
    ),
    "ngboost_ensemble": (
        "bochan.models.ordinal.external",
        "NGBoostOrdinalEnsembleModel",
    ),
    "random_forest": (
        "bochan.models.ordinal.external",
        "RandomForestOrdinalModel",
    ),
}

_MIXED_ORDINAL_MODELS = {
    "ngboost": (
        "bochan.models.ordinal.external",
        "NGBoostMixedOrdinalModel",
    ),
    "ngboost_ensemble": (
        "bochan.models.ordinal.external",
        "NGBoostMixedOrdinalEnsembleModel",
    ),
    "random_forest": (
        "bochan.models.ordinal.external",
        "RandomForestMixedOrdinalModel",
    ),
}


def register_external_ordinal_models(registry: Any) -> None:
    """Add external ordinal model paths without replacing existing mappings."""
    tree: MutableMapping[str, Any] = registry.raw()
    for input_type, additions in (
        ("normal", _NORMAL_ORDINAL_MODELS),
        ("mixed", _MIXED_ORDINAL_MODELS),
    ):
        ordinal = tree[input_type]["ordinal"]
        for key, path in additions.items():
            existing = ordinal.get(key)
            if existing is not None and existing != path:
                raise RuntimeError(
                    "External ordinal registry would overwrite an existing model key: "
                    f"{input_type}/ordinal/{key}."
                )
            ordinal[key] = path


__all__ = ["register_external_ordinal_models"]
