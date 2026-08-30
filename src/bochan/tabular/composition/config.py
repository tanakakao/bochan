"""Configuration normalization for tabular composition sites."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_SITE_DEFAULTS: dict[str, Any] = {
    "normalization": "atomic_fraction",
    "representation": "ilr",
    "reference_element": None,
    "pseudocount": 1e-12,
    "include_descriptors": False,
    "descriptor_properties": ("atomic_number", "atomic_weight"),
    "descriptor_statistics": ("mean", "std", "min", "max", "range"),
    "element_properties": {},
    "prefix": None,
    "precision": 6,
    "total": 1.0,
    "bounds": {},
    "steps": {},
    "min_components": 1,
    "max_components": None,
    "required_components": (),
    "forbidden_components": (),
    "support_selection": "repair",
    "best_subset_strategy": None,
    "best_subset_max_combinations": None,
    "best_subset_beam_width": None,
    "best_subset_beam_steps": None,
    "best_subset_max_evaluations": None,
    "coordinate_bounds": (-8.0, 8.0),
}

_BEST_SUBSET_INT_SETTINGS = {
    "best_subset_max_combinations": 1,
    "best_subset_beam_width": 1,
    "best_subset_beam_steps": 0,
    "best_subset_max_evaluations": 1,
}


def _normalize_best_subset_settings(name: str, resolved: dict[str, Any]) -> None:
    strategy = resolved.get("best_subset_strategy")
    if strategy is not None:
        strategy = str(strategy).lower()
        if strategy not in {"exact", "beam", "auto"}:
            raise ValueError(
                f"Composition site {name!r} best_subset_strategy must be exact, beam, or auto."
            )
        resolved["best_subset_strategy"] = strategy

    for key, minimum in _BEST_SUBSET_INT_SETTINGS.items():
        value = resolved.get(key)
        if value is None:
            continue
        value = int(value)
        if value < minimum:
            raise ValueError(
                f"Composition site {name!r} {key} must be >= {minimum}."
            )
        resolved[key] = value


def normalize_composition_sites(
    sites: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Normalize canonical formula-column composition site declarations."""

    if not sites:
        return {}

    normalized: dict[str, dict[str, Any]] = {}
    allowed = {"column", "elements", *_SITE_DEFAULTS}
    for raw_name, raw_config in sites.items():
        name = str(raw_name)
        if not isinstance(raw_config, Mapping):
            raise TypeError(f"Composition site {name!r} must be a mapping.")

        config = dict(raw_config)
        unknown = set(config) - allowed
        if unknown:
            raise KeyError(
                f"Unknown composition-site settings for {name!r}: {sorted(unknown)!r}."
            )
        if "column" not in config:
            raise ValueError(f"Composition site {name!r} requires a column.")
        if "elements" not in config or not config["elements"]:
            raise ValueError(
                f"Composition site {name!r} requires one or more candidate elements."
            )

        resolved = dict(_SITE_DEFAULTS)
        resolved.update(config)
        resolved["elements"] = tuple(resolved["elements"])
        resolved["descriptor_properties"] = tuple(resolved["descriptor_properties"])
        resolved["descriptor_statistics"] = tuple(resolved["descriptor_statistics"])
        resolved["element_properties"] = dict(resolved["element_properties"] or {})
        resolved["bounds"] = dict(resolved["bounds"] or {})
        resolved["steps"] = dict(resolved["steps"] or {})
        resolved["required_components"] = tuple(dict.fromkeys(resolved["required_components"] or ()))
        resolved["forbidden_components"] = tuple(dict.fromkeys(resolved["forbidden_components"] or ()))
        resolved["support_selection"] = str(resolved["support_selection"]).lower()
        if resolved["support_selection"] not in {"repair", "best_subset"}:
            raise ValueError(
                f"Composition site {name!r} support_selection must be 'repair' or 'best_subset'."
            )
        _normalize_best_subset_settings(name, resolved)
        known_elements = set(resolved["elements"])
        unknown_required = set(resolved["required_components"]) - known_elements
        if unknown_required:
            raise KeyError(
                f"Unknown required components at site {name!r}: {sorted(unknown_required)!r}."
            )
        unknown_forbidden = set(resolved["forbidden_components"]) - known_elements
        if unknown_forbidden:
            raise KeyError(
                f"Unknown forbidden components at site {name!r}: {sorted(unknown_forbidden)!r}."
            )
        overlap = set(resolved["required_components"]) & set(resolved["forbidden_components"])
        if overlap:
            raise ValueError(
                f"Composition site {name!r} cannot require and forbid the same components: {sorted(overlap)!r}."
            )
        for component in resolved["forbidden_components"]:
            pair = tuple(resolved["bounds"].get(component, (0.0, 0.0)))
            if len(pair) != 2:
                raise ValueError(f"Bounds for {component!r} must have length 2.")
            if float(pair[0]) > 1e-12:
                raise ValueError(
                    f"Forbidden component {component!r} at site {name!r} cannot have a positive lower bound."
                )
            resolved["bounds"][component] = (0.0, 0.0)
        resolved["min_components"] = int(resolved["min_components"])
        if resolved["max_components"] is not None:
            resolved["max_components"] = int(resolved["max_components"])
        resolved["total"] = float(resolved["total"])
        resolved["precision"] = int(resolved["precision"])
        resolved["pseudocount"] = float(resolved["pseudocount"])
        resolved["include_descriptors"] = bool(resolved["include_descriptors"])
        resolved["prefix"] = resolved["prefix"] or name
        normalized[name] = resolved

    columns = [config["column"] for config in normalized.values()]
    if len(set(columns)) != len(columns):
        raise ValueError("Each composition site must use a unique formula column.")
    return normalized


__all__ = ["normalize_composition_sites"]
