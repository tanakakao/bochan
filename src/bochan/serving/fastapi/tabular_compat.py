"""FastAPI compatibility helpers for tabular category metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from . import converters as _converters

_CATEGORY_MAPS_ATTR = "_bochan_fastapi_target_category_maps"
_TARGET_NAMES_ATTR = "_bochan_fastapi_target_names"


def _dump(value: Any) -> dict[str, Any]:
    """Return a mutable request payload without losing mapping values."""
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=False)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(
        f"Expected a Pydantic model or mapping. Got {type(value).__name__}."
    )


def _clean_output_category_nones(value: Any) -> Any:
    """Remove Pydantic ``None`` defaults before tabular metadata extraction."""
    if not isinstance(value, Mapping):
        return value
    cleaned = dict(value)
    for key in ("ordered_categories", "categories", "category_map"):
        if cleaned.get(key) is None:
            cleaned.pop(key, None)
    return cleaned


def _extract_category_metadata(
    model_config_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[Any, dict[Any, int]], list[Any]]:
    """Strip tabular category fields and return their resolved mappings."""
    resolved = dict(model_config_payload)
    multi_output = resolved.get("multi_output_config")
    if multi_output is None:
        return resolved, {}, []

    multi_output_payload = _dump(multi_output)
    output_configs = multi_output_payload.get("output_configs")
    if output_configs is not None:
        multi_output_payload["output_configs"] = [
            _clean_output_category_nones(item) for item in output_configs
        ]

    # Reuse the tabular implementation so FastAPI and notebook validation stay
    # identical for ordered_categories / categories / category_map.
    from bochan.tabular.multi_output_categories import _extract_output_category_maps

    cleaned_multi_output, category_maps = _extract_output_category_maps(
        multi_output_payload
    )
    resolved["multi_output_config"] = cleaned_multi_output

    target_names: list[Any] = []
    for item in cleaned_multi_output.get("output_configs") or []:
        if isinstance(item, Mapping) and item.get("name") is not None:
            target_names.append(item["name"])
        else:
            target_names.append(None)

    return resolved, category_maps, target_names


def _set_category_metadata(
    value: Any,
    *,
    category_maps: Mapping[Any, Mapping[Any, int]],
    target_names: list[Any],
) -> None:
    """Attach JSON-only category metadata to a core config or optimizer."""
    maps = {
        output: {label: int(index) for label, index in mapping.items()}
        for output, mapping in category_maps.items()
    }
    setattr(value, _CATEGORY_MAPS_ATTR, maps)
    setattr(value, _TARGET_NAMES_ATTR, list(target_names))


def to_model_config(value: Any, options: Any | None = None) -> Any:
    """Convert a FastAPI model payload while retaining tabular label metadata."""
    payload = _dump(value)
    payload, category_maps, target_names = _extract_category_metadata(payload)
    config = _converters.to_model_config(payload, options)

    _set_category_metadata(
        config,
        category_maps=category_maps,
        target_names=target_names,
    )
    multi_output_config = getattr(config, "multi_output_config", None)
    if multi_output_config is not None:
        _set_category_metadata(
            multi_output_config,
            category_maps=category_maps,
            target_names=target_names,
        )
    return config


def bind_category_metadata(optimizer: Any, model_config: Any) -> None:
    """Bind retained category metadata to a fitted FastAPI optimizer."""
    category_maps = getattr(model_config, _CATEGORY_MAPS_ATTR, {})
    target_names = getattr(model_config, _TARGET_NAMES_ATTR, [])
    _set_category_metadata(
        optimizer,
        category_maps=category_maps,
        target_names=list(target_names),
    )


def _category_context(optimizer: Any) -> tuple[list[Any], dict[Any, dict[Any, int]]]:
    """Return output names and category maps retained by FastAPI model fitting."""
    category_maps = dict(getattr(optimizer, _CATEGORY_MAPS_ATTR, {}) or {})
    target_names = list(getattr(optimizer, _TARGET_NAMES_ATTR, []) or [])

    model_config = getattr(optimizer, "model_config", None)
    if not category_maps and model_config is not None:
        category_maps = dict(
            getattr(model_config, _CATEGORY_MAPS_ATTR, {}) or {}
        )
        multi_output_config = getattr(model_config, "multi_output_config", None)
        if not category_maps and multi_output_config is not None:
            category_maps = dict(
                getattr(multi_output_config, _CATEGORY_MAPS_ATTR, {}) or {}
            )

    model = getattr(optimizer, "model", None)
    model_names = getattr(model, "output_names", None)
    if model_names is not None:
        target_names = list(model_names)

    if not target_names and category_maps:
        target_names = list(category_maps)
    return target_names, category_maps


def _category_context_from_model_config(
    model_config: Any,
) -> tuple[list[Any], dict[Any, dict[Any, int]]]:
    """Return retained output names and category maps from a model config."""
    category_maps = dict(
        getattr(model_config, _CATEGORY_MAPS_ATTR, {}) or {}
    )
    target_names = list(
        getattr(model_config, _TARGET_NAMES_ATTR, []) or []
    )
    multi_output_config = getattr(model_config, "multi_output_config", None)
    if not category_maps and multi_output_config is not None:
        category_maps = dict(
            getattr(multi_output_config, _CATEGORY_MAPS_ATTR, {}) or {}
        )
    if not target_names and category_maps:
        target_names = list(category_maps)
    return target_names, category_maps


def _resolve_category_label(
    value: Any,
    mapping: Mapping[Any, int],
    *,
    output: Any,
) -> Any:
    """Resolve one original label while preserving already encoded indices."""
    try:
        if value in mapping:
            return int(mapping[value])
    except TypeError:
        pass

    for label, index in mapping.items():
        if str(label) == str(value):
            return int(index)

    if isinstance(value, int) and value in set(mapping.values()):
        return value
    if isinstance(value, float) and value.is_integer():
        encoded = int(value)
        if encoded in set(mapping.values()):
            return encoded

    raise KeyError(
        f"Unknown target label {value!r} for output {output!r}. "
        f"Available labels: {list(mapping)!r}."
    )


def to_target_tensor(
    value: Any,
    options: Any | None = None,
    *,
    model_config: Any | None = None,
    optimizer: Any | None = None,
) -> Any:
    """Encode FastAPI string target labels and convert them to a tensor."""
    if optimizer is not None:
        target_names, category_maps = _category_context(optimizer)
    elif model_config is not None:
        target_names, category_maps = _category_context_from_model_config(
            model_config
        )
    else:
        target_names, category_maps = [], {}

    if not category_maps:
        return _converters.to_tensor(value, options)

    if hasattr(value, "detach"):
        return _converters.to_tensor(value, options)
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        value = value.tolist()

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("train_Y / new_Y must be a sequence of target rows.")

    rows = list(value)
    single_output = len(target_names) == 1
    is_matrix = bool(
        rows
        and not isinstance(rows[0], (str, bytes))
        and isinstance(rows[0], Sequence)
    )
    if single_output and not is_matrix:
        encoded = [
            _resolve_category_label(
                item,
                category_maps.get(target_names[0], {}),
                output=target_names[0],
            )
            for item in rows
        ]
        return _converters.to_tensor(encoded, options)

    if not is_matrix:
        raise ValueError(
            "Multi-output train_Y / new_Y must be a two-dimensional sequence."
        )

    encoded_rows: list[list[Any]] = []
    for row_index, row in enumerate(rows):
        row_values = list(row)
        if len(row_values) != len(target_names):
            raise ValueError(
                f"Target row {row_index} has {len(row_values)} values, "
                f"but {len(target_names)} outputs are configured."
            )
        for output_index, output_name in enumerate(target_names):
            mapping = category_maps.get(output_name)
            if mapping is None:
                mapping = category_maps.get(str(output_name))
            if mapping is not None:
                row_values[output_index] = _resolve_category_label(
                    row_values[output_index],
                    mapping,
                    output=output_name,
                )
        encoded_rows.append(row_values)
    return _converters.to_tensor(encoded_rows, options)


def to_acquisition_config(
    value: Any,
    options: Any | None = None,
    *,
    optimizer: Any | None = None,
) -> Any:
    """Convert acquisition settings and resolve original target labels."""
    payload = _dump(value)
    if optimizer is not None:
        target_names, category_maps = _category_context(optimizer)
        if target_names and category_maps:
            # Importing bochan.tabular applies the string target-class and
            # ordinal-rank label resolvers used by TabularBayesianOptimizer.
            import bochan.tabular  # noqa: F401
            from bochan.tabular.optimizer_api import (
                _resolve_acquisition_config_columns,
            )

            payload = _resolve_acquisition_config_columns(
                payload,
                target_names,
                category_maps,
            )
    return _converters.to_acquisition_config(payload, options)


__all__ = [
    "bind_category_metadata",
    "to_acquisition_config",
    "to_model_config",
    "to_target_tensor",
]
