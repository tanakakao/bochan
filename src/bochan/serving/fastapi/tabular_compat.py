"""FastAPI helpers for tabular category metadata and label conversion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import converters as _converters

_CATEGORY_METADATA_KEY = "target_category_metadata"


@dataclass(frozen=True)
class TargetCategoryMetadata:
    """Transport metadata required to encode/decode categorical target labels."""

    target_names: tuple[Any, ...] = ()
    category_maps: Mapping[Any, Mapping[Any, int]] | None = None

    def normalized_maps(self) -> dict[Any, dict[Any, int]]:
        """Return mutable integer-valued category maps."""

        return {
            output: {label: int(index) for label, index in mapping.items()}
            for output, mapping in dict(self.category_maps or {}).items()
        }

    @property
    def empty(self) -> bool:
        return not bool(self.category_maps)


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
) -> tuple[dict[str, Any], TargetCategoryMetadata]:
    """Strip tabular category fields and return an explicit metadata object."""

    resolved = dict(model_config_payload)
    multi_output = resolved.get("multi_output_config")
    if multi_output is None:
        return resolved, TargetCategoryMetadata()

    multi_output_payload = _dump(multi_output)
    output_configs = multi_output_payload.get("output_configs")
    if output_configs is not None:
        multi_output_payload["output_configs"] = [
            _clean_output_category_nones(item) for item in output_configs
        ]

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

    return resolved, TargetCategoryMetadata(
        target_names=tuple(target_names),
        category_maps=category_maps,
    )


def to_model_config_with_metadata(
    value: Any,
    options: Any | None = None,
) -> tuple[Any, TargetCategoryMetadata]:
    """Convert a model payload and return its transport-only target metadata."""

    payload = _dump(value)
    payload, metadata = _extract_category_metadata(payload)
    return _converters.to_model_config(payload, options), metadata


def to_model_config(value: Any, options: Any | None = None) -> Any:
    """Convert a FastAPI model payload to the tensor-oriented core config."""

    config, _ = to_model_config_with_metadata(value, options)
    return config


def bind_category_metadata(
    optimizer: Any,
    metadata: TargetCategoryMetadata,
) -> None:
    """Bind explicit target metadata to a fitted optimizer and its bundle."""

    optimizer.target_category_metadata = metadata
    bundle = getattr(optimizer, "bundle", None)
    bundle_metadata = getattr(bundle, "metadata", None)
    if isinstance(bundle_metadata, dict):
        bundle_metadata[_CATEGORY_METADATA_KEY] = metadata


def _category_metadata_from_optimizer(optimizer: Any) -> TargetCategoryMetadata:
    metadata = getattr(optimizer, "target_category_metadata", None)
    if isinstance(metadata, TargetCategoryMetadata):
        return metadata

    bundle = getattr(optimizer, "bundle", None)
    bundle_metadata = getattr(bundle, "metadata", None)
    if isinstance(bundle_metadata, Mapping):
        metadata = bundle_metadata.get(_CATEGORY_METADATA_KEY)
        if isinstance(metadata, TargetCategoryMetadata):
            return metadata
        if isinstance(metadata, Mapping):
            return TargetCategoryMetadata(
                target_names=tuple(metadata.get("target_names", ())),
                category_maps=metadata.get("category_maps", {}),
            )
    return TargetCategoryMetadata()


def _category_context(
    optimizer: Any,
) -> tuple[list[Any], dict[Any, dict[Any, int]]]:
    """Return output names and category maps retained by FastAPI model fitting."""

    metadata = _category_metadata_from_optimizer(optimizer)
    category_maps = metadata.normalized_maps()
    target_names = list(metadata.target_names)

    model = getattr(optimizer, "model", None)
    model_names = getattr(model, "output_names", None)
    if model_names is not None:
        target_names = list(model_names)
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
    metadata: TargetCategoryMetadata | None = None,
    optimizer: Any | None = None,
) -> Any:
    """Encode FastAPI string target labels and convert them to a tensor."""

    if optimizer is not None:
        target_names, category_maps = _category_context(optimizer)
    elif metadata is not None:
        target_names = list(metadata.target_names)
        category_maps = metadata.normalized_maps()
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
            from bochan.tabular.optimizer_api import (
                _resolve_acquisition_config_columns,
            )
            from bochan.tabular.ordinal_rank_labels import (
                resolve_acquisition_ordinal_ranks,
            )

            payload = _resolve_acquisition_config_columns(
                payload,
                target_names,
                category_maps,
            )
            payload = resolve_acquisition_ordinal_ranks(
                payload,
                target_names=target_names,
                target_category_maps=category_maps,
            )
    return _converters.to_acquisition_config(payload, options)


__all__ = [
    "TargetCategoryMetadata",
    "bind_category_metadata",
    "to_acquisition_config",
    "to_model_config",
    "to_model_config_with_metadata",
    "to_target_tensor",
]
