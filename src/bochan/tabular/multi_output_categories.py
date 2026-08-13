"""Target-category normalization and label resolution for tabular outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .builders import UNSET
from .converter import resolve_column_indices

_CATEGORY_KEYS = ("ordered_categories", "categories", "category_map")


def category_map_from_output_config(
    value: Any,
    *,
    key: str,
    output_name: Any,
) -> dict[Any, int]:
    """Normalize one output-level category declaration to a label-index map."""

    if key == "category_map":
        if not isinstance(value, Mapping):
            raise TypeError(
                f"category_map for output {output_name!r} must be a mapping."
            )
        mapping = dict(value)
    else:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(
                f"{key} for output {output_name!r} must be a sequence of labels."
            )
        labels = list(value)
        if not labels:
            raise ValueError(f"{key} for output {output_name!r} must not be empty.")
        try:
            mapping = {label: index for index, label in enumerate(labels)}
        except TypeError as exc:
            raise TypeError(
                f"{key} for output {output_name!r} must contain hashable labels."
            ) from exc
        if len(mapping) != len(labels):
            raise ValueError(
                f"{key} for output {output_name!r} contains duplicate labels."
            )

    if not mapping:
        raise ValueError(
            f"Category mapping for output {output_name!r} must not be empty."
        )

    indices: list[int] = []
    for label, index in mapping.items():
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(
                f"Category index for label {label!r} in output {output_name!r} "
                "must be an integer."
            )
        indices.append(int(index))

    if sorted(indices) != list(range(len(indices))):
        raise ValueError(
            f"Category indices for output {output_name!r} must be unique, "
            "consecutive integers starting at 0."
        )
    return {label: int(index) for label, index in mapping.items()}


def extract_output_category_maps(
    multi_output_config: Any,
) -> tuple[Any, dict[Any, dict[Any, int]]]:
    """Remove tabular category metadata from output configs and return mappings."""

    if not isinstance(multi_output_config, Mapping):
        return multi_output_config, {}

    resolved_multi_output = dict(multi_output_config)
    output_configs = resolved_multi_output.get("output_configs")
    if output_configs is None:
        return resolved_multi_output, {}

    resolved_outputs: list[Any] = []
    inferred_maps: dict[Any, dict[Any, int]] = {}

    for raw in output_configs:
        if not isinstance(raw, Mapping):
            resolved_outputs.append(raw)
            continue

        output_config = dict(raw)
        present_keys = [
            key
            for key in _CATEGORY_KEYS
            if key in output_config and output_config[key] is not None
        ]
        if len(present_keys) > 1:
            raise ValueError(
                "Specify only one of ordered_categories, categories, or "
                f"category_map for output {output_config.get('name')!r}."
            )
        if not present_keys:
            resolved_outputs.append(output_config)
            continue

        output_name = output_config.get("name")
        if output_name is None:
            raise ValueError(
                f"{present_keys[0]} requires a named output config."
            )

        key = present_keys[0]
        task_type = str(output_config.get("task_type", "")).lower()
        if key == "ordered_categories" and task_type not in {"", "ordinal"}:
            raise ValueError(
                "ordered_categories is only valid for ordinal outputs. "
                f"Got task_type={task_type!r} for output {output_name!r}."
            )

        category_map = category_map_from_output_config(
            output_config[key],
            key=key,
            output_name=output_name,
        )
        existing = inferred_maps.get(output_name)
        if existing is not None and existing != category_map:
            raise ValueError(
                f"Conflicting category declarations for output {output_name!r}."
            )
        inferred_maps[output_name] = category_map

        for category_key in _CATEGORY_KEYS:
            output_config.pop(category_key, None)
        resolved_outputs.append(output_config)

    resolved_multi_output["output_configs"] = resolved_outputs
    return resolved_multi_output, inferred_maps


def merge_target_category_metadata(
    kwargs: dict[str, Any],
    inferred_maps: Mapping[Any, Mapping[Any, int]],
) -> None:
    """Merge inferred maps with explicit tabular data configuration."""

    if not inferred_maps:
        return

    data_config = kwargs.get("data_config")
    explicit_maps = kwargs.get("target_category_maps")
    if explicit_maps is None and data_config is not None:
        explicit_maps = getattr(data_config, "target_category_maps", None)
    merged_maps = dict(explicit_maps or {})

    for output_name, inferred_map in inferred_maps.items():
        existing = merged_maps.get(output_name)
        if existing is None:
            existing = merged_maps.get(str(output_name))
        if existing is not None and dict(existing) != dict(inferred_map):
            raise ValueError(
                f"Category mapping for output {output_name!r} conflicts with "
                "target_category_maps."
            )
        merged_maps[output_name] = dict(inferred_map)

    explicit_cols = kwargs.get("target_categorical_cols")
    if explicit_cols is None and data_config is not None:
        explicit_cols = getattr(data_config, "target_categorical_cols", None)
    if explicit_cols is None:
        merged_cols: list[Any] = []
    elif isinstance(explicit_cols, (str, bytes)):
        merged_cols = [explicit_cols]
    else:
        merged_cols = list(explicit_cols)

    for output_name in inferred_maps:
        if output_name not in merged_cols:
            merged_cols.append(output_name)

    kwargs["target_categorical_cols"] = merged_cols
    kwargs["target_category_maps"] = merged_maps


def target_category_map_for_output(
    output: Any,
    target_names: Sequence[Any],
    target_category_maps: Mapping[Any, Mapping[Any, int]] | None,
) -> Mapping[Any, int] | None:
    """Return the original-label to class-index map for one target output."""

    if not target_category_maps:
        return None
    resolved = resolve_column_indices([output], list(target_names))
    if not resolved:
        return None
    target_name = target_names[resolved[0]]
    mapping = target_category_maps.get(target_name)
    if mapping is None:
        mapping = target_category_maps.get(str(target_name))
    return mapping


def resolve_target_class_value(
    value: Any,
    *,
    output: Any,
    target_names: Sequence[Any],
    target_category_maps: Mapping[Any, Mapping[Any, int]] | None,
) -> Any:
    """Resolve a categorical target label to its encoded class index."""

    if not isinstance(value, str):
        return value

    mapping = target_category_map_for_output(
        output,
        target_names,
        target_category_maps,
    )
    if mapping is None:
        raise ValueError(
            f"String target class {value!r} for output {output!r} cannot be "
            "resolved because no target category map is available."
        )
    if value in mapping:
        return int(mapping[value])
    for label, class_index in mapping.items():
        if str(label) == value:
            return int(class_index)
    raise KeyError(
        f"Unknown target class label {value!r} for output {output!r}. "
        f"Available labels: {list(mapping)!r}."
    )


def resolve_constraint_target_classes(
    value: Any,
    *,
    target_names: Sequence[Any],
    target_category_maps: Mapping[Any, Mapping[Any, int]] | None,
) -> Any:
    """Resolve string target-class fields in one constraint mapping."""

    if not isinstance(value, Mapping):
        return value
    if "target_class" not in value and "target_classes" not in value:
        return value

    resolved = dict(value)
    output = resolved.get("output")
    if output is None:
        return resolved

    target_class = resolved.get("target_class")
    if target_class is not None:
        resolved["target_class"] = resolve_target_class_value(
            target_class,
            output=output,
            target_names=target_names,
            target_category_maps=target_category_maps,
        )

    target_classes = resolved.get("target_classes")
    if target_classes is not None:
        values = (
            [target_classes]
            if isinstance(target_classes, str)
            else list(target_classes)
        )
        resolved["target_classes"] = [
            resolve_target_class_value(
                item,
                output=output,
                target_names=target_names,
                target_category_maps=target_category_maps,
            )
            for item in values
        ]
    return resolved


def resolve_outcome_constraint_config_columns(
    value: Any,
    target_names: Sequence[Any],
    target_category_maps: Mapping[Any, Mapping[Any, int]] | None = None,
) -> Any:
    """Resolve tabular target names and class labels in an outcome config."""

    if value is UNSET or value is None or not isinstance(value, Mapping):
        return value

    resolved = dict(value)
    if "output_indices" in resolved:
        output_indices = resolve_column_indices(
            resolved["output_indices"],
            list(target_names),
        )
        resolved["output_indices"] = output_indices or []

    constraints = resolved.get("constraints")
    if constraints is not None:
        constraint_values = (
            [constraints] if isinstance(constraints, Mapping) else list(constraints)
        )
        resolved["constraints"] = [
            resolve_constraint_target_classes(
                item,
                target_names=target_names,
                target_category_maps=target_category_maps,
            )
            for item in constraint_values
        ]
    return resolved


def resolve_acquisition_config_columns(
    acq_config: Any,
    target_names: Sequence[Any],
    target_category_maps: Mapping[Any, Mapping[Any, int]] | None = None,
) -> Any:
    """Resolve named outcomes and target labels nested in an acquisition config."""

    if not isinstance(acq_config, Mapping):
        return acq_config
    if "outcome_constraint_config" not in acq_config:
        return acq_config

    resolved = dict(acq_config)
    resolved["outcome_constraint_config"] = resolve_outcome_constraint_config_columns(
        resolved["outcome_constraint_config"],
        target_names,
        target_category_maps,
    )
    return resolved


__all__ = [
    "category_map_from_output_config",
    "extract_output_category_maps",
    "merge_target_category_metadata",
    "resolve_acquisition_config_columns",
    "resolve_constraint_target_classes",
    "resolve_outcome_constraint_config_columns",
    "resolve_target_class_value",
    "target_category_map_for_output",
]
