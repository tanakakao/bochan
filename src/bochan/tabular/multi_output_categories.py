'''Resolve tabular target category metadata declared in output configs.'''

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

_APPLIED = False
_CATEGORY_KEYS = ("ordered_categories", "categories", "category_map")


def _category_map_from_output_config(
    value: Any,
    *,
    key: str,
    output_name: Any,
) -> dict[Any, int]:
    '''Normalize one output-level category declaration to a label-index map.'''

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


def _extract_output_category_maps(
    multi_output_config: Any,
) -> tuple[Any, dict[Any, dict[Any, int]]]:
    '''Remove tabular category metadata from output configs and return mappings.'''

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

        category_map = _category_map_from_output_config(
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


def _merge_target_category_metadata(
    kwargs: dict[str, Any],
    inferred_maps: Mapping[Any, Mapping[Any, int]],
) -> None:
    '''Merge inferred maps with explicit tabular data configuration.'''

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


def apply_tabular_multi_output_categories() -> None:
    '''Allow output configs to carry tabular category ordering and mappings.'''

    global _APPLIED
    if _APPLIED:
        return

    from . import optimizer_api

    original_init = optimizer_api.TabularBayesianOptimizer.__init__
    if getattr(original_init, "_bochan_supports_output_categories", False):
        _APPLIED = True
        return

    @wraps(original_init)
    def init_with_output_categories(
        self,
        model_config=None,
        fit_config=None,
        **kwargs: Any,
    ) -> None:
        inferred_maps: dict[Any, dict[Any, int]] = {}

        if isinstance(model_config, Mapping):
            resolved_model_config = dict(model_config)
            multi_output_config = resolved_model_config.get("multi_output_config")
            if multi_output_config is not None:
                resolved_multi_output, maps = _extract_output_category_maps(
                    multi_output_config
                )
                resolved_model_config["multi_output_config"] = resolved_multi_output
                inferred_maps.update(maps)
            model_config = resolved_model_config

        direct_multi_output = kwargs.get("multi_output_config")
        if direct_multi_output is not None:
            resolved_multi_output, maps = _extract_output_category_maps(
                direct_multi_output
            )
            kwargs["multi_output_config"] = resolved_multi_output
            for output_name, category_map in maps.items():
                existing = inferred_maps.get(output_name)
                if existing is not None and existing != category_map:
                    raise ValueError(
                        f"Conflicting category declarations for output "
                        f"{output_name!r}."
                    )
                inferred_maps[output_name] = category_map

        _merge_target_category_metadata(kwargs, inferred_maps)
        original_init(
            self,
            model_config=model_config,
            fit_config=fit_config,
            **kwargs,
        )

    setattr(init_with_output_categories, "_bochan_supports_output_categories", True)
    optimizer_api.TabularBayesianOptimizer.__init__ = init_with_output_categories
    _APPLIED = True


__all__ = ["apply_tabular_multi_output_categories"]
