"""Material-model execution helpers for FastAPI adapters."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from bochan.api.modeling.materials import (
    MaterialAPIModelSpec,
    make_material_model_config,
    material_task_fixed_features,
)


def to_material_api_spec(value: Any) -> MaterialAPIModelSpec:
    """Convert a material-axis request into the high-level API specification."""

    task = getattr(value, "task", None)
    return MaterialAPIModelSpec(
        family=value.family,
        kind=value.kind,
        input_mode=value.input_mode,
        output_mode=value.output_mode,
        task_mode=value.task_mode,
        fidelity_mode=value.fidelity_mode,
        task_feature=-1 if task is None else task.task_feature,
        all_tasks=None if task is None or task.all_tasks is None else tuple(task.all_tasks),
        output_tasks=None if task is None or task.output_tasks is None else tuple(task.output_tasks),
        backend_kwargs=dict(value.backend_kwargs),
    )


def to_material_model_config(value: Any) -> tuple[Any, MaterialAPIModelSpec]:
    """Build the canonical ModelConfig for one material-axis request."""

    spec = to_material_api_spec(value)
    config = make_material_model_config(spec, cat_dims=value.cat_dims)
    return config, spec


def bind_material_model_spec(optimizer: Any, spec: MaterialAPIModelSpec, *, input_dim: int) -> None:
    """Persist transport metadata required by later candidate requests."""

    optimizer.material_api_model_spec = spec
    optimizer.material_input_dim = int(input_dim)


def material_model_spec(optimizer: Any) -> MaterialAPIModelSpec | None:
    """Return material metadata attached at fit time, if present."""

    spec = getattr(optimizer, "material_api_model_spec", None)
    return spec if isinstance(spec, MaterialAPIModelSpec) else None


def _merge_fixed_features(
    existing: dict[int, float] | None,
    required: dict[int, float],
) -> dict[int, float]:
    merged = dict(existing or {})
    for index, value in required.items():
        if index in merged and float(merged[index]) != float(value):
            raise ValueError(
                f"fixed_features[{index}]={merged[index]} conflicts with required material task value {value}."
            )
        merged[index] = float(value)
    return merged


def apply_material_target_task(optimizer: Any, opt_config: Any, target_task: int | None) -> Any:
    """Fix the explicit task coordinate for material candidate optimization."""

    spec = material_model_spec(optimizer)
    if spec is None:
        if target_task is not None:
            raise ValueError("target_task is only valid for a material explicit-task model.")
        return opt_config

    axes = spec.axes
    if axes.task_mode != "explicit":
        if target_task is not None:
            raise ValueError("target_task is only valid when material task_mode='explicit'.")
        return opt_config
    if target_task is None:
        raise ValueError("target_task is required for material task_mode='explicit' candidate generation.")

    input_dim = getattr(optimizer, "material_input_dim", None)
    required = material_task_fixed_features(spec, target_task, input_dim=input_dim)
    updates: dict[str, Any] = {
        "fixed_features": _merge_fixed_features(getattr(opt_config, "fixed_features", None), required)
    }

    fixed_features_list = getattr(opt_config, "fixed_features_list", None)
    if fixed_features_list is not None:
        updates["fixed_features_list"] = [
            _merge_fixed_features(item, required) for item in fixed_features_list
        ]

    repair_config = getattr(opt_config, "repair_config", None)
    if repair_config is not None:
        updates["repair_config"] = replace(
            repair_config,
            fixed_features=_merge_fixed_features(getattr(repair_config, "fixed_features", None), required),
        )

    return replace(opt_config, **updates)


__all__ = [
    "apply_material_target_task",
    "bind_material_model_spec",
    "material_model_spec",
    "to_material_api_spec",
    "to_material_model_config",
]
