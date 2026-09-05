"""High-level API adapter for material model-axis surrogates.

Phase 40 connects the Phase 39 material axis contract to the generic bochan
model/acquisition/candidate pipeline without expanding the static model registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bochan.models.regression.gaussian.materials import (
    MaterialExplicitTaskSpec,
    MaterialModelAxesSpec,
    create_material_model_from_axes,
)

from ..configs import ModelConfig


@dataclass(frozen=True, slots=True)
class MaterialAPIModelSpec:
    """API-facing specification for one material surrogate configuration."""

    family: str
    kind: str = "gp"
    input_mode: str = "continuous"
    output_mode: str = "scalar"
    task_mode: str = "none"
    fidelity_mode: str = "none"
    task_feature: int = -1
    all_tasks: tuple[int, ...] | None = None
    output_tasks: tuple[int, ...] | None = None
    backend_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def axes(self) -> MaterialModelAxesSpec:
        return MaterialModelAxesSpec(
            family=self.family,
            kind=self.kind,
            input_mode=self.input_mode,
            output_mode=self.output_mode,
            task_mode=self.task_mode,
            fidelity_mode=self.fidelity_mode,
        )

    @property
    def task_spec(self) -> MaterialExplicitTaskSpec | None:
        if self.axes.task_mode != "explicit":
            return None
        return MaterialExplicitTaskSpec(
            task_feature=self.task_feature,
            all_tasks=self.all_tasks,
            output_tasks=self.output_tasks,
        )

    def as_dict(self) -> dict[str, Any]:
        payload = self.axes.as_dict()
        payload.update(
            {
                "task_feature": self.task_feature if self.axes.task_mode == "explicit" else None,
                "all_tasks": list(self.all_tasks) if self.all_tasks is not None else None,
                "output_tasks": list(self.output_tasks) if self.output_tasks is not None else None,
            }
        )
        return payload


def _material_model_factory(spec: MaterialAPIModelSpec):
    axes = spec.axes
    task_spec = spec.task_spec

    def factory(
        train_X: Any,
        train_Y: Any,
        train_Yvar: Any | None = None,
        **runtime_kwargs: Any,
    ) -> Any:
        kwargs = dict(spec.backend_kwargs)
        kwargs.update(runtime_kwargs)
        model = create_material_model_from_axes(
            axes.family,
            train_X,
            train_Y,
            train_Yvar,
            kind=str(axes.kind),
            input_mode=str(axes.input_mode),
            output_mode=str(axes.output_mode),
            task_mode=str(axes.task_mode),
            fidelity_mode=str(axes.fidelity_mode),
            task_spec=task_spec,
            **kwargs,
        )
        model.material_model_axes = spec.as_dict()
        return model

    factory.__name__ = f"material_{axes.family}_{axes.route}_factory"
    return factory


def make_material_model_config(
    spec: MaterialAPIModelSpec,
    *,
    cat_dims: list[int] | tuple[int, ...] | None = None,
    outcome_transform: bool | Any = True,
) -> ModelConfig:
    """Create a ``ModelConfig`` that routes through the material-axis factory.

    The returned config plugs directly into the existing high-level model build,
    fit, acquisition, and candidate pipeline. Existing registry model types remain
    unchanged and backward compatible.
    """

    if not isinstance(spec, MaterialAPIModelSpec):
        raise TypeError("spec must be a MaterialAPIModelSpec.")
    axes = spec.axes
    if axes.fidelity_mode != "none":
        raise NotImplementedError(
            "Material fidelity remains reserved but unimplemented. "
            "Use fidelity_mode='none' for the high-level API route."
        )

    normalized_cat_dims = None if cat_dims is None else list(cat_dims)
    if axes.input_mode == "mixed" and not normalized_cat_dims:
        raise ValueError("cat_dims is required when input_mode='mixed'.")
    if axes.input_mode == "continuous" and normalized_cat_dims:
        raise ValueError("cat_dims must be omitted when input_mode='continuous'.")

    return ModelConfig(
        model_factory=_material_model_factory(spec),
        task_type="regression",
        model_type=f"material:{axes.family}:{axes.route}",
        input_type="mixed" if axes.input_mode == "mixed" else "normal",
        cat_dims=normalized_cat_dims,
        outcome_transform=outcome_transform,
        pass_cat_dims=axes.input_mode == "mixed",
    )


def material_task_fixed_features(
    spec: MaterialAPIModelSpec,
    target_task: int,
    *,
    input_dim: int | None = None,
) -> dict[int, float]:
    """Return fixed-feature mapping for optimizing one explicit material task.

    Explicit-task models represent task id as an input coordinate. BO over a
    specific task must fix that coordinate so the optimizer does not search over
    task identity itself.
    """

    axes = spec.axes
    if axes.task_mode != "explicit":
        raise ValueError("target_task is only valid for task_mode='explicit'.")
    if isinstance(target_task, bool) or not isinstance(target_task, int):
        raise TypeError("target_task must be an integer task id.")
    if spec.all_tasks is not None and target_task not in spec.all_tasks:
        raise ValueError(f"target_task={target_task} is not included in all_tasks={spec.all_tasks!r}.")

    task_feature = int(spec.task_feature)
    if task_feature < 0:
        if input_dim is None:
            raise ValueError(
                "input_dim is required when task_feature is negative so the API can "
                "resolve the raw input coordinate."
            )
        task_feature += int(input_dim)
    if task_feature < 0 or (input_dim is not None and task_feature >= int(input_dim)):
        raise ValueError("task_feature is out of bounds for the provided input_dim.")
    return {task_feature: float(target_task)}


__all__ = [
    "MaterialAPIModelSpec",
    "make_material_model_config",
    "material_task_fixed_features",
]
