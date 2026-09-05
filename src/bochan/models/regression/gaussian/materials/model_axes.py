"""Orthogonal model-axis contract for material Gaussian surrogates.

Phase 39 separates wide-output semantics, explicit task indices, and fidelity
coordinates instead of overloading the historical ``multitask`` terminology.
The public dispatcher preserves the existing lower-level factories and routes
only combinations that are implemented today.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from torch import Tensor

from .common import MaterialExplicitTaskSpec, get_material_family
from .explicit_task_factory import create_registered_material_explicit_task_surrogate
from .surrogate_factory import (
    MaterialGaussianKind,
    MaterialInputMode,
    MaterialOutputMode,
    create_material_surrogate,
    normalize_material_gaussian_kind,
    normalize_material_input_mode,
    normalize_material_output_mode,
)

MaterialTaskAxisMode = Literal["none", "explicit"]
MaterialFidelityMode = Literal["none", "continuous"]

SUPPORTED_MATERIAL_TASK_MODES: tuple[MaterialTaskAxisMode, ...] = ("none", "explicit")
SUPPORTED_MATERIAL_FIDELITY_MODES: tuple[MaterialFidelityMode, ...] = ("none", "continuous")

_TASK_ALIASES = {
    "none": "none",
    "single": "none",
    "no-task": "none",
    "no_task": "none",
    "explicit": "explicit",
    "task": "explicit",
    "task-index": "explicit",
    "task_index": "explicit",
    "indexed": "explicit",
}
_FIDELITY_ALIASES = {
    "none": "none",
    "single": "none",
    "single-fidelity": "none",
    "single_fidelity": "none",
    "continuous": "continuous",
    "multi-fidelity": "continuous",
    "multi_fidelity": "continuous",
    "multifidelity": "continuous",
    "fidelity": "continuous",
}


def _normalize_axis(value: str, aliases: dict[str, str], name: str, supported: tuple[str, ...]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    normalized = aliases.get(value.strip().lower())
    if normalized is None:
        raise ValueError(
            f"Unsupported {name} {value!r}. Supported values: {', '.join(supported)}."
        )
    return normalized


def normalize_material_task_mode(mode: str) -> MaterialTaskAxisMode:
    """Normalize whether observations use an explicit discrete task index."""

    return cast(
        MaterialTaskAxisMode,
        _normalize_axis(mode, _TASK_ALIASES, "material task mode", SUPPORTED_MATERIAL_TASK_MODES),
    )


def normalize_material_fidelity_mode(mode: str) -> MaterialFidelityMode:
    """Normalize the fidelity axis without treating it as a task alias."""

    return cast(
        MaterialFidelityMode,
        _normalize_axis(
            mode,
            _FIDELITY_ALIASES,
            "material fidelity mode",
            SUPPORTED_MATERIAL_FIDELITY_MODES,
        ),
    )


@dataclass(frozen=True, slots=True)
class MaterialModelAxesSpec:
    """Serializable identity of orthogonal material-surrogate modeling axes.

    ``output_mode`` describes the shape/correlation of wide targets.
    ``task_mode`` describes a discrete task-id coordinate in long-format data.
    ``fidelity_mode`` describes an ordered accuracy/cost coordinate. Fidelity is
    intentionally distinct from task and is not yet dispatched by Phase 39.
    """

    family: str
    kind: MaterialGaussianKind | str = "gp"
    input_mode: MaterialInputMode | str = "continuous"
    output_mode: MaterialOutputMode | str = "scalar"
    task_mode: MaterialTaskAxisMode | str = "none"
    fidelity_mode: MaterialFidelityMode | str = "none"

    def __post_init__(self) -> None:
        registration = get_material_family(self.family)
        object.__setattr__(self, "family", registration.family)
        object.__setattr__(self, "kind", normalize_material_gaussian_kind(cast(str, self.kind)))
        object.__setattr__(self, "input_mode", normalize_material_input_mode(cast(str, self.input_mode)))
        object.__setattr__(self, "output_mode", normalize_material_output_mode(cast(str, self.output_mode)))
        object.__setattr__(self, "task_mode", normalize_material_task_mode(cast(str, self.task_mode)))
        object.__setattr__(
            self,
            "fidelity_mode",
            normalize_material_fidelity_mode(cast(str, self.fidelity_mode)),
        )
        if self.task_mode == "explicit" and self.output_mode != "scalar":
            raise ValueError(
                "Explicit task-index models use scalar long-format observations; "
                "set output_mode='scalar'. Wide independent/correlated outputs are a separate axis."
            )

    @property
    def domain(self) -> str:
        return get_material_family(self.family).domain

    @property
    def route(self) -> str:
        if self.fidelity_mode != "none":
            return "fidelity"
        if self.task_mode == "explicit":
            return "explicit_task"
        return "wide_output"

    @property
    def implemented(self) -> bool:
        """Whether Phase 39 can dispatch this exact axis combination."""

        return self.fidelity_mode == "none"

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "domain": self.domain,
            "kind": cast(str, self.kind),
            "input_mode": cast(str, self.input_mode),
            "output_mode": cast(str, self.output_mode),
            "task_mode": cast(str, self.task_mode),
            "fidelity_mode": cast(str, self.fidelity_mode),
            "route": self.route,
            "implemented": self.implemented,
        }


def create_material_model_from_axes(
    family: str,
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor | None = None,
    /,
    *,
    kind: str = "gp",
    input_mode: str = "continuous",
    output_mode: str = "scalar",
    task_mode: str = "none",
    fidelity_mode: str = "none",
    task_spec: MaterialExplicitTaskSpec | None = None,
    **kwargs: Any,
) -> Any:
    """Dispatch a material surrogate from orthogonal modeling axes.

    Existing factories remain the implementation backends. ``task_mode='none'``
    routes to the wide-output factory. ``task_mode='explicit'`` routes to the
    explicit-task factory. Continuous/multi-fidelity is intentionally rejected
    until a dedicated fidelity model is wired, preventing silent use of a task
    kernel for an ordered fidelity coordinate.
    """

    spec = MaterialModelAxesSpec(
        family=family,
        kind=kind,
        input_mode=input_mode,
        output_mode=output_mode,
        task_mode=task_mode,
        fidelity_mode=fidelity_mode,
    )
    if spec.fidelity_mode != "none":
        raise NotImplementedError(
            "Continuous fidelity is a separate ordered accuracy/cost axis and is not yet "
            "connected by the material model-axis dispatcher. Do not encode fidelity as "
            "an explicit task id; use fidelity_mode='none' until the dedicated fidelity "
            "backend is added."
        )

    if spec.task_mode == "explicit":
        return create_registered_material_explicit_task_surrogate(
            spec.family,
            train_X,
            train_Y,
            train_Yvar,
            kind=cast(str, spec.kind),
            input_mode=cast(str, spec.input_mode),
            task_spec=task_spec,
            **kwargs,
        )

    if task_spec is not None:
        raise ValueError("task_spec is only valid when task_mode='explicit'.")
    return create_material_surrogate(
        spec.family,
        train_X,
        train_Y,
        train_Yvar,
        kind=cast(str, spec.kind),
        input_mode=cast(str, spec.input_mode),
        output_mode=cast(str, spec.output_mode),
        **kwargs,
    )


def material_model_axes_capabilities(family: str) -> dict[str, Any]:
    """Describe implemented routes and reserved fidelity semantics."""

    registration = get_material_family(family)
    return {
        "family": registration.family,
        "domain": registration.domain,
        "axes": {
            "input_mode": ["continuous", "mixed"],
            "output_mode": ["scalar", "independent", "correlated"],
            "task_mode": ["none", "explicit"],
            "fidelity_mode": ["none", "continuous"],
        },
        "implemented_routes": ["wide_output", "explicit_task"],
        "fidelity_route_implemented": False,
        "notes": {
            "output": "wide target shape/correlation",
            "task": "discrete long-format task index",
            "fidelity": "ordered accuracy/cost coordinate; not a task alias",
        },
    }


__all__ = [
    "MaterialFidelityMode",
    "MaterialModelAxesSpec",
    "MaterialTaskAxisMode",
    "SUPPORTED_MATERIAL_FIDELITY_MODES",
    "SUPPORTED_MATERIAL_TASK_MODES",
    "create_material_model_from_axes",
    "material_model_axes_capabilities",
    "normalize_material_fidelity_mode",
    "normalize_material_task_mode",
]
