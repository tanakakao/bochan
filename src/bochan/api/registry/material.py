"""Public model-type registrations for pretrained material residual GPs.

This module keeps material-specific model names out of the generic registry
literal while preserving the same lazy import contract.  Registration mutates
only the registry tree returned by :class:`LazyModelRegistry`; concrete material
backends are imported later by ``resolve_model_cls``.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from .model import DEFAULT_MODEL_REGISTRY

_MATERIAL_NAMESPACE = "bochan.models.regression.gaussian.materials.structure"
_MATERIAL_FAMILIES = ("chgnet", "m3gnet", "mace")

# Public model_type -> canonical class name.  The input-type split intentionally
# mirrors the generic API registry: categorical process variables select the
# ``mixed`` branch; the structure selector itself is not a cat_dim.
_NORMAL_VARIANTS: dict[str, str] = {
    "residual_gp": "ResidualGPModel",
    "multitask_residual_gp": "MultiTaskResidualGPModel",
}
_MIXED_VARIANTS: dict[str, str] = {
    "mixed_residual_gp": "MixedResidualGPModel",
    "mixed_multitask_residual_gp": "MixedMultiTaskResidualGPModel",
}

_PREFIXES = {
    "chgnet": "CHGNet",
    "m3gnet": "M3GNet",
    "mace": "MACE",
}


def material_residual_model_types(*, input_type: str | None = None) -> tuple[str, ...]:
    """Return stable public ``model_type`` names for material residual GPs.

    Args:
        input_type: Optional ``"normal"`` or ``"mixed"`` filter.

    Returns:
        Deterministically ordered public model type names.
    """

    if input_type not in {None, "normal", "mixed"}:
        raise ValueError("input_type must be 'normal', 'mixed', or None.")
    variants: dict[str, str] = {}
    if input_type in {None, "normal"}:
        variants.update(_NORMAL_VARIANTS)
    if input_type in {None, "mixed"}:
        variants.update(_MIXED_VARIANTS)
    return tuple(
        f"{family}_{variant}"
        for family in _MATERIAL_FAMILIES
        for variant in variants
    )


def _registration_entries(variants: dict[str, str]) -> dict[str, tuple[str, str]]:
    entries: dict[str, tuple[str, str]] = {}
    for family in _MATERIAL_FAMILIES:
        prefix = _PREFIXES[family]
        for variant, class_suffix in variants.items():
            entries[f"{family}_{variant}"] = (
                _MATERIAL_NAMESPACE,
                f"{prefix}{class_suffix}",
            )
    return entries


def _task_registry(tree: MutableMapping[str, Any], input_type: str, task_type: str) -> MutableMapping[str, Any]:
    try:
        task_registry = tree[input_type][task_type]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "The default model registry does not expose the expected "
            f"{input_type!r}/{task_type!r} branch."
        ) from error
    if not isinstance(task_registry, MutableMapping):
        raise RuntimeError("The default model registry task branch must be mutable.")
    return task_registry


def register_material_residual_model_types() -> None:
    """Register residual material ``model_type`` names in the public API tree.

    The operation is idempotent.  Existing registrations are never silently
    replaced with a different path, protecting historical public model names.
    Correlated multitask residuals are additionally exposed under
    ``task_type='multi_objective'`` because they natively accept wide targets.
    """

    tree = DEFAULT_MODEL_REGISTRY.raw()
    normal_entries = _registration_entries(_NORMAL_VARIANTS)
    mixed_entries = _registration_entries(_MIXED_VARIANTS)

    targets = (
        ("normal", "regression", normal_entries),
        ("mixed", "regression", mixed_entries),
        (
            "normal",
            "multi_objective",
            {
                name: path
                for name, path in normal_entries.items()
                if "multitask_residual_gp" in name
            },
        ),
        (
            "mixed",
            "multi_objective",
            {
                name: path
                for name, path in mixed_entries.items()
                if "mixed_multitask_residual_gp" in name
            },
        ),
    )

    for input_type, task_type, entries in targets:
        registry = _task_registry(tree, input_type, task_type)
        for model_type, path in entries.items():
            existing = registry.get(model_type)
            if existing is not None and existing != path:
                raise RuntimeError(
                    f"Refusing to replace existing model_type {model_type!r}: "
                    f"{existing!r} != {path!r}."
                )
            registry[model_type] = path


__all__ = [
    "material_residual_model_types",
    "register_material_residual_model_types",
]
