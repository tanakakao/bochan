"""Public model-type registration for Gaussian multi-fidelity GPs."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from .model import DEFAULT_MODEL_REGISTRY

_MODEL_TYPE = "multifidelity_gp"
_ADAPTER_PATH = (
    "bochan.models.multifidelity.configured",
    "create_configured_fidelity_surrogate",
)
_CORRELATED_MODEL_TYPE = "correlated_multifidelity_gp"
_CORRELATED_ADAPTER_PATH = (
    "bochan.models.multifidelity.configured",
    "create_configured_correlated_fidelity_surrogate",
)
_SOURCE_MODEL_TYPES = ("multisource_gp", "information_source_gp")
_SOURCE_ADAPTER_PATH = (
    "bochan.models.multifidelity.configured",
    "create_configured_information_source_surrogate",
)


def _task_registry(
    tree: MutableMapping[str, Any],
    input_type: str,
    task_type: str,
) -> MutableMapping[str, Any]:
    try:
        registry = tree[input_type][task_type]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "The default model registry does not expose the expected "
            f"{input_type!r}/{task_type!r} branch."
        ) from error
    if not isinstance(registry, MutableMapping):
        raise RuntimeError("The default model registry task branch must be mutable.")
    return registry


def _register_alias(
    registry: MutableMapping[str, Any],
    *,
    model_type: str,
    adapter_path: tuple[str, str],
) -> None:
    existing = registry.get(model_type)
    if existing is not None and existing != adapter_path:
        raise RuntimeError(
            f"Refusing to replace existing model_type {model_type!r}: "
            f"{existing!r} != {adapter_path!r}."
        )
    registry[model_type] = adapter_path


def register_multifidelity_gp_model_types() -> None:
    """Register Gaussian fidelity and discrete information-source model types."""

    tree = DEFAULT_MODEL_REGISTRY.raw()
    for input_type in ("normal", "mixed"):
        registry = _task_registry(tree, input_type, "regression")
        _register_alias(
            registry,
            model_type=_MODEL_TYPE,
            adapter_path=_ADAPTER_PATH,
        )

    normal_registry = _task_registry(tree, "normal", "regression")
    _register_alias(
        normal_registry,
        model_type=_CORRELATED_MODEL_TYPE,
        adapter_path=_CORRELATED_ADAPTER_PATH,
    )
    for model_type in _SOURCE_MODEL_TYPES:
        _register_alias(
            normal_registry,
            model_type=model_type,
            adapter_path=_SOURCE_ADAPTER_PATH,
        )


__all__ = ["register_multifidelity_gp_model_types"]
