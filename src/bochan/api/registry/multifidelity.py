"""Public model-type registration for long-format Gaussian multi-fidelity GPs."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from .model import DEFAULT_MODEL_REGISTRY

_MODEL_TYPE = "multifidelity_gp"
_ADAPTER_PATH = (
    "bochan.models.multifidelity.configured",
    "create_configured_fidelity_surrogate",
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


def register_multifidelity_gp_model_types() -> None:
    """Register ``multifidelity_gp`` for normal and mixed regression inputs.

    The historical ``model_type='multifidelity'`` entries are intentionally left
    unchanged because they represent the existing wide-format contract.
    Registration is idempotent and refuses conflicting replacements.
    """

    tree = DEFAULT_MODEL_REGISTRY.raw()
    for input_type in ("normal", "mixed"):
        registry = _task_registry(tree, input_type, "regression")
        existing = registry.get(_MODEL_TYPE)
        if existing is not None and existing != _ADAPTER_PATH:
            raise RuntimeError(
                f"Refusing to replace existing model_type {_MODEL_TYPE!r}: "
                f"{existing!r} != {_ADAPTER_PATH!r}."
            )
        registry[_MODEL_TYPE] = _ADAPTER_PATH


__all__ = ["register_multifidelity_gp_model_types"]
