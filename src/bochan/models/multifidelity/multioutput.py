"""Shared metadata helpers for independent multi-output multi-fidelity models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _metadata_value(model: Any, name: str, default: Any = None) -> Any:
    value = getattr(model, name, None)
    if value is not None:
        return value
    metadata = getattr(model, "fidelity_metadata", None)
    if callable(metadata):
        metadata = metadata()
    if isinstance(metadata, Mapping):
        return metadata.get(name, default)
    return default


def _features(model: Any) -> tuple[int, ...]:
    value = _metadata_value(model, "fidelity_features", ())
    return tuple(int(index) for index in (value or ()))


def _targets(model: Any) -> dict[int, float]:
    value = _metadata_value(model, "target_fidelities", {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("target_fidelities must be a mapping for multi-output MF models.")
    return {int(index): float(target) for index, target in value.items()}


def _cat_dims(model: Any) -> tuple[int, ...]:
    value = getattr(model, "cat_dims", None)
    return tuple(int(index) for index in (value or ()))


def _input_mode(model: Any) -> str:
    return str(getattr(model, "input_mode", "continuous"))


def shared_multifidelity_metadata(submodels: Sequence[Any]) -> dict[str, Any] | None:
    """Validate and return a shared fidelity contract for independent outputs.

    Returns ``None`` when none of the supplied models are multi-fidelity. If any
    output is multi-fidelity, all outputs must expose the same fidelity contract.
    Phase 54 intentionally supports a shared fidelity axis because one candidate
    row is evaluated for every output in the independent ``ModelListGP``.
    """

    models = list(submodels)
    if not models:
        return None

    features = [_features(model) for model in models]
    has_fidelity = [bool(value) for value in features]
    if not any(has_fidelity):
        return None
    if not all(has_fidelity):
        raise ValueError(
            "Independent multi-output multi-fidelity models require every output "
            "to use a multi-fidelity surrogate."
        )

    first_features = features[0]
    if any(value != first_features for value in features[1:]):
        raise ValueError(
            "All outputs in an independent multi-output multi-fidelity model must "
            "use the same fidelity_features."
        )

    targets = [_targets(model) for model in models]
    first_targets = targets[0]
    if any(value != first_targets for value in targets[1:]):
        raise ValueError(
            "All outputs in an independent multi-output multi-fidelity model must "
            "use the same target_fidelities."
        )

    input_modes = [_input_mode(model) for model in models]
    first_mode = input_modes[0]
    if any(value != first_mode for value in input_modes[1:]):
        raise ValueError(
            "All outputs in an independent multi-output multi-fidelity model must "
            "use the same input mode."
        )

    cat_dims = [_cat_dims(model) for model in models]
    first_cat_dims = cat_dims[0]
    if any(value != first_cat_dims for value in cat_dims[1:]):
        raise ValueError(
            "All outputs in an independent multi-output multi-fidelity model must "
            "use the same categorical dimensions."
        )

    return {
        "fidelity_mode": "feature",
        "fidelity_features": first_features,
        "target_fidelities": first_targets,
        "input_mode": first_mode,
        "cat_dims": first_cat_dims,
        "multi_output_fidelity": "independent",
        "num_fidelity_outputs": len(models),
    }


def bind_shared_multifidelity_metadata(wrapper: Any, submodels: Sequence[Any]) -> dict[str, Any] | None:
    """Validate submodels and expose their shared fidelity contract on a wrapper."""

    metadata = shared_multifidelity_metadata(submodels)
    if metadata is None:
        return None

    for name in (
        "fidelity_mode",
        "fidelity_features",
        "target_fidelities",
        "input_mode",
        "cat_dims",
        "multi_output_fidelity",
    ):
        setattr(wrapper, name, metadata[name])
    wrapper.is_multifidelity_model = True
    return metadata


__all__ = [
    "bind_shared_multifidelity_metadata",
    "shared_multifidelity_metadata",
]
