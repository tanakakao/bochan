"""Target-fidelity helpers for candidate optimization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any


def target_fidelity_fixed_features(model: Any) -> dict[int, float]:
    """Return resolved target fidelities as optimizer fixed features.

    Models created by the generic long-format multi-fidelity subsystem expose
    ``target_fidelities`` directly and through ``fidelity_metadata``. This
    helper intentionally returns an empty mapping for non-MF models or MF models
    without configured targets so the high-level optimizer can stay generic.
    """

    targets = getattr(model, "target_fidelities", None)
    if targets is None:
        metadata = getattr(model, "fidelity_metadata", None)
        if callable(metadata):
            metadata = metadata()
        if isinstance(metadata, Mapping):
            targets = metadata.get("target_fidelities")

    if targets is None:
        return {}
    if not isinstance(targets, Mapping):
        raise TypeError("target_fidelities must be a mapping of feature index to value.")

    return {int(key): float(value) for key, value in targets.items()}


def merge_target_fidelities_into_opt_config(
    opt_config: Any,
    *,
    model: Any,
) -> Any:
    """Merge model target fidelities into an ``OptimizeConfig``-like object.

    User-specified fixed features remain valid, but conflicting values on a
    fidelity feature are rejected rather than silently overriding the model's
    declared target fidelity.
    """

    targets = target_fidelity_fixed_features(model)
    if not targets:
        return opt_config

    existing = {int(k): float(v) for k, v in (opt_config.fixed_features or {}).items()}
    for index, value in targets.items():
        if index in existing and existing[index] != value:
            raise ValueError(
                "OptimizeConfig.fixed_features conflicts with the model target fidelity: "
                f"feature {index} has {existing[index]!r}, expected {value!r}."
            )
        existing[index] = value

    fixed_features_list = opt_config.fixed_features_list
    if fixed_features_list is not None:
        merged_list: list[dict[int, float]] = []
        for assignment in fixed_features_list:
            item = {int(k): float(v) for k, v in assignment.items()}
            for index, value in targets.items():
                if index in item and item[index] != value:
                    raise ValueError(
                        "OptimizeConfig.fixed_features_list conflicts with the model target fidelity: "
                        f"feature {index} has {item[index]!r}, expected {value!r}."
                    )
            merged_list.append(item)
        fixed_features_list = merged_list

    return replace(
        opt_config,
        fixed_features=existing,
        fixed_features_list=fixed_features_list,
    )


__all__ = [
    "merge_target_fidelities_into_opt_config",
    "target_fidelity_fixed_features",
]
