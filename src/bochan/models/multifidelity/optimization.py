"""Target and discrete-fidelity helpers for candidate optimization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import product
from typing import Any


def _fidelity_features(model: Any) -> tuple[int, ...]:
    features = getattr(model, "fidelity_features", None)
    if features is None:
        metadata = getattr(model, "fidelity_metadata", None)
        if callable(metadata):
            metadata = metadata()
        if isinstance(metadata, Mapping):
            features = metadata.get("fidelity_features")
    if features is None:
        return ()
    return tuple(int(index) for index in features)


def target_fidelity_fixed_features(model: Any) -> dict[int, float]:
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


def _validate_fidelity_values(values: Sequence[float], *, bounds: Any, index: int) -> tuple[float, ...]:
    resolved = tuple(float(value) for value in values)
    if not resolved:
        raise ValueError("fidelity_values must not be empty when supplied.")
    if bounds is not None:
        lower = float(bounds[0][index])
        upper = float(bounds[1][index])
        outside = [value for value in resolved if value < lower or value > upper]
        if outside:
            raise ValueError(
                f"fidelity_values must lie within bounds for feature {index}: "
                f"[{lower}, {upper}], got {outside}."
            )
    return resolved


def enumerate_discrete_fidelities_into_opt_config(
    opt_config: Any,
    *,
    model: Any,
    bounds: Any = None,
) -> Any:
    """Expand discrete fidelity choices into mixed fixed-feature assignments.

    Phase 49 formally supports one fidelity feature. Existing categorical or
    user-provided ``fixed_features_list`` assignments are crossed with every
    requested fidelity value. Global non-fidelity ``fixed_features`` remain
    global and are merged by the existing optimizer factory.
    """

    values = getattr(opt_config, "fidelity_values", None)
    if values is None:
        return opt_config

    features = _fidelity_features(model)
    if len(features) != 1:
        if not features:
            raise ValueError("fidelity_values requires a multi-fidelity model with one fidelity feature.")
        raise NotImplementedError("Discrete fidelity optimization currently supports exactly one fidelity feature.")
    index = features[0]
    values = _validate_fidelity_values(values, bounds=bounds, index=index)

    fixed = {int(k): float(v) for k, v in (opt_config.fixed_features or {}).items()}
    if index in fixed:
        if fixed[index] not in values:
            raise ValueError(
                "OptimizeConfig.fixed_features fixes the fidelity outside fidelity_values: "
                f"feature {index}={fixed[index]!r}, values={values!r}."
            )
        return replace(opt_config, fidelity_values=values)

    base_list = opt_config.fixed_features_list
    if base_list is None:
        base_list = [{}]
    if len(base_list) == 0:
        raise ValueError("fixed_features_list must not be empty when supplied.")

    expanded: list[dict[int, float]] = []
    for assignment, value in product(base_list, values):
        item = {int(k): float(v) for k, v in assignment.items()}
        if index in item and item[index] != value:
            continue
        item[index] = value
        expanded.append(item)
    if not expanded:
        raise ValueError("No valid fixed-feature assignments remain after applying fidelity_values.")

    return replace(
        opt_config,
        fidelity_values=values,
        fixed_features_list=expanded,
    )


def merge_target_fidelities_into_opt_config(opt_config: Any, *, model: Any) -> Any:
    """Merge model target fidelities unless discrete fidelity search is active."""

    if getattr(opt_config, "fidelity_values", None) is not None:
        return opt_config
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

    return replace(opt_config, fixed_features=existing, fixed_features_list=fixed_features_list)


__all__ = [
    "enumerate_discrete_fidelities_into_opt_config",
    "merge_target_fidelities_into_opt_config",
    "target_fidelity_fixed_features",
]
