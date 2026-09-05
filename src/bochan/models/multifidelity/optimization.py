"""Target, discrete, and continuous-fidelity helpers for candidate optimization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import product
from typing import Any

from .multioutput import shared_multifidelity_metadata


def _wrapper_fidelity_metadata(model: Any) -> dict[str, Any] | None:
    models = getattr(model, "models", None)
    if models is None:
        return None
    try:
        return shared_multifidelity_metadata(list(models))
    except TypeError:
        return None


def _fidelity_features(model: Any) -> tuple[int, ...]:
    features = getattr(model, "fidelity_features", None)
    if features is None:
        metadata = getattr(model, "fidelity_metadata", None)
        if callable(metadata):
            metadata = metadata()
        if isinstance(metadata, Mapping):
            features = metadata.get("fidelity_features")
    if features is None:
        wrapper_metadata = _wrapper_fidelity_metadata(model)
        if wrapper_metadata is not None:
            features = wrapper_metadata.get("fidelity_features")
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
        wrapper_metadata = _wrapper_fidelity_metadata(model)
        if wrapper_metadata is not None:
            targets = wrapper_metadata.get("target_fidelities")
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


def _validate_continuous_fidelity_search(
    opt_config: Any,
    *,
    model: Any,
    bounds: Any,
) -> Any:
    """Validate that the configured fidelity feature is free for joint search."""

    if not bool(getattr(opt_config, "optimize_fidelity", False)):
        return opt_config

    if getattr(opt_config, "fidelity_values", None) is not None:
        raise ValueError(
            "fidelity_values and optimize_fidelity=True are mutually exclusive."
        )

    features = _fidelity_features(model)
    if len(features) != 1:
        if not features:
            raise ValueError(
                "optimize_fidelity=True requires a multi-fidelity model with one fidelity feature."
            )
        raise NotImplementedError(
            "Continuous fidelity optimization currently supports exactly one fidelity feature."
        )
    index = features[0]

    if bounds is None:
        raise ValueError("Continuous fidelity optimization requires bounds.")
    try:
        lower = float(bounds[0][index])
        upper = float(bounds[1][index])
    except (IndexError, TypeError) as exc:
        raise ValueError(
            f"bounds do not contain fidelity feature {index}."
        ) from exc
    if not lower <= upper:
        raise ValueError(
            f"Fidelity bounds must satisfy lower <= upper, got [{lower}, {upper}]."
        )

    fixed = {int(k): float(v) for k, v in (getattr(opt_config, "fixed_features", None) or {}).items()}
    if index in fixed:
        raise ValueError(
            "optimize_fidelity=True conflicts with fixed_features fixing the fidelity: "
            f"feature {index}={fixed[index]!r}."
        )

    fixed_features_list = getattr(opt_config, "fixed_features_list", None)
    if fixed_features_list is not None:
        for assignment in fixed_features_list:
            if index in {int(k): v for k, v in assignment.items()}:
                raise ValueError(
                    "optimize_fidelity=True conflicts with fixed_features_list fixing "
                    f"fidelity feature {index}."
                )

    return opt_config


def enumerate_discrete_fidelities_into_opt_config(
    opt_config: Any,
    *,
    model: Any,
    bounds: Any = None,
) -> Any:
    """Expand discrete fidelity choices into mixed fixed-feature assignments."""

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


def prepare_continuous_fidelity_optimization(
    opt_config: Any,
    *,
    model: Any,
    bounds: Any,
) -> Any:
    """Prepare joint x/fidelity optimization without target-fidelity fixing."""

    return _validate_continuous_fidelity_search(
        opt_config,
        model=model,
        bounds=bounds,
    )


def merge_target_fidelities_into_opt_config(opt_config: Any, *, model: Any) -> Any:
    """Merge model target fidelities unless query-fidelity search is active."""

    if getattr(opt_config, "fidelity_values", None) is not None:
        return opt_config
    if bool(getattr(opt_config, "optimize_fidelity", False)):
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
    "prepare_continuous_fidelity_optimization",
    "target_fidelity_fixed_features",
]
