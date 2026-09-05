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


def _resolve_feature_index(raw_index: int, *, d: int) -> int:
    index = int(raw_index)
    if index < 0:
        index += int(d)
    if index < 0 or index >= int(d):
        raise ValueError(f"Invalid fidelity feature {raw_index} for input dimension {d}.")
    return index


def _model_dimension(model: Any, bounds: Any) -> int:
    if bounds is not None:
        try:
            return int(bounds.shape[-1])
        except AttributeError:
            return int(len(bounds[0]))
    train_X = getattr(model, "train_X_raw", None)
    if train_X is None:
        train_X = getattr(model, "train_inputs", None)
        if isinstance(train_X, tuple) and train_X:
            train_X = train_X[0]
    if train_X is None:
        raise ValueError("Cannot resolve negative fidelity indices without bounds or model inputs.")
    return int(train_X.shape[-1])


def _validate_fidelity_values(
    values: Sequence[float],
    *,
    bounds: Any,
    index: int,
) -> tuple[float, ...]:
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


def _resolved_fidelity_value_map(
    values: Any,
    *,
    features: tuple[int, ...],
    bounds: Any,
    model: Any,
) -> dict[int, tuple[float, ...]]:
    if isinstance(values, Mapping):
        d = _model_dimension(model, bounds)
        resolved: dict[int, tuple[float, ...]] = {}
        for raw_index, raw_values in values.items():
            index = _resolve_feature_index(int(raw_index), d=d)
            if index not in features:
                raise ValueError(
                    f"fidelity_values key {raw_index} resolves to feature {index}, which is not "
                    "a configured fidelity feature."
                )
            if index in resolved:
                raise ValueError(
                    "fidelity_values contains duplicate keys after negative-index resolution."
                )
            resolved[index] = _validate_fidelity_values(
                raw_values,
                bounds=bounds,
                index=index,
            )
        missing = [index for index in features if index not in resolved]
        if missing:
            raise ValueError(
                "Multidimensional fidelity_values must provide allowed values for every fidelity "
                f"feature; missing={missing}."
            )
        return resolved

    if len(features) != 1:
        raise ValueError(
            "Sequence fidelity_values is only valid for a model with one fidelity feature. "
            "Use a mapping {feature: values} for multiple fidelity dimensions."
        )
    index = features[0]
    return {index: _validate_fidelity_values(values, bounds=bounds, index=index)}


def _resolved_explicit_assignments(
    assignments: Sequence[Mapping[int, float]],
    *,
    features: tuple[int, ...],
    bounds: Any,
    model: Any,
) -> tuple[dict[int, float], ...]:
    d = _model_dimension(model, bounds)
    resolved_assignments: list[dict[int, float]] = []
    seen: set[tuple[tuple[int, float], ...]] = set()
    for raw_assignment in assignments:
        resolved: dict[int, float] = {}
        for raw_index, raw_value in raw_assignment.items():
            index = _resolve_feature_index(int(raw_index), d=d)
            if index not in features:
                raise ValueError(
                    f"fidelity_assignments key {raw_index} resolves to feature {index}, which is "
                    "not a configured fidelity feature."
                )
            if index in resolved:
                raise ValueError(
                    "fidelity_assignments contains duplicate keys after negative-index resolution."
                )
            value = float(raw_value)
            _validate_fidelity_values((value,), bounds=bounds, index=index)
            resolved[index] = value
        missing = [index for index in features if index not in resolved]
        if missing:
            raise ValueError(
                "Each fidelity_assignments item must provide every configured fidelity feature; "
                f"missing={missing}."
            )
        key = tuple(sorted(resolved.items()))
        if key in seen:
            raise ValueError("fidelity_assignments must not contain duplicate assignments.")
        seen.add(key)
        resolved_assignments.append(resolved)
    if not resolved_assignments:
        raise ValueError("fidelity_assignments must not be empty when supplied.")
    return tuple(resolved_assignments)


def _validate_continuous_fidelity_search(
    opt_config: Any,
    *,
    model: Any,
    bounds: Any,
) -> Any:
    """Validate that every configured fidelity feature is free for joint search."""

    if not bool(getattr(opt_config, "optimize_fidelity", False)):
        return opt_config

    if getattr(opt_config, "fidelity_values", None) is not None or getattr(
        opt_config, "fidelity_assignments", None
    ) is not None:
        raise ValueError(
            "fidelity_values / fidelity_assignments and optimize_fidelity=True are mutually exclusive."
        )

    features = _fidelity_features(model)
    if not features:
        raise ValueError("optimize_fidelity=True requires a multi-fidelity model.")
    if bounds is None:
        raise ValueError("Continuous fidelity optimization requires bounds.")

    for index in features:
        try:
            lower = float(bounds[0][index])
            upper = float(bounds[1][index])
        except (IndexError, TypeError) as exc:
            raise ValueError(f"bounds do not contain fidelity feature {index}.") from exc
        if not lower <= upper:
            raise ValueError(
                f"Fidelity bounds must satisfy lower <= upper, got [{lower}, {upper}] "
                f"for feature {index}."
            )

    fixed = {
        int(k): float(v)
        for k, v in (getattr(opt_config, "fixed_features", None) or {}).items()
    }
    conflicts = [index for index in features if index in fixed]
    if conflicts:
        raise ValueError(
            "optimize_fidelity=True conflicts with fixed_features fixing fidelity features: "
            f"{conflicts}."
        )

    fixed_features_list = getattr(opt_config, "fixed_features_list", None)
    if fixed_features_list is not None:
        for assignment in fixed_features_list:
            assignment_keys = {int(k) for k in assignment}
            conflicts = sorted(set(features).intersection(assignment_keys))
            if conflicts:
                raise ValueError(
                    "optimize_fidelity=True conflicts with fixed_features_list fixing fidelity "
                    f"features: {conflicts}."
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
    explicit = getattr(opt_config, "fidelity_assignments", None)
    if values is None and explicit is None:
        return opt_config
    if values is not None and explicit is not None:
        raise ValueError("Specify either fidelity_values or fidelity_assignments, not both.")

    features = _fidelity_features(model)
    if not features:
        raise ValueError("Discrete fidelity search requires a multi-fidelity model.")

    if explicit is not None:
        fidelity_assignments = _resolved_explicit_assignments(
            explicit,
            features=features,
            bounds=bounds,
            model=model,
        )
    else:
        value_map = _resolved_fidelity_value_map(
            values,
            features=features,
            bounds=bounds,
            model=model,
        )
        fidelity_assignments = tuple(
            dict(zip(features, combination, strict=True))
            for combination in product(*(value_map[index] for index in features))
        )

    fixed = {int(k): float(v) for k, v in (opt_config.fixed_features or {}).items()}
    fixed_fidelity = {index: fixed[index] for index in features if index in fixed}
    for index, fixed_value in fixed_fidelity.items():
        if not any(assignment[index] == fixed_value for assignment in fidelity_assignments):
            raise ValueError(
                "OptimizeConfig.fixed_features fixes a fidelity outside the allowed assignments: "
                f"feature {index}={fixed_value!r}."
            )

    base_list = opt_config.fixed_features_list
    if base_list is None:
        base_list = [{}]
    if len(base_list) == 0:
        raise ValueError("fixed_features_list must not be empty when supplied.")

    expanded: list[dict[int, float]] = []
    for base_assignment, fidelity_assignment in product(base_list, fidelity_assignments):
        item = {int(k): float(v) for k, v in base_assignment.items()}
        conflict = False
        for index, value in fidelity_assignment.items():
            if index in fixed and fixed[index] != value:
                conflict = True
                break
            if index in item and item[index] != value:
                conflict = True
                break
            item[index] = value
        if not conflict:
            expanded.append(item)
    if not expanded:
        raise ValueError("No valid fixed-feature assignments remain after applying fidelity search.")

    return replace(
        opt_config,
        fidelity_assignments=fidelity_assignments,
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
    if getattr(opt_config, "fidelity_assignments", None) is not None:
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
                item[index] = value
            merged_list.append(item)
        fixed_features_list = merged_list

    return replace(
        opt_config,
        fixed_features=existing,
        fixed_features_list=fixed_features_list,
    )


__all__ = [
    "enumerate_discrete_fidelities_into_opt_config",
    "merge_target_fidelities_into_opt_config",
    "prepare_continuous_fidelity_optimization",
    "target_fidelity_fixed_features",
]
