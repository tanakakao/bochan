"""Converters between FastAPI/Pydantic schemas and bochan API dataclasses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any


def _dump(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=False)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Expected a Pydantic model or mapping. Got {type(value).__name__}.")


def _option_value(options: Any | None, name: str, default: Any = None) -> Any:
    if options is None:
        return default
    if isinstance(options, Mapping):
        return options.get(name, default)
    return getattr(options, name, default)


def _torch_dtype(dtype: Any | None) -> Any:
    """Resolve JSON-friendly dtype names to torch dtypes."""

    import torch

    if dtype is None:
        return torch.double
    if not isinstance(dtype, str):
        return dtype

    normalized = dtype.replace("torch.", "").lower()
    aliases = {
        "float64": torch.double,
        "double": torch.double,
        "float32": torch.float,
        "float": torch.float,
        "int64": torch.long,
        "long": torch.long,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported tensor dtype: {dtype!r}.")
    return aliases[normalized]


def to_tensor(value: Any, options: Any | None = None, *, dtype: Any | None = None) -> Any:
    """Convert JSON-like numeric data to a torch tensor."""

    if value is None:
        return None
    import torch

    tensor_dtype = _torch_dtype(dtype if dtype is not None else _option_value(options, "dtype", None))
    device = _option_value(options, "device", None)
    if torch.is_tensor(value):
        out = value.to(dtype=tensor_dtype)
        return out.to(device) if device is not None else out
    return torch.as_tensor(value, dtype=tensor_dtype, device=device)


def to_serializable(value: Any) -> Any:
    """Convert torch/numpy/dataclass objects and containers to JSON-serializable values."""

    if value is None:
        return None

    try:
        import torch

        if torch.is_tensor(value):
            detached = value.detach().cpu()
            if detached.ndim == 0:
                return detached.item()
            return detached.tolist()
    except Exception:
        pass

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass

    if is_dataclass(value):
        return to_serializable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [to_serializable(v) for v in value]
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _normalize_fixed_features(value: Any) -> dict[int, float] | None:
    """Normalize JSON object keys to BoTorch fixed feature indices."""

    if value is None:
        return None
    return {int(key): float(val) for key, val in dict(value).items()}


def _normalize_fixed_features_list(value: Any) -> list[dict[int, float]] | None:
    if value is None:
        return None
    return [_normalize_fixed_features(item) or {} for item in value]


def _normalize_linear_constraints(value: Any, options: Any | None = None) -> Any:
    """Convert JSON constraints to BoTorch-compatible linear constraints.

    Accepted formats:
        - {"indices": [0, 1], "coefficients": [1.0, 1.0], "rhs": 1.0}
        - [[0, 1], [1.0, 1.0], 1.0]
    """

    if value is None:
        return None

    import torch

    constraints = []
    for item in value:
        if isinstance(item, Mapping):
            indices = item["indices"]
            coefficients = item["coefficients"]
            rhs = item["rhs"]
        else:
            indices, coefficients, rhs = item
        constraints.append(
            (
                to_tensor(indices, options, dtype=torch.long),
                to_tensor(coefficients, options),
                float(rhs),
            )
        )
    return constraints


def _maybe_tensor_keys(data: dict[str, Any], keys: Sequence[str], options: Any | None = None) -> None:
    for key in keys:
        if data.get(key) is not None:
            data[key] = to_tensor(data[key], options)


def _normalize_acqf_kwargs(value: Any, options: Any | None = None) -> dict[str, Any]:
    """Normalize common tensor-like acquisition kwargs accepted through JSON."""

    data = dict(value or {})
    _maybe_tensor_keys(
        data,
        (
            "X_baseline",
            "X_pending",
            "Y_baseline",
            "best_f",
            "ref_point",
            "objective_thresholds",
            "mc_points",
        ),
        options,
    )
    if data.get("inequality_constraints") is not None:
        data["inequality_constraints"] = _normalize_linear_constraints(data["inequality_constraints"], options)
    if data.get("equality_constraints") is not None:
        data["equality_constraints"] = _normalize_linear_constraints(data["equality_constraints"], options)
    return data


def to_input_transform_config(value: Any, options: Any | None = None) -> Any:
    from bochan.api import InputTransformConfig

    if value is None or isinstance(value, InputTransformConfig):
        return value
    data = _dump(value)
    if data.get("bounds") is not None:
        data["bounds"] = to_tensor(data["bounds"], options)
    return InputTransformConfig(**data)


def to_fit_config(value: Any) -> Any:
    from bochan.api import FitConfig

    if value is None:
        return FitConfig()
    if isinstance(value, FitConfig):
        return value
    return FitConfig(**_dump(value))


def to_output_config(value: Any, options: Any | None = None) -> Any:
    from bochan.api import OutputConfig

    if value is None or isinstance(value, OutputConfig):
        return value
    if isinstance(value, str):
        return value
    data = _dump(value)
    if data.get("input_transform_config") is not None:
        data["input_transform_config"] = to_input_transform_config(data["input_transform_config"], options)
    if data.get("fit_config") is not None:
        data["fit_config"] = to_fit_config(data["fit_config"])
    return OutputConfig(**data)


def to_multi_output_config(value: Any, options: Any | None = None) -> Any:
    from bochan.api import MultiOutputConfig

    if value is None or isinstance(value, MultiOutputConfig):
        return value
    data = _dump(value)
    if data.get("output_configs") is not None:
        data["output_configs"] = [to_output_config(item, options) for item in data["output_configs"]]
    if data.get("output_fit_configs") is not None:
        output_fit_configs = data["output_fit_configs"]
        if isinstance(output_fit_configs, Mapping) or hasattr(output_fit_configs, "model_dump"):
            data["output_fit_configs"] = to_fit_config(output_fit_configs)
        else:
            data["output_fit_configs"] = [to_fit_config(item) if item is not None else None for item in output_fit_configs]
    return MultiOutputConfig(**data)


def to_model_config(value: Any, options: Any | None = None) -> Any:
    from bochan.api import ModelConfig

    if isinstance(value, ModelConfig):
        return value
    data = _dump(value)
    if data.get("input_transform_config") is not None:
        data["input_transform_config"] = to_input_transform_config(data["input_transform_config"], options)
    if data.get("multi_output_config") is not None:
        data["multi_output_config"] = to_multi_output_config(data["multi_output_config"], options)
    return ModelConfig(**data)


def to_objective_config(value: Any) -> Any:
    from bochan.api import ObjectiveConfig

    if value is None or isinstance(value, ObjectiveConfig):
        return value
    return ObjectiveConfig(**_dump(value))


def to_outcome_constraint_config(value: Any) -> Any:
    from bochan.api import OutcomeConstraintConfig

    if value is None or isinstance(value, OutcomeConstraintConfig):
        return value
    return OutcomeConstraintConfig(**_dump(value))


def to_multi_objective_config(value: Any, options: Any | None = None) -> Any:
    from bochan.api import MultiObjectiveConfig

    if value is None or isinstance(value, MultiObjectiveConfig):
        return value
    data = _dump(value)
    _maybe_tensor_keys(
        data,
        (
            "ref_point",
            "Y_baseline",
            "objective_thresholds",
            "scalarization_weights",
        ),
        options,
    )
    return MultiObjectiveConfig(**data)


def to_data_context(value: Any, options: Any | None = None) -> Any:
    from bochan.api import DataContext

    if value is None or isinstance(value, DataContext):
        return value
    data = _dump(value)
    _maybe_tensor_keys(
        data,
        (
            "bounds",
            "X_baseline",
            "X_pending",
            "Y_baseline",
            "best_f",
            "ref_point",
            "objective_thresholds",
            "mc_points",
        ),
        options,
    )
    if data.get("multi_objective") is not None:
        data["multi_objective"] = to_multi_objective_config(data["multi_objective"], options)
    return DataContext(**data)


def to_candidate_repair_config(value: Any, options: Any | None = None) -> Any:
    from bochan.api import CandidateRepairConfig

    if value is None or isinstance(value, CandidateRepairConfig):
        return value
    data = _dump(value)
    _maybe_tensor_keys(data, ("bounds", "steps"), options)
    if data.get("equality_constraints") is not None:
        data["equality_constraints"] = _normalize_linear_constraints(data["equality_constraints"], options)
    if data.get("inequality_constraints") is not None:
        data["inequality_constraints"] = _normalize_linear_constraints(data["inequality_constraints"], options)
    if data.get("fixed_features") is not None:
        data["fixed_features"] = _normalize_fixed_features(data["fixed_features"])
    return CandidateRepairConfig(**data)


def to_optimize_config(value: Any, options: Any | None = None) -> Any:
    from bochan.api import OptimizeConfig

    if isinstance(value, OptimizeConfig):
        return value
    data = _dump(value)
    if data.get("repair_config") is not None:
        data["repair_config"] = to_candidate_repair_config(data["repair_config"], options)
    if data.get("fixed_features") is not None:
        data["fixed_features"] = _normalize_fixed_features(data["fixed_features"])
    if data.get("fixed_features_list") is not None:
        data["fixed_features_list"] = _normalize_fixed_features_list(data["fixed_features_list"])
    if data.get("equality_constraints") is not None:
        data["equality_constraints"] = _normalize_linear_constraints(data["equality_constraints"], options)
    if data.get("inequality_constraints") is not None:
        data["inequality_constraints"] = _normalize_linear_constraints(data["inequality_constraints"], options)
    return OptimizeConfig(**data)


def to_acquisition_config(value: Any, options: Any | None = None) -> Any:
    from bochan.api import AcquisitionConfig

    if isinstance(value, AcquisitionConfig):
        return value
    data = _dump(value)
    if data.get("objective_config") is not None:
        data["objective_config"] = to_objective_config(data["objective_config"])
    if data.get("outcome_constraint_config") is not None:
        data["outcome_constraint_config"] = to_outcome_constraint_config(data["outcome_constraint_config"])
    data["acqf_kwargs"] = _normalize_acqf_kwargs(data.get("acqf_kwargs"), options)
    return AcquisitionConfig(**data)


def model_metadata(optimizer: Any) -> dict[str, Any]:
    bundle = getattr(optimizer, "bundle", None)
    if bundle is None:
        return {}
    metadata = dict(getattr(bundle, "metadata", {}) or {})
    metadata.pop("sub_bundles", None)
    return to_serializable(metadata)
