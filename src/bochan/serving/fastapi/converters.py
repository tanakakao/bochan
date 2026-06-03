"""Converters between FastAPI/Pydantic schemas and bochan API dataclasses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _dump(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=False)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Expected a Pydantic model or mapping. Got {type(value).__name__}.")


def to_tensor(value: Any, *, dtype: Any | None = None) -> Any:
    """Convert JSON-like numeric data to a torch tensor."""

    if value is None:
        return None
    import torch

    if torch.is_tensor(value):
        return value
    return torch.as_tensor(value, dtype=dtype or torch.double)


def to_serializable(value: Any) -> Any:
    """Convert torch tensors and nested containers to JSON-serializable values."""

    try:
        import torch

        if torch.is_tensor(value):
            detached = value.detach().cpu()
            if detached.ndim == 0:
                return detached.item()
            return detached.tolist()
    except Exception:
        pass

    if isinstance(value, Mapping):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [to_serializable(v) for v in value]
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    return value


def to_input_transform_config(value: Any) -> Any:
    from bochan.api import InputTransformConfig

    if value is None or isinstance(value, InputTransformConfig):
        return value
    return InputTransformConfig(**_dump(value))


def to_fit_config(value: Any) -> Any:
    from bochan.api import FitConfig

    if value is None:
        return FitConfig()
    if isinstance(value, FitConfig):
        return value
    return FitConfig(**_dump(value))


def to_output_config(value: Any) -> Any:
    from bochan.api import OutputConfig

    if value is None or isinstance(value, OutputConfig):
        return value
    if isinstance(value, str):
        return value
    data = _dump(value)
    if data.get("input_transform_config") is not None:
        data["input_transform_config"] = to_input_transform_config(data["input_transform_config"])
    if data.get("fit_config") is not None:
        data["fit_config"] = to_fit_config(data["fit_config"])
    return OutputConfig(**data)


def to_multi_output_config(value: Any) -> Any:
    from bochan.api import MultiOutputConfig

    if value is None or isinstance(value, MultiOutputConfig):
        return value
    data = _dump(value)
    if data.get("output_configs") is not None:
        data["output_configs"] = [to_output_config(item) for item in data["output_configs"]]
    if data.get("output_fit_configs") is not None:
        output_fit_configs = data["output_fit_configs"]
        if isinstance(output_fit_configs, Mapping) or hasattr(output_fit_configs, "model_dump"):
            data["output_fit_configs"] = to_fit_config(output_fit_configs)
        else:
            data["output_fit_configs"] = [to_fit_config(item) if item is not None else None for item in output_fit_configs]
    return MultiOutputConfig(**data)


def to_model_config(value: Any) -> Any:
    from bochan.api import ModelConfig

    if isinstance(value, ModelConfig):
        return value
    data = _dump(value)
    if data.get("input_transform_config") is not None:
        data["input_transform_config"] = to_input_transform_config(data["input_transform_config"])
    if data.get("multi_output_config") is not None:
        data["multi_output_config"] = to_multi_output_config(data["multi_output_config"])
    return ModelConfig(**data)


def to_objective_config(value: Any) -> Any:
    from bochan.api import ObjectiveConfig

    if value is None or isinstance(value, ObjectiveConfig):
        return value
    return ObjectiveConfig(**_dump(value))


def to_multi_objective_config(value: Any) -> Any:
    from bochan.api import MultiObjectiveConfig

    if value is None or isinstance(value, MultiObjectiveConfig):
        return value
    data = _dump(value)
    for key in ("ref_point", "Y_baseline", "objective_thresholds"):
        if data.get(key) is not None:
            data[key] = to_tensor(data[key])
    return MultiObjectiveConfig(**data)


def to_data_context(value: Any) -> Any:
    from bochan.api import DataContext

    if value is None or isinstance(value, DataContext):
        return value
    data = _dump(value)
    for key in (
        "bounds",
        "X_baseline",
        "X_pending",
        "Y_baseline",
        "ref_point",
        "objective_thresholds",
        "mc_points",
    ):
        if data.get(key) is not None:
            data[key] = to_tensor(data[key])
    if data.get("multi_objective") is not None:
        data["multi_objective"] = to_multi_objective_config(data["multi_objective"])
    return DataContext(**data)


def to_candidate_repair_config(value: Any) -> Any:
    from bochan.api import CandidateRepairConfig

    if value is None or isinstance(value, CandidateRepairConfig):
        return value
    data = _dump(value)
    if data.get("bounds") is not None:
        data["bounds"] = to_tensor(data["bounds"])
    return CandidateRepairConfig(**data)


def to_optimize_config(value: Any) -> Any:
    from bochan.api import OptimizeConfig

    if isinstance(value, OptimizeConfig):
        return value
    data = _dump(value)
    if data.get("repair_config") is not None:
        data["repair_config"] = to_candidate_repair_config(data["repair_config"])
    return OptimizeConfig(**data)


def to_acquisition_config(value: Any) -> Any:
    from bochan.api import AcquisitionConfig

    if isinstance(value, AcquisitionConfig):
        return value
    data = _dump(value)
    if data.get("objective_config") is not None:
        data["objective_config"] = to_objective_config(data["objective_config"])
    return AcquisitionConfig(**data)


def model_metadata(optimizer: Any) -> dict[str, Any]:
    bundle = getattr(optimizer, "bundle", None)
    if bundle is None:
        return {}
    metadata = dict(getattr(bundle, "metadata", {}) or {})
    metadata.pop("sub_bundles", None)
    return to_serializable(metadata)
