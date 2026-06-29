"""Shared helpers for automatic Bayesian optimization defaults."""

from __future__ import annotations

from typing import Any

from .configs import AcquisitionConfig, ModelBundle


def _num_outputs(train_Y: Any) -> int:
    """Infer the number of target columns from tensor or sequence targets.

    Hybrid and auto-wrapped model bundles may retain targets as a sequence of
    tensors, one entry per sub-model. In that representation, the total output
    count is the sum of the output counts of each entry.
    """

    if isinstance(train_Y, (list, tuple)):
        if len(train_Y) == 0:
            raise ValueError("train_Y sequence must not be empty.")
        return sum(_num_outputs(target) for target in train_Y)

    shape = getattr(train_Y, "shape", None)
    if shape is None:
        raise TypeError(
            "train_Y must have a shape attribute or be a non-empty sequence "
            "of array-like targets."
        )
    if len(shape) == 0:
        return 1
    return 1 if len(shape) == 1 else int(shape[-1])


def _objective_config_value(
    config: AcquisitionConfig, name: str, default: Any = None
) -> Any:
    objective_config = config.objective_config
    if objective_config is None:
        return default
    value = getattr(objective_config, name, default)
    return default if value is None else value


def _sub_bundles(bundle: ModelBundle) -> list[ModelBundle]:
    """Return homogeneous sub-bundles when the model was auto-wrapped."""

    return list(bundle.metadata.get("sub_bundles", []) or [])


def _direction_sign(direction: Any) -> float:
    """Normalize a maximize/minimize direction to ``+1`` or ``-1``."""

    if isinstance(direction, str):
        if direction == "maximize":
            return 1.0
        if direction == "minimize":
            return -1.0
    if isinstance(direction, bool):
        return 1.0 if direction else -1.0
    try:
        return 1.0 if float(direction) > 0.0 else -1.0
    except (TypeError, ValueError):
        raise ValueError(f"Unsupported objective direction: {direction!r}.") from None


def _infer_ordinal_utility_values(model: Any) -> Any:
    """Infer ordinal utilities as ``0, ..., K-1`` from model likelihoods."""

    import torch

    models = list(getattr(model, "models", []) or [])
    objects = models if models else [model]
    utility_values = []
    for obj in objects:
        likelihood = getattr(obj, "ordinal_likelihood", None)
        if likelihood is None:
            likelihood = getattr(obj, "likelihood", None)
        num_classes = getattr(likelihood, "num_classes", None)
        if num_classes is None:
            num_classes = getattr(obj, "num_classes", None)
        if num_classes is None and likelihood is not None:
            for name in (
                "cutpoints",
                "thresholds",
                "ordered_cutpoints",
                "transformed_cutpoints",
                "raw_cutpoints",
                "raw_thresholds",
            ):
                if not hasattr(likelihood, name):
                    continue
                value = getattr(likelihood, name)
                value = value() if callable(value) else value
                num_classes = int(torch.as_tensor(value).numel()) + 1
                break
        if num_classes is None:
            raise ValueError(
                "Could not infer ordinal utility values. "
                "Pass ObjectiveConfig.utility_values explicitly."
            )
        utility_values.append(torch.arange(int(num_classes), dtype=torch.double))
    return utility_values[0] if len(utility_values) == 1 else utility_values


def _call_objective(objective: Any, values: Any, X: Any) -> Any:
    """Apply an MC objective to deterministic observed values."""

    import torch

    attempts = [values]
    if torch.is_tensor(values) and values.ndim <= 2:
        attempts.append(values.unsqueeze(0))
    last_error: Exception | None = None
    for candidate in attempts:
        for with_X in (True, False):
            try:
                result = objective(candidate, X=X) if with_X else objective(candidate)
                if not torch.is_tensor(result) and not isinstance(
                    result, (list, tuple)
                ):
                    result = torch.as_tensor(
                        result, device=values.device, dtype=values.dtype
                    )
                if torch.is_tensor(result):
                    while result.ndim > 2 and result.shape[0] == 1:
                        result = result.squeeze(0)
                return result
            except (TypeError, RuntimeError, ValueError) as exc:
                last_error = exc
    if last_error is not None:
        raise last_error
    return values
