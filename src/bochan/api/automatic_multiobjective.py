"""Automatic multi-objective baselines for high-level acquisitions."""

from __future__ import annotations

from typing import Any

from .automatic_default_utils import (
    _call_objective,
    _direction_sign,
    _infer_ordinal_utility_values,
    _num_outputs,
    _objective_config_value,
)
from .configs import AcquisitionConfig, DataContext, ModelBundle


def _regression_observed_values(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> Any:
    from .factory import build_objective

    values = bundle.train_Y
    objective = build_objective(bundle=bundle, config=config, data_context=context)
    if objective is not None:
        values = _call_objective(objective, values, bundle.train_X)
    return values


def _binary_observed_values(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> Any:
    import torch

    from bochan.acquisition.binary.bayesian_optimization._utils import (
        ensure_q_batch,
        get_model_posterior,
        normalize_mean_shape,
        shape_X_for_model,
        to_probability,
    )
    from .factory import build_objective

    Xq = ensure_q_batch(bundle.train_X)
    with torch.no_grad():
        shape_X = shape_X_for_model(bundle.model, Xq)
        posterior = get_model_posterior(
            bundle.model,
            Xq,
            samples_are_probs=True,
        )
        values = normalize_mean_shape(posterior.mean, shape_X)
        values = to_probability(
            values,
            apply_sigmoid_if_needed=config.acqf_kwargs.get(
                "apply_sigmoid_if_needed",
                False,
            ),
            eps=config.acqf_kwargs.get("eps", 1e-6),
            name="posterior.mean",
            model=bundle.model,
        )
    while values.ndim > 2 and values.shape[0] == 1:
        values = values.squeeze(0)
    objective = build_objective(bundle=bundle, config=config, data_context=context)
    if objective is not None:
        values = _call_objective(objective, values, bundle.train_X)
    return values


def _ordinal_observed_values(bundle: ModelBundle, config: AcquisitionConfig) -> Any:
    from bochan.acquisition.ordinal.bayesian_optimization import (
        compute_observed_ordinal_utility,
    )

    utility_values = config.acqf_kwargs.get("utility_values")
    if utility_values is None:
        utility_values = _objective_config_value(config, "utility_values")
    if utility_values is None:
        utility_values = _infer_ordinal_utility_values(bundle.model)

    n_outputs = _num_outputs(bundle.train_Y)
    if n_outputs > 1 and not (
        isinstance(utility_values, (list, tuple))
        and len(utility_values) == n_outputs
        and (
            isinstance(utility_values[0], (list, tuple))
            or hasattr(utility_values[0], "shape")
        )
    ):
        utility_values = [utility_values for _ in range(n_outputs)]

    objective_signs = config.acqf_kwargs.get("objective_signs")
    if objective_signs is None:
        directions = _objective_config_value(config, "directions")
        if directions is not None:
            objective_signs = [_direction_sign(direction) for direction in directions]
    return compute_observed_ordinal_utility(
        bundle.train_Y,
        utility_values=utility_values,
        objective_signs=objective_signs,
        class_offset=config.acqf_kwargs.get("class_offset", 0),
    )


def _multiclass_observed_values(bundle: ModelBundle, config: AcquisitionConfig) -> Any:
    from bochan.acquisition.multiclass.bayesian_optimization import (
        compute_observed_multiclass_utility,
    )

    utility_values = config.acqf_kwargs.get("utility_values")
    objective_signs = config.acqf_kwargs.get("objective_signs")
    if objective_signs is None:
        directions = _objective_config_value(config, "directions")
        if directions is not None:
            objective_signs = [_direction_sign(direction) for direction in directions]
    return compute_observed_multiclass_utility(
        bundle.train_Y,
        target_class=config.acqf_kwargs.get("target_class"),
        output_target_classes=config.acqf_kwargs.get("output_target_classes"),
        class_reduction=config.acqf_kwargs.get("class_reduction", "mean"),
        utility_values=utility_values,
        objective_signs=objective_signs,
        class_offset=config.acqf_kwargs.get("class_offset", 0),
    )


def _observed_multiobjective_values(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> Any:
    task_type = str(bundle.task_type)
    if task_type == "binary":
        values = _binary_observed_values(bundle, config, context)
    elif task_type == "ordinal":
        values = _ordinal_observed_values(bundle, config)
    elif task_type == "multiclass":
        values = _multiclass_observed_values(bundle, config)
    else:
        values = _regression_observed_values(bundle, config, context)

    import torch

    values = torch.as_tensor(values)
    train_X = bundle.train_X
    if torch.is_tensor(train_X) and torch.is_floating_point(train_X):
        values = values.to(device=train_X.device, dtype=train_X.dtype)
    elif not torch.is_floating_point(values):
        values = values.to(dtype=torch.get_default_dtype())
    if values.ndim == 1:
        values = values.unsqueeze(-1)
    while values.ndim > 2 and values.shape[0] == 1:
        values = values.squeeze(0)
    if values.ndim > 2:
        values = values.reshape(-1, values.shape[-1])
    if values.ndim != 2:
        raise RuntimeError(
            "Could not convert observed objective values to shape [n, m]. "
            f"Got shape={tuple(values.shape)}."
        )
    return values.detach()


def _make_default_ref_point(values: Any, margin: float = 0.1) -> Any:
    """Create a maximization-space reference point below all observed values."""

    return (values.min(dim=-2).values - float(margin)).detach()


def _make_partitioning(ref_point: Any, values: Any) -> Any:
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        FastNondominatedPartitioning,
    )

    return FastNondominatedPartitioning(ref_point=ref_point, Y=values)


def observed_multiobjective_values(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> Any:
    """Map observations to a two-dimensional maximization objective tensor."""

    return _observed_multiobjective_values(bundle, config, context)


def make_default_ref_point(values: Any, margin: float = 0.1) -> Any:
    """Return an objective-wise reference point below all observations."""

    return _make_default_ref_point(values, margin=margin)


def make_partitioning(ref_point: Any, values: Any) -> Any:
    """Build the EHVI non-dominated partitioning."""

    return _make_partitioning(ref_point, values)
