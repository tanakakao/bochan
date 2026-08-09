"""Automatic multi-objective baselines for high-level acquisitions."""

# ruff: noqa: I001

from __future__ import annotations

from typing import Any

from .automatic_default_utils import (
    _call_objective,
    _direction_sign,
    _infer_ordinal_utility_values,
    _num_outputs,
    _objective_config_value,
    _sub_bundles,
)
from .configs import AcquisitionConfig, DataContext, ModelBundle


def _observed_train_targets(bundle: ModelBundle) -> Any:
    """Return original observed targets, including auto-wrapped outputs.

    Multi-output wrappers such as ``ModelListGP`` may expose ``train_targets`` as
    a tuple of one-dimensional tensors. The original, untransformed targets are
    retained by the sub-bundles, so prefer those when available.
    """

    sub_bundles = _sub_bundles(bundle)
    if sub_bundles:
        targets = [sub.train_Y for sub in sub_bundles]
        if any(target is None for target in targets):
            raise ValueError("Every sub-bundle must provide train_Y.")
        return targets
    return bundle.train_Y


def _as_observed_matrix(
    values: Any,
    *,
    train_X: Any | None = None,
    match_train_x_dtype: bool = False,
) -> Any:
    """Normalize tensor-like observed values to shape ``[n, m]``.

    ``torch.as_tensor`` cannot directly convert ``list[Tensor]`` when each
    tensor has multiple elements. Model-list and hybrid wrappers commonly
    expose targets in exactly that form, so tensor sequences are converted to
    columns and concatenated explicitly.
    """

    import torch

    expected_rows = None
    train_x_shape = getattr(train_X, "shape", None)
    if train_x_shape is not None and len(train_x_shape) >= 2:
        expected_rows = int(train_x_shape[-2])

    def _normalize_part(part: Any) -> Any:
        tensor = part if torch.is_tensor(part) else torch.as_tensor(part)
        while tensor.ndim > 2 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim == 0:
            tensor = tensor.reshape(1, 1)
        elif tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)
        elif tensor.ndim > 2:
            tensor = tensor.reshape(-1, tensor.shape[-1])
        if (
            tensor.ndim == 2
            and expected_rows is not None
            and tensor.shape[0] != expected_rows
            and tensor.shape[1] == expected_rows
        ):
            tensor = tensor.transpose(-1, -2)
        if tensor.ndim != 2:
            raise RuntimeError(
                "Could not convert an observed target to shape [n, m]. "
                f"Got shape={tuple(tensor.shape)}."
            )
        return tensor

    if isinstance(values, (list, tuple)):
        if len(values) == 0:
            raise ValueError("Observed target sequence must not be empty.")
        parts = [_normalize_part(part) for part in values]
        row_counts = {int(part.shape[0]) for part in parts}
        if len(row_counts) != 1:
            raise RuntimeError(
                "Observed target tensors must have the same number of rows. "
                f"Got row counts={sorted(row_counts)}."
            )
        tensor = torch.cat(parts, dim=-1)
    else:
        tensor = _normalize_part(values)

    if match_train_x_dtype:
        if torch.is_tensor(train_X):
            tensor = tensor.to(device=train_X.device)
            if torch.is_floating_point(train_X):
                tensor = tensor.to(dtype=train_X.dtype)
        elif not torch.is_floating_point(tensor):
            tensor = tensor.to(dtype=torch.get_default_dtype())
    return tensor


def _complete_observed_rows(values: Any, *, source: str) -> Any:
    """Keep rows with all objectives observed for Pareto baseline operations.

    Correlated multi-task models may train from partially observed wide targets,
    but a Pareto point, reference point, and EHVI partitioning require a complete
    objective vector. Missing cells therefore remain valid for model fitting but
    their rows are excluded from observed multi-objective baselines.
    """

    import torch

    tensor = values if torch.is_tensor(values) else torch.as_tensor(values)
    if tensor.ndim != 2:
        raise RuntimeError(
            f"{source} must have shape [n, m] before complete-row filtering. "
            f"Got shape={tuple(tensor.shape)}."
        )
    finite = torch.isfinite(tensor)
    complete = finite.all(dim=-1)
    if bool(complete.any()):
        return tensor[complete]

    observed_counts = finite.sum(dim=0).detach().cpu().tolist()
    raise ValueError(
        "Automatic multi-objective baselines require at least one training row "
        "with every objective observed. Partially observed rows remain usable for "
        f"model fitting, but {source} contains no complete objective vector. "
        f"Observed counts per output={observed_counts}."
    )


def _regression_observed_values(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> Any:
    from .factory import build_objective

    values = _as_observed_matrix(
        _observed_train_targets(bundle),
        train_X=bundle.train_X,
    )
    objective = build_objective(bundle=bundle, config=config, data_context=context)
    if objective is not None:
        values = _call_objective(objective, values, bundle.train_X)
    return values


def _normalize_binary_posterior_mean(
    bundle: ModelBundle,
    mean: Any,
    shape_X: Any,
) -> Any:
    """Preserve the explicit output axis of binary multi-output posteriors.

    Projected models with one-to-many input transforms can expose posterior means
    as ``[..., q * n_w, m]`` while ``shape_X_for_model`` falls back to the raw
    ``[..., q, d]`` input shape. Blindly normalizing by the raw point count then
    folds ``n_w`` into the output axis and turns ``m`` objectives into
    ``n_w * m`` objectives. When the last dimension already matches the trained
    target count, it is the explicit output axis and must be retained.
    """

    from bochan.acquisition.binary.bayesian_optimization._utils import (
        normalize_mean_shape,
    )

    n_outputs = _num_outputs(_observed_train_targets(bundle))
    if (
        n_outputs > 1
        and getattr(mean, "ndim", 0) >= 2
        and int(mean.shape[-1]) == n_outputs
    ):
        return mean
    return normalize_mean_shape(mean, shape_X)


def _binary_observed_values(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> Any:
    import torch

    from bochan.acquisition.binary.bayesian_optimization._utils import (
        ensure_q_batch,
        get_model_posterior,
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
        values = _normalize_binary_posterior_mean(
            bundle,
            posterior.mean,
            shape_X,
        )
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

    train_Y = _as_observed_matrix(
        _observed_train_targets(bundle),
        train_X=bundle.train_X,
    )
    train_Y = _complete_observed_rows(train_Y, source="ordinal train_Y")
    n_outputs = _num_outputs(train_Y)
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
        train_Y,
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
    train_Y = _as_observed_matrix(
        _observed_train_targets(bundle),
        train_X=bundle.train_X,
    )
    train_Y = _complete_observed_rows(train_Y, source="multiclass train_Y")
    return compute_observed_multiclass_utility(
        train_Y,
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

    values = _as_observed_matrix(
        values,
        train_X=bundle.train_X,
        match_train_x_dtype=True,
    )
    values = _complete_observed_rows(values, source="multi-objective values")
    return values.detach()


def _make_default_ref_point(values: Any, margin: float = 0.1) -> Any:
    """Create a NaN-safe maximization-space reference point."""

    from .nan_multiobjective import make_nan_safe_default_ref_point

    return make_nan_safe_default_ref_point(values, margin=margin)


def _make_partitioning(ref_point: Any, values: Any) -> Any:
    """Build an EHVI partitioning from complete finite objective rows."""

    from .nan_multiobjective import make_nan_safe_partitioning

    return make_nan_safe_partitioning(ref_point, values)


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
