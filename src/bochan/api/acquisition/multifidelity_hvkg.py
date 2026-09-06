"""Multi-objective multi-fidelity Hypervolume Knowledge Gradient support."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    _get_hv_value_function,
    qMultiFidelityHypervolumeKnowledgeGradient,
)
from botorch.optim import optimize_acqf, optimize_acqf_mixed

from ..configs import AcquisitionConfig, DataContext, ModelBundle
from .multifidelity import (
    _attach_cost_metadata,
    _bounds_tensor,
    _categorical_assignments,
    _model_dimension,
    _projector,
    _reference_train_Y,
    _resolve_cost_aware_utility,
    _target_fidelities,
)

_MFHVKG_NAMES = {
    "mfhvkg",
    "qmfhvkg",
    "multifidelityhypervolumeknowledgegradient",
    "qmultifidelityhypervolumeknowledgegradient",
}


def _normalize_name(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def is_multifidelity_hvkg_name(name: Any) -> bool:
    """Return whether ``name`` identifies the MF-HVKG acquisition."""

    return _normalize_name(name) in _MFHVKG_NAMES


def _bundle_is_multi_output(bundle: ModelBundle) -> bool:
    """Infer multi-output capability while preserving explicit metadata overrides."""

    metadata = getattr(bundle, "metadata", {})
    if "multi_output" in metadata:
        return bool(metadata["multi_output"])
    try:
        return int(getattr(bundle.model, "num_outputs", 1)) > 1
    except (TypeError, ValueError):
        return False


def _ref_point_tensor(
    *,
    bundle: ModelBundle,
    data_context: DataContext,
    kwargs: dict[str, Any],
) -> torch.Tensor:
    raw = kwargs.pop("ref_point", None)
    if raw is None:
        raw = data_context.ref_point
    if raw is None:
        raise ValueError(
            "MF-HVKG requires a multi-objective ref_point. Provide DataContext.ref_point "
            "or AcquisitionConfig.acqf_kwargs['ref_point']."
        )
    train_Y = _reference_train_Y(bundle)
    ref_point = torch.as_tensor(raw, dtype=train_Y.dtype, device=train_Y.device)
    if ref_point.ndim != 1:
        raise ValueError("MF-HVKG ref_point must be a one-dimensional tensor.")
    num_outputs = int(getattr(bundle.model, "num_outputs", ref_point.numel()))
    if ref_point.numel() != num_outputs:
        raise ValueError(
            f"MF-HVKG ref_point must contain {num_outputs} values; "
            f"received {ref_point.numel()}."
        )
    if not bool(torch.isfinite(ref_point).all()):
        raise ValueError("MF-HVKG ref_point must contain only finite values.")
    return ref_point


def _validate_targets(
    *,
    targets: Mapping[int, float],
    bounds: torch.Tensor,
    d: int,
) -> None:
    for index, value in targets.items():
        if index < 0 or index >= d:
            raise ValueError(
                f"target fidelity feature {index} is outside model dimension d={d}."
            )
        lower = float(bounds[0, index])
        upper = float(bounds[1, index])
        if value < lower or value > upper:
            raise ValueError(
                f"target fidelity {value} is outside bounds for feature {index}: "
                f"[{lower}, {upper}]."
            )


def _current_hypervolume_value(
    *,
    bundle: ModelBundle,
    bounds: torch.Tensor,
    ref_point: torch.Tensor,
    targets: Mapping[int, float],
    project: Any,
    objective: Any,
    inner_sampler: Any,
    num_pareto: int,
    num_restarts: int,
    raw_samples: int,
    use_posterior_mean: bool,
    log: bool,
) -> torch.Tensor:
    """Optimize the terminal target-fidelity hypervolume for current data."""

    value_function = _get_hv_value_function(
        model=bundle.model,
        ref_point=ref_point,
        objective=objective,
        sampler=inner_sampler,
        project=project,
        use_posterior_mean=use_posterior_mean,
        log=log,
    )
    categorical = _categorical_assignments(bundle)
    if categorical:
        fixed_features_list: list[dict[int, float]] = []
        for assignment in categorical:
            item = dict(assignment)
            for index, value in targets.items():
                if index in item and item[index] != value:
                    raise ValueError(
                        "A categorical dimension overlaps the target fidelity dimension."
                    )
                item[index] = value
            fixed_features_list.append(item)
        _, values = optimize_acqf_mixed(
            acq_function=value_function,
            bounds=bounds,
            q=num_pareto,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            fixed_features_list=fixed_features_list,
            sequential=False,
            return_best_only=False,
        )
    else:
        _, values = optimize_acqf(
            acq_function=value_function,
            bounds=bounds,
            q=num_pareto,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            fixed_features=dict(targets),
            sequential=False,
            return_best_only=False,
        )
    return values.max().detach()


def _resolve_objective(
    *,
    bundle: ModelBundle,
    config: AcquisitionConfig,
    data_context: DataContext,
    kwargs: dict[str, Any],
) -> Any:
    """Resolve explicit and configured objectives through the shared API factory."""

    objective = kwargs.pop("objective", config.objective)
    if objective is not None:
        return objective
    if config.objective_factory is None and config.objective_config is None:
        return None

    from ..factory import build_objective

    return build_objective(bundle=bundle, config=config, data_context=data_context)


def build_multifidelity_hvkg_acquisition(
    *,
    bundle: ModelBundle,
    config: AcquisitionConfig,
    data_context: DataContext,
) -> qMultiFidelityHypervolumeKnowledgeGradient:
    """Build BoTorch MF-HVKG for independent or correlated multi-output MF."""

    if not is_multifidelity_hvkg_name(config.name):
        raise ValueError(f"Unsupported MF-HVKG acquisition name: {config.name!r}.")
    if not _bundle_is_multi_output(bundle):
        raise ValueError("MF-HVKG requires a multi-output multi-fidelity model.")

    model = bundle.model
    d = _model_dimension(bundle)
    bounds = _bounds_tensor(bundle, data_context)
    kwargs = dict(config.acqf_kwargs)
    ref_point = _ref_point_tensor(bundle=bundle, data_context=data_context, kwargs=kwargs)

    target_fidelity = kwargs.pop("target_fidelity", None)
    target_fidelities = kwargs.pop("target_fidelities", None)
    targets = _target_fidelities(
        model,
        target_fidelity=target_fidelity,
        target_fidelities=target_fidelities,
    )
    _validate_targets(targets=targets, bounds=bounds, d=d)
    project = _projector(targets=targets, d=d)

    X_pending = kwargs.pop("X_pending", data_context.X_pending)
    objective = _resolve_objective(
        bundle=bundle,
        config=config,
        data_context=data_context,
        kwargs=kwargs,
    )
    for field_name in ("X_evaluation_mask", "X_pending_evaluation_mask"):
        if field_name not in kwargs and field_name in data_context.extra:
            kwargs[field_name] = data_context.extra[field_name]

    num_fantasies = int(kwargs.pop("num_fantasies", 8))
    num_pareto = int(kwargs.pop("num_pareto", 10))
    if num_fantasies < 1 or num_pareto < 1:
        raise ValueError("MF-HVKG num_fantasies and num_pareto must be positive.")
    use_posterior_mean = bool(kwargs.pop("use_posterior_mean", True))
    log = bool(kwargs.pop("log", False))
    inner_sampler = kwargs.pop("inner_sampler", None)

    cost_model, cost_utility = _resolve_cost_aware_utility(model, kwargs, d=d)
    kwargs.pop("cost_aware_utility", None)

    current_value = kwargs.pop("current_value", None)
    current_value_num_restarts = int(kwargs.pop("current_value_num_restarts", 10))
    current_value_raw_samples = int(kwargs.pop("current_value_raw_samples", 256))
    if current_value is None:
        current_value = _current_hypervolume_value(
            bundle=bundle,
            bounds=bounds,
            ref_point=ref_point,
            targets=targets,
            project=project,
            objective=objective,
            inner_sampler=inner_sampler,
            num_pareto=num_pareto,
            num_restarts=current_value_num_restarts,
            raw_samples=current_value_raw_samples,
            use_posterior_mean=use_posterior_mean,
            log=log,
        )
    else:
        current_value = torch.as_tensor(
            current_value,
            dtype=ref_point.dtype,
            device=ref_point.device,
        )

    acqf = qMultiFidelityHypervolumeKnowledgeGradient(
        model=model,
        ref_point=ref_point,
        target_fidelities=dict(targets),
        num_fantasies=num_fantasies,
        num_pareto=num_pareto,
        objective=objective,
        inner_sampler=inner_sampler,
        X_pending=X_pending,
        current_value=current_value,
        cost_aware_utility=cost_utility,
        project=project,
        use_posterior_mean=use_posterior_mean,
        log=log,
        **kwargs,
    )
    object.__setattr__(acqf, "_bochan_multifidelity_kind", "mfhvkg")
    return _attach_cost_metadata(acqf, cost_model=cost_model, utility=cost_utility)


__all__ = [
    "build_multifidelity_hvkg_acquisition",
    "is_multifidelity_hvkg_name",
]
