"""Multi-objective multi-fidelity optimization with MOMF."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from botorch.acquisition.multi_objective.multi_fidelity import MOMF
from botorch.acquisition.multi_objective.objective import GenericMCMultiOutputObjective
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)

from bochan.models.multifidelity.cost import (
    FidelityCostConfig,
    build_fidelity_cost_utility,
    evaluate_fidelity_cost_mean,
)

from ..configs import AcquisitionConfig, DataContext, ModelBundle
from .multifidelity import (
    _bounds_tensor,
    _bundle_tensors,
    _fidelity_features,
    _model_dimension,
    _reference_train_X,
    _target_fidelities,
)

_MOMF_NAMES = {
    "momf",
    "qmomf",
    "multiobjectivemultifidelity",
    "qmultiobjectivemultifidelity",
}


def _normalize_name(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def is_momf_name(name: Any) -> bool:
    """Return whether ``name`` identifies the MOMF strategy."""

    return _normalize_name(name) in _MOMF_NAMES


def _bundle_is_multi_output(bundle: ModelBundle) -> bool:
    """Infer multi-output capability from metadata or the concrete model."""

    if bool(bundle.metadata.get("multi_output", False)):
        return True
    try:
        return int(getattr(bundle.model, "num_outputs", 1)) > 1
    except (TypeError, ValueError):
        return False


def _training_outcomes(bundle: ModelBundle) -> torch.Tensor:
    """Return observed physical outcomes as an ``n x m`` tensor."""

    tensors = _bundle_tensors(bundle.train_Y, name="train_Y")
    reference = tensors[0]
    aligned = [tensor.to(dtype=reference.dtype, device=reference.device) for tensor in tensors]
    if len(aligned) == 1:
        train_Y = aligned[0]
        if train_Y.ndim == 1:
            train_Y = train_Y.unsqueeze(-1)
        return train_Y
    if any(tensor.ndim == 1 for tensor in aligned):
        aligned = [tensor.unsqueeze(-1) if tensor.ndim == 1 else tensor for tensor in aligned]
    n = int(aligned[0].shape[-2])
    if any(tensor.ndim != 2 or int(tensor.shape[-2]) != n for tensor in aligned):
        raise ValueError("MOMF requires aligned 2-D training outcomes across outputs.")
    return torch.cat(aligned, dim=-1)


def _default_fidelity_objective(
    *,
    feature: int,
    target: float,
    bounds: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a trust objective with value one at the target fidelity."""

    lower = bounds[0, feature]
    upper = bounds[1, feature]
    width = (upper - lower).abs()
    if not bool(width > 0):
        raise ValueError("MOMF fidelity bounds must have non-zero width.")

    def fidelity_objective(X: torch.Tensor) -> torch.Tensor:
        values = X[..., feature]
        return 1.0 - (values - target).abs() / width.to(values)

    return fidelity_objective


def _trust_values(
    objective: Callable[[torch.Tensor], torch.Tensor],
    X: torch.Tensor,
) -> torch.Tensor:
    values = torch.as_tensor(objective(X), dtype=X.dtype, device=X.device)
    if values.ndim == X.ndim and values.shape[-1] == 1:
        values = values.squeeze(-1)
    if values.shape != X.shape[:-1]:
        raise ValueError(
            "MOMF fidelity_objective must return shape batch_shape x q (or a trailing singleton)."
        )
    if not bool(torch.isfinite(values).all()):
        raise ValueError("MOMF fidelity_objective must return finite values.")
    return values


def _append_trust_to_samples(
    physical: torch.Tensor,
    *,
    X: torch.Tensor,
    fidelity_objective: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    trust = _trust_values(fidelity_objective, X)
    while trust.ndim < physical.ndim - 1:
        trust = trust.unsqueeze(0)
    trust = trust.expand(physical.shape[:-1])
    return torch.cat([physical, trust.unsqueeze(-1)], dim=-1)


def _resolve_physical_objective(
    *,
    bundle: ModelBundle,
    config: AcquisitionConfig,
    data_context: DataContext,
    kwargs: dict[str, Any],
) -> Any:
    objective = kwargs.pop("objective", config.objective)
    if objective is not None:
        return objective
    if config.objective_factory is None and config.objective_config is None:
        return None
    from .service import build_objective

    return build_objective(bundle=bundle, config=config, data_context=data_context)


def _physical_observations(
    train_Y: torch.Tensor,
    train_X: torch.Tensor,
    objective: Any,
) -> torch.Tensor:
    if objective is None:
        return train_Y
    transformed = objective(train_Y.unsqueeze(0), X=train_X)
    transformed = torch.as_tensor(transformed, dtype=train_Y.dtype, device=train_Y.device)
    if transformed.ndim == 3 and transformed.shape[0] == 1:
        transformed = transformed.squeeze(0)
    if transformed.ndim != 2 or transformed.shape[-2] != train_Y.shape[-2]:
        raise ValueError("MOMF objective must map observed outcomes to an n x m tensor.")
    return transformed


def _resolve_ref_point(
    *,
    raw: Any,
    num_physical_outputs: int,
    train_Y: torch.Tensor,
    fidelity_ref_point: float,
) -> torch.Tensor:
    if raw is None:
        raise ValueError("MOMF requires DataContext.ref_point or acqf_kwargs['ref_point'].")
    ref_point = torch.as_tensor(raw, dtype=train_Y.dtype, device=train_Y.device).reshape(-1)
    if ref_point.numel() == num_physical_outputs:
        ref_point = torch.cat([ref_point, ref_point.new_tensor([fidelity_ref_point])])
    elif ref_point.numel() != num_physical_outputs + 1:
        raise ValueError(
            "MOMF ref_point must contain one value per physical objective, optionally "
            "followed by the fidelity-objective reference value."
        )
    if not bool(torch.isfinite(ref_point).all()):
        raise ValueError("MOMF ref_point must contain only finite values.")
    return ref_point


def _resolve_cost_call(
    *,
    model: Any,
    d: int,
    kwargs: dict[str, Any],
) -> tuple[Callable[[torch.Tensor], torch.Tensor], Any | None]:
    explicit = kwargs.pop("cost_call", None)
    raw_config = kwargs.pop("cost_config", None)
    if explicit is not None and raw_config is not None:
        raise ValueError("Provide either MOMF cost_call or cost_config, not both.")
    if explicit is not None:
        if not callable(explicit):
            raise TypeError("MOMF cost_call must be callable.")
        return explicit, None

    config = (
        FidelityCostConfig()
        if raw_config is None
        else raw_config
        if isinstance(raw_config, FidelityCostConfig)
        else FidelityCostConfig(**dict(raw_config))
        if isinstance(raw_config, Mapping)
        else None
    )
    if config is None:
        raise TypeError("MOMF cost_config must be FidelityCostConfig or a mapping.")
    features = _fidelity_features(model)
    cost_model, utility = build_fidelity_cost_utility(
        config,
        fidelity_features=features,
        d=d,
    )

    def cost_call(X: torch.Tensor) -> torch.Tensor:
        return evaluate_fidelity_cost_mean(
            cost_model,
            utility,
            X,
            min_cost=config.min_cost,
        )

    return cost_call, cost_model


def build_momf_acquisition(
    *,
    bundle: ModelBundle,
    config: AcquisitionConfig,
    data_context: DataContext,
) -> MOMF:
    """Build BoTorch MOMF from independent or correlated multi-output MF."""

    if not is_momf_name(config.name):
        raise ValueError(f"Unsupported MOMF acquisition name: {config.name!r}.")
    if not _bundle_is_multi_output(bundle):
        raise ValueError("MOMF requires a multi-output multi-fidelity model.")

    model = bundle.model
    d = _model_dimension(bundle)
    bounds = _bounds_tensor(bundle, data_context)
    features = _fidelity_features(model)
    if len(features) != 1:
        raise ValueError("MOMF currently requires exactly one fidelity feature.")
    feature = features[0]
    targets = _target_fidelities(model)
    if set(targets) != {feature}:
        raise ValueError("MOMF requires one target value for its fidelity feature.")
    target = float(targets[feature])

    kwargs = dict(config.acqf_kwargs)
    physical_objective = _resolve_physical_objective(
        bundle=bundle,
        config=config,
        data_context=data_context,
        kwargs=kwargs,
    )
    fidelity_objective = kwargs.pop("fidelity_objective", None)
    if fidelity_objective is None:
        fidelity_objective = _default_fidelity_objective(
            feature=feature,
            target=target,
            bounds=bounds,
        )
    if not callable(fidelity_objective):
        raise TypeError("MOMF fidelity_objective must be callable.")

    train_X = _reference_train_X(bundle)
    train_Y = _training_outcomes(bundle)
    physical_train = _physical_observations(train_Y, train_X, physical_objective)
    if physical_train.shape[-1] < 2:
        raise ValueError("MOMF requires at least two physical objectives.")
    trust_train = _trust_values(fidelity_objective, train_X).unsqueeze(-1)
    augmented_train = torch.cat([physical_train, trust_train], dim=-1)

    fidelity_ref_point = float(kwargs.pop("fidelity_ref_point", 0.0))
    raw_ref_point = kwargs.pop("ref_point", data_context.ref_point)
    ref_point = _resolve_ref_point(
        raw=raw_ref_point,
        num_physical_outputs=int(physical_train.shape[-1]),
        train_Y=physical_train,
        fidelity_ref_point=fidelity_ref_point,
    )
    partitioning = kwargs.pop("partitioning", None)
    if partitioning is None:
        partitioning = FastNondominatedPartitioning(
            ref_point=ref_point,
            Y=augmented_train,
        )

    def augmented_objective(samples: torch.Tensor, X: torch.Tensor | None = None):
        if X is None:
            raise ValueError("MOMF augmented objective requires candidate X.")
        physical = samples if physical_objective is None else physical_objective(samples, X=X)
        return _append_trust_to_samples(
            physical,
            X=X,
            fidelity_objective=fidelity_objective,
        )

    objective = GenericMCMultiOutputObjective(augmented_objective)
    cost_call, cost_model = _resolve_cost_call(model=model, d=d, kwargs=kwargs)
    sampler = kwargs.pop("sampler", config.sampler)
    X_pending = kwargs.pop("X_pending", data_context.X_pending)
    constraints = kwargs.pop("constraints", data_context.constraints)
    eta = kwargs.pop("eta", 1e-3)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected MOMF acquisition option(s): {unexpected}.")

    acqf = MOMF(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        sampler=sampler,
        objective=objective,
        constraints=constraints,
        eta=eta,
        X_pending=X_pending,
        cost_call=cost_call,
    )
    object.__setattr__(acqf, "_bochan_multifidelity_kind", "momf")
    object.__setattr__(acqf, "_bochan_fidelity_objective", fidelity_objective)
    if cost_model is not None:
        object.__setattr__(acqf, "_bochan_cost_model", cost_model)
    return acqf


__all__ = ["build_momf_acquisition", "is_momf_name"]
