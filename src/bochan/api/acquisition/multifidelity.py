"""Factories for generic multi-fidelity acquisition functions.

This module keeps multi-fidelity acquisition construction in the API layer so
ordinary BoTorch acquisition functions remain available for multi-fidelity
surrogates. The helpers below add the target-fidelity and optional cost-aware
plumbing required by MFKG and MF-MES.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

import torch
from botorch.acquisition import PosteriorMean
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.max_value_entropy_search import qMultiFidelityMaxValueEntropy
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.optim import optimize_acqf, optimize_acqf_mixed

from bochan.models.multifidelity.cost import (
    FidelityCostConfig,
    build_fidelity_cost_utility,
)

from ..configs import AcquisitionConfig, DataContext, ModelBundle


_MFKG_NAMES = {
    "mfkg",
    "qmfkg",
    "multifidelityknowledgegradient",
    "qmultifidelityknowledgegradient",
}
_MFMES_NAMES = {
    "mfmes",
    "qmfmes",
    "multifidelitymaxvalueentropy",
    "qmultifidelitymaxvalueentropy",
}


def _normalize_name(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def is_multifidelity_acquisition_name(name: Any) -> bool:
    """Return whether ``name`` is handled by the MF acquisition factory."""

    normalized = _normalize_name(name)
    return normalized in _MFKG_NAMES or normalized in _MFMES_NAMES


def _target_fidelities(model: Any) -> dict[int, float]:
    targets = getattr(model, "target_fidelities", None)
    if targets is None:
        metadata = getattr(model, "fidelity_metadata", None)
        if callable(metadata):
            metadata = metadata()
        if isinstance(metadata, Mapping):
            targets = metadata.get("target_fidelities")
    if not isinstance(targets, Mapping) or not targets:
        raise ValueError(
            "Multi-fidelity acquisition functions require model target_fidelities. "
            "Configure target_fidelities on ModelConfig.model_kwargs."
        )
    return {int(index): float(value) for index, value in targets.items()}


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


def _model_dimension(bundle: ModelBundle) -> int:
    train_X = torch.as_tensor(bundle.train_X)
    if train_X.ndim != 2:
        raise ValueError("Multi-fidelity acquisition construction requires 2-D train_X.")
    return int(train_X.shape[-1])


def _bounds_tensor(bundle: ModelBundle, context: DataContext) -> torch.Tensor:
    bounds = context.bounds
    if bounds is None:
        bounds = getattr(bundle, "bounds", None)
    if bounds is None:
        raise ValueError(
            "Multi-fidelity acquisition functions require bounds in DataContext."
        )
    train_X = torch.as_tensor(bundle.train_X)
    tensor = torch.as_tensor(bounds, dtype=train_X.dtype, device=train_X.device)
    if tensor.shape != (2, train_X.shape[-1]):
        raise ValueError(
            "Multi-fidelity acquisition bounds must have shape (2, d); "
            f"received {tuple(tensor.shape)} for d={train_X.shape[-1]}."
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("Multi-fidelity acquisition bounds must be finite.")
    if bool((tensor[0] > tensor[1]).any()):
        raise ValueError("Multi-fidelity acquisition bounds must satisfy lower <= upper.")
    return tensor


def _projector(*, targets: Mapping[int, float], d: int):
    return partial(project_to_target_fidelity, target_fidelities=dict(targets), d=d)


def _categorical_assignments(bundle: ModelBundle) -> list[dict[int, float]] | None:
    model = bundle.model
    cat_dims = tuple(int(index) for index in (getattr(model, "cat_dims", None) or ()))
    if not cat_dims:
        return None

    train_X = torch.as_tensor(bundle.train_X)
    combos = torch.unique(train_X[:, list(cat_dims)], dim=0)
    return [
        {index: float(value) for index, value in zip(cat_dims, row.tolist(), strict=True)}
        for row in combos
    ]


def _sign_posterior_transform(bundle: ModelBundle, *, maximize: bool) -> Any | None:
    if maximize:
        return None
    train_Y = torch.as_tensor(bundle.train_Y)
    if train_Y.ndim < 2 or int(train_Y.shape[-1]) != 1:
        raise ValueError("maximize=False is currently supported only for single-output MF models.")
    weights = train_Y.new_tensor([-1.0])
    return ScalarizedPosteriorTransform(weights=weights)


def _current_value(
    *,
    bundle: ModelBundle,
    bounds: torch.Tensor,
    targets: Mapping[int, float],
    posterior_transform: Any | None,
    num_restarts: int,
    raw_samples: int,
) -> torch.Tensor:
    """Compute the current terminal value at the configured target fidelity."""

    posterior_mean = PosteriorMean(
        model=bundle.model,
        posterior_transform=posterior_transform,
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
        _, value = optimize_acqf_mixed(
            acq_function=posterior_mean,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            fixed_features_list=fixed_features_list,
        )
    else:
        _, value = optimize_acqf(
            acq_function=posterior_mean,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            fixed_features=dict(targets),
        )
    return value.max().detach()


def _candidate_set(
    *,
    bundle: ModelBundle,
    bounds: torch.Tensor,
    targets: Mapping[int, float],
    size: int,
) -> torch.Tensor:
    """Generate an MF-MES discretization set and project it to target fidelity."""

    if size < 2:
        raise ValueError("candidate_set_size must be at least 2.")
    train_X = torch.as_tensor(bundle.train_X, dtype=bounds.dtype, device=bounds.device)
    n, d = train_X.shape
    random = torch.rand(size, d, dtype=bounds.dtype, device=bounds.device)
    candidates = bounds[0] + (bounds[1] - bounds[0]) * random

    cat_dims = tuple(int(index) for index in (getattr(bundle.model, "cat_dims", None) or ()))
    for index in cat_dims:
        values = torch.unique(train_X[:, index])
        draw = torch.randint(values.numel(), (size,), device=bounds.device)
        candidates[:, index] = values[draw]

    keep = min(n, size)
    if keep:
        candidates[:keep] = train_X[:keep]

    return project_to_target_fidelity(
        candidates,
        target_fidelities=dict(targets),
        d=d,
    )


def _resolve_cost_aware_utility(model: Any, kwargs: dict[str, Any]) -> tuple[Any | None, Any | None]:
    """Resolve optional Phase 51 cost configuration from acquisition kwargs."""

    explicit_utility = kwargs.get("cost_aware_utility")
    raw_config = kwargs.pop("cost_config", None)
    if explicit_utility is not None and raw_config is not None:
        raise ValueError("Provide either cost_aware_utility or cost_config, not both.")
    if raw_config is None:
        return None, explicit_utility
    if isinstance(raw_config, Mapping):
        raw_config = FidelityCostConfig(**dict(raw_config))
    if not isinstance(raw_config, FidelityCostConfig):
        raise TypeError("cost_config must be FidelityCostConfig or a mapping of its fields.")
    features = _fidelity_features(model)
    if not features:
        raise ValueError("cost_config requires model fidelity_features metadata.")
    cost_model, utility = build_fidelity_cost_utility(
        raw_config,
        fidelity_features=features,
    )
    kwargs["cost_aware_utility"] = utility
    return cost_model, utility


def _attach_cost_metadata(acqf: Any, *, cost_model: Any | None, utility: Any | None) -> Any:
    if cost_model is not None:
        object.__setattr__(acqf, "_bochan_cost_model", cost_model)
    if utility is not None:
        object.__setattr__(acqf, "_bochan_cost_aware_utility", utility)
    return acqf


def build_multifidelity_acquisition(
    *,
    bundle: ModelBundle,
    config: AcquisitionConfig,
    data_context: DataContext,
) -> Any:
    """Build MFKG or MF-MES from the public :class:`AcquisitionConfig` API."""

    name = _normalize_name(config.name)
    if name not in _MFKG_NAMES | _MFMES_NAMES:
        raise ValueError(f"Unsupported multi-fidelity acquisition name: {config.name!r}.")

    model = bundle.model
    targets = _target_fidelities(model)
    d = _model_dimension(bundle)
    bounds = _bounds_tensor(bundle, data_context)
    project = _projector(targets=targets, d=d)
    kwargs = dict(config.acqf_kwargs)
    X_pending = kwargs.pop("X_pending", data_context.X_pending)
    cost_model, cost_utility = _resolve_cost_aware_utility(model, kwargs)

    if name in _MFKG_NAMES:
        maximize = bool(kwargs.pop("maximize", True))
        posterior_transform = kwargs.pop("posterior_transform", None)
        if posterior_transform is None:
            posterior_transform = _sign_posterior_transform(bundle, maximize=maximize)
        current_value = kwargs.pop("current_value", None)
        num_restarts = int(kwargs.pop("current_value_num_restarts", 10))
        raw_samples = int(kwargs.pop("current_value_raw_samples", 256))
        if current_value is None:
            current_value = _current_value(
                bundle=bundle,
                bounds=bounds,
                targets=targets,
                posterior_transform=posterior_transform,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
            )
        acqf = qMultiFidelityKnowledgeGradient(
            model=model,
            current_value=current_value,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            project=project,
            **kwargs,
        )
        return _attach_cost_metadata(acqf, cost_model=cost_model, utility=cost_utility)

    candidate_set = kwargs.pop("candidate_set", None)
    candidate_set_size = int(kwargs.pop("candidate_set_size", 1024))
    if candidate_set is None:
        candidate_set = _candidate_set(
            bundle=bundle,
            bounds=bounds,
            targets=targets,
            size=candidate_set_size,
        )
    else:
        candidate_set = torch.as_tensor(
            candidate_set,
            dtype=bounds.dtype,
            device=bounds.device,
        )
        candidate_set = project(candidate_set)

    acqf = qMultiFidelityMaxValueEntropy(
        model=model,
        candidate_set=candidate_set,
        X_pending=X_pending,
        project=project,
        **kwargs,
    )
    return _attach_cost_metadata(acqf, cost_model=cost_model, utility=cost_utility)


__all__ = [
    "build_multifidelity_acquisition",
    "is_multifidelity_acquisition_name",
]
