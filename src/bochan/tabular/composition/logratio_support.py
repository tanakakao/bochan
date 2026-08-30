"""Acquisition-aware composition support search through raw fraction decisions.

The fitted surrogate may use CLR, ALR, or ILR coordinates, but element support
must be selected in raw fraction space. This module adapts the existing generic
best-subset optimizer by replacing the composition coordinate block with one raw
fraction per element during candidate optimization and wrapping the acquisition
function with a differentiable raw-to-model transform.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor, nn

from bochan.api import (
    OptimizeConfig,
    resolve_optimizer_from_cat_dims,
    uses_mixed_fixed_features,
)
from bochan.tabular.data import resolve_optimize_config_columns

from .raw_bridge import CompositionRawDecisionBridge
from .support import resolve_composition_best_subset

_LOG_RATIO_REPRESENTATIONS = {"clr", "alr", "ilr"}


def is_logratio_best_subset_site(config: Mapping[str, Any]) -> bool:
    """Return whether one site requests raw-space best-subset over log ratios."""

    return (
        str(config.get("support_selection", "repair")).lower() == "best_subset"
        and str(config.get("representation", "fractions")).lower()
        in _LOG_RATIO_REPRESENTATIONS
    )


def resolve_logratio_best_subset_site(
    sites: Mapping[str, Mapping[str, Any]],
) -> tuple[str, Mapping[str, Any]] | None:
    """Return the single log-ratio best-subset site, if configured."""

    selected = [
        (str(name), config)
        for name, config in sites.items()
        if is_logratio_best_subset_site(config)
    ]
    if not selected:
        return None
    all_best_subset = [
        str(name)
        for name, config in sites.items()
        if str(config.get("support_selection", "repair")).lower() == "best_subset"
    ]
    if len(all_best_subset) > 1:
        raise ValueError(
            "Composition best_subset currently supports one composition site per "
            "candidate optimization."
        )
    return selected[0]


def _map_index(index: Any, bridge: CompositionRawDecisionBridge) -> Any:
    if isinstance(index, str):
        if index in bridge.coordinate_names:
            raise ValueError(
                "Direct constraints or fixed values on CLR/ALR/ILR coordinates cannot "
                "be combined with raw-space composition best_subset. Constrain raw "
                "elements instead."
            )
        return index
    resolved = int(index)
    try:
        return bridge.process_index_map[resolved]
    except KeyError as exc:
        raise ValueError(
            "Direct constraints or fixed values on CLR/ALR/ILR coordinate indices "
            "cannot be combined with raw-space composition best_subset."
        ) from exc


def _map_indices(indices: Any, bridge: CompositionRawDecisionBridge) -> Any:
    if indices is None:
        return None
    if isinstance(indices, str):
        return _map_index(indices, bridge)
    if isinstance(indices, int):
        return [_map_index(indices, bridge)]
    if torch.is_tensor(indices):
        values = indices.detach().cpu().reshape(-1).tolist()
        mapped = [_map_index(int(value), bridge) for value in values]
        return torch.as_tensor(mapped, dtype=indices.dtype, device=indices.device)
    return [_map_index(value, bridge) for value in indices]


def _map_fixed_features(
    values: Mapping[Any, Any] | None,
    bridge: CompositionRawDecisionBridge,
) -> dict[Any, float] | None:
    if not values:
        return None
    return {_map_index(key, bridge): float(value) for key, value in values.items()}


def _map_fixed_features_list(
    values: Sequence[Mapping[Any, Any]] | None,
    bridge: CompositionRawDecisionBridge,
) -> list[dict[Any, float]] | None:
    if values is None:
        return None
    return [
        {_map_index(key, bridge): float(value) for key, value in item.items()}
        for item in values
    ]


def _map_constraints(
    constraints: Sequence[tuple[Any, Any, Any]] | None,
    bridge: CompositionRawDecisionBridge,
) -> list[tuple[Any, Any, Any]] | None:
    if constraints is None:
        return None
    return [
        (_map_indices(indices, bridge), coefficients, rhs)
        for indices, coefficients, rhs in constraints
    ]


def _raw_duplicate_tolerances(
    config: OptimizeConfig,
    bridge: CompositionRawDecisionBridge,
) -> tuple[float, ...] | None:
    tolerances = getattr(config, "duplicate_tolerances", None)
    if tolerances is None:
        return None
    values = tuple(float(value) for value in tolerances)
    if len(values) != bridge.model_dim:
        raise ValueError(
            "duplicate_tolerances width must match the fitted model feature dimension."
        )
    fraction_tolerance = max(
        float(getattr(config, "duplicate_tolerance", 1e-10)),
        1e-12,
    )
    return (
        values[: bridge.coordinate_start]
        + (fraction_tolerance,) * bridge.fraction_width
        + values[bridge.coordinate_stop :]
    )


def _remap_optimize_config(
    config: OptimizeConfig,
    bridge: CompositionRawDecisionBridge,
    raw_bounds: Tensor,
) -> OptimizeConfig:
    """Map model-space process indices to the expanded raw decision layout."""

    if config.post_processing_func is not None:
        raise ValueError(
            "Custom model-space post_processing_func is not supported with raw-space "
            "composition best_subset. Use CandidateRepairConfig/process constraints "
            "instead."
        )
    if getattr(config, "final_candidate_postprocess", None) is not None:
        raise ValueError(
            "final_candidate_postprocess is defined in fitted model coordinates and "
            "cannot be applied during raw-space composition best_subset."
        )

    repair = config.repair_config
    if repair is not None:
        numeric_indices = _map_indices(repair.numeric_indices, bridge)
        repair = replace(
            repair,
            bounds=raw_bounds,
            numeric_indices=numeric_indices,
            comp_idx=_map_indices(repair.comp_idx, bridge),
            equality_constraints=_map_constraints(
                repair.equality_constraints,
                bridge,
            ),
            inequality_constraints=_map_constraints(
                repair.inequality_constraints,
                bridge,
            ),
            fixed_features=_map_fixed_features(repair.fixed_features, bridge),
        )

    replacements: dict[str, Any] = {
        "repair_config": repair,
        "fixed_features": _map_fixed_features(config.fixed_features, bridge),
        "fixed_features_list": _map_fixed_features_list(
            config.fixed_features_list,
            bridge,
        ),
        "equality_constraints": _map_constraints(
            config.equality_constraints,
            bridge,
        ),
        "inequality_constraints": _map_constraints(
            config.inequality_constraints,
            bridge,
        ),
    }
    if hasattr(config, "duplicate_tolerances"):
        replacements["duplicate_tolerances"] = _raw_duplicate_tolerances(
            config,
            bridge,
        )
    return replace(config, **replacements)


def _raw_fixed_features_list_from_training(
    train_x: Tensor | None,
    cat_dims: Sequence[int],
) -> list[dict[int, float]] | None:
    if train_x is None or not cat_dims:
        return None
    values = train_x[..., list(cat_dims)]
    if values.ndim > 2:
        values = values.reshape(-1, len(cat_dims))
    unique = torch.unique(values, dim=0)
    if unique.numel() == 0:
        return None
    return [
        {
            int(index): float(value)
            for index, value in zip(cat_dims, row, strict=True)
        }
        for row in unique.detach().cpu().tolist()
    ]


def prepare_logratio_best_subset_config(
    opt_config: OptimizeConfig,
    *,
    site_name: str,
    site_config: Mapping[str, Any],
    transformer: Any,
    model_feature_names: Sequence[Any],
    model_bounds: Tensor,
    dtype: Any = None,
    device: Any = None,
    model_cat_dims: Sequence[int] | None = None,
    train_x: Tensor | None = None,
) -> tuple[CompositionRawDecisionBridge, OptimizeConfig, Tensor]:
    """Build the raw decision bridge, bounds, and resolved best-subset config."""

    if not is_logratio_best_subset_site(site_config):
        raise ValueError(
            f"Composition site {site_name!r} is not a CLR/ALR/ILR best_subset site."
        )
    if site_config.get("variable_total"):
        raise ValueError(
            "Log-ratio composition best_subset currently requires fixed total."
        )
    if site_config.get("steps"):
        raise ValueError(
            "Log-ratio composition best_subset currently requires continuous fractions."
        )

    bridge = CompositionRawDecisionBridge.from_transformer(
        transformer,
        model_feature_names,
    )
    raw_bounds = bridge.decision_bounds(
        model_bounds,
        component_bounds=site_config.get("bounds") or {},
        total=float(site_config.get("total", 1.0)),
    )
    raw_named_config = _remap_optimize_config(opt_config, bridge, raw_bounds)

    # Reuse the fraction-space resolver verbatim. Only its decision feature
    # layout is synthetic; the fitted surrogate representation is unchanged.
    raw_site = dict(site_config)
    raw_site["representation"] = "fractions"
    raw_named_config = resolve_composition_best_subset(
        raw_named_config,
        composition_sites={site_name: raw_site},
        composition_transformers={site_name: transformer},
        feature_names=bridge.decision_feature_names,
    )
    raw_config = resolve_optimize_config_columns(
        raw_named_config,
        bridge.decision_feature_names,
        dtype=dtype,
        device=device,
    )

    raw_cat_dims = [
        bridge.process_index_map[int(index)] for index in (model_cat_dims or ())
    ]
    raw_config = resolve_optimizer_from_cat_dims(
        opt_config=raw_config,
        cat_dims=raw_cat_dims,
    )
    if (
        uses_mixed_fixed_features(raw_config.optimizer)
        and raw_config.fixed_features_list is None
    ):
        raw_train_x = (
            bridge.model_to_decision(train_x) if train_x is not None else None
        )
        inferred = _raw_fixed_features_list_from_training(
            raw_train_x,
            raw_cat_dims,
        )
        if inferred:
            raw_config = replace(raw_config, fixed_features_list=inferred)

    return bridge, raw_config, raw_bounds


class RawDecisionAcquisition(nn.Module):
    """Evaluate a fitted model-space acquisition on raw decision candidates."""

    def __init__(
        self,
        base_acqf: Any,
        bridge: CompositionRawDecisionBridge,
    ) -> None:
        super().__init__()
        self.base_acqf = base_acqf
        self.bridge = bridge

    @property
    def model(self) -> Any:
        return getattr(self.base_acqf, "model", None)

    @property
    def X_pending(self) -> Tensor | None:
        """Expose pending points in raw decision coordinates for sequential BO."""

        pending = getattr(self.base_acqf, "X_pending", None)
        if pending is None:
            return None
        return self.bridge.model_to_decision(pending)

    def set_X_pending(self, X_pending: Tensor | None = None) -> RawDecisionAcquisition:
        """Map sequential raw pending points back to fitted model coordinates."""

        setter = getattr(self.base_acqf, "set_X_pending", None)
        if not callable(setter):
            if X_pending is not None:
                raise AttributeError(
                    "The wrapped acquisition does not support pending points."
                )
            return self
        model_pending = (
            None
            if X_pending is None
            else self.bridge.decision_to_model(X_pending)
        )
        setter(model_pending)
        return self

    def forward(self, values: Tensor) -> Tensor:
        return self.base_acqf(self.bridge.decision_to_model(values))


@dataclass(frozen=True)
class LogRatioBestSubsetResult:
    """Raw and fitted-space results from one best-subset optimization."""

    candidates: Tensor
    raw_candidates: Tensor
    acq_value: Any
    raw_opt_config: OptimizeConfig
    bridge: CompositionRawDecisionBridge


def _reject_one_shot_acquisition(acqf: Any) -> None:
    """Reject one-shot acquisitions until their augmented variables are bridged."""

    try:
        from botorch.acquisition.acquisition import OneShotAcquisitionFunction
    except ImportError:
        return
    if isinstance(acqf, OneShotAcquisitionFunction):
        raise NotImplementedError(
            "Raw-space composition best_subset does not yet support one-shot "
            "acquisition functions such as KG. Use EI/NEI/UCB/EHVI/NEHVI or "
            "another acquisition without augmented one-shot variables."
        )


def optimize_logratio_best_subset(
    base_acqf: Any,
    opt_config: OptimizeConfig,
    *,
    site_name: str,
    site_config: Mapping[str, Any],
    transformer: Any,
    model_feature_names: Sequence[Any],
    model_bounds: Tensor,
    dtype: Any = None,
    device: Any = None,
    model_cat_dims: Sequence[int] | None = None,
    train_x: Tensor | None = None,
    optimize_fn: Callable[..., tuple[Any, Any]] | None = None,
) -> LogRatioBestSubsetResult:
    """Optimize element support in raw fractions and return model-space candidates."""

    _reject_one_shot_acquisition(base_acqf)
    bridge, raw_config, raw_bounds = prepare_logratio_best_subset_config(
        opt_config,
        site_name=site_name,
        site_config=site_config,
        transformer=transformer,
        model_feature_names=model_feature_names,
        model_bounds=model_bounds,
        dtype=dtype,
        device=device,
        model_cat_dims=model_cat_dims,
        train_x=train_x,
    )
    wrapped = RawDecisionAcquisition(base_acqf, bridge)
    if optimize_fn is None:
        from bochan.api.optimizer.service import optimize_candidates as optimize_fn

    raw_candidates, _raw_value = optimize_fn(
        acqf=wrapped,
        bounds=raw_bounds,
        config=raw_config,
    )
    model_candidates = bridge.decision_to_model(raw_candidates)
    final_value = base_acqf(model_candidates)
    if hasattr(final_value, "detach"):
        final_value = final_value.detach()
    return LogRatioBestSubsetResult(
        candidates=model_candidates,
        raw_candidates=raw_candidates,
        acq_value=final_value,
        raw_opt_config=raw_config,
        bridge=bridge,
    )


__all__ = [
    "LogRatioBestSubsetResult",
    "RawDecisionAcquisition",
    "is_logratio_best_subset_site",
    "optimize_logratio_best_subset",
    "prepare_logratio_best_subset_config",
    "resolve_logratio_best_subset_site",
]
