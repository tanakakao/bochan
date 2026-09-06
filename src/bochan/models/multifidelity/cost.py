"""Cost models and cost-aware utilities for multi-fidelity BO."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.objective import GenericMCObjective
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.deterministic import GenericDeterministicModel

FidelityCostKind = Literal["affine", "fixed", "callable"]
FidelityCostCallable = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class FidelityCostConfig:
    """Configuration for a known evaluation cost.

    ``kind='affine'`` preserves the historical bochan contract: ``fixed_cost``
    is the fidelity-independent intercept and ``fidelity_weights`` defines the
    linear fidelity contribution. ``kind='fixed'`` treats ``fixed_cost`` as a
    constant cost for every evaluation. ``kind='callable'`` accepts a Python
    callable mapping candidate ``X`` to cost and is intentionally a Python-API
    feature because callables cannot be represented by JSON transports.

    ``min_cost`` is applied through the cost objective so every public mode
    remains strictly positive for inverse-cost weighting.
    """

    kind: FidelityCostKind | str = "affine"
    fixed_cost: float = 1.0
    fidelity_weights: Mapping[int, float] | None = None
    min_cost: float = 1e-2
    cost_callable: FidelityCostCallable | None = None

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind not in {"affine", "fixed", "callable"}:
            raise ValueError(
                "FidelityCostConfig.kind must be 'affine', 'fixed', or 'callable'."
            )
        fixed_cost = float(self.fixed_cost)
        min_cost = float(self.min_cost)
        if not math.isfinite(fixed_cost) or fixed_cost < 0:
            raise ValueError("fixed_cost must be finite and non-negative.")
        if not math.isfinite(min_cost) or min_cost <= 0:
            raise ValueError("min_cost must be finite and positive.")

        weights = self.fidelity_weights
        if weights is not None:
            normalized = {int(index): float(value) for index, value in weights.items()}
            if not normalized:
                raise ValueError("fidelity_weights must not be empty when supplied.")
            if any(not math.isfinite(value) or value < 0 for value in normalized.values()):
                raise ValueError("fidelity_weights must contain finite non-negative values.")
            weights = normalized

        if kind == "affine":
            if self.cost_callable is not None:
                raise ValueError("cost_callable is only valid for kind='callable'.")
        elif kind == "fixed":
            if weights is not None:
                raise ValueError("fidelity_weights is only valid for kind='affine'.")
            if self.cost_callable is not None:
                raise ValueError("cost_callable is only valid for kind='callable'.")
        else:
            if weights is not None:
                raise ValueError("fidelity_weights is only valid for kind='affine'.")
            if self.cost_callable is None or not callable(self.cost_callable):
                raise ValueError("kind='callable' requires a callable cost_callable.")

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "fixed_cost", fixed_cost)
        object.__setattr__(self, "fidelity_weights", weights)
        object.__setattr__(self, "min_cost", min_cost)


def _resolve_index(index: int, *, d: int | None) -> int:
    index = int(index)
    if index >= 0:
        return index
    if d is None:
        raise ValueError(
            "Negative fidelity cost indices require the model dimension d."
        )
    resolved = d + index
    if resolved < 0 or resolved >= d:
        raise ValueError(
            f"Fidelity cost feature index {index} is out of range for d={d}."
        )
    return resolved


def _resolved_weights(
    config: FidelityCostConfig,
    *,
    fidelity_features: tuple[int, ...],
    d: int | None = None,
) -> dict[int, float]:
    features = tuple(int(index) for index in fidelity_features)
    if config.fidelity_weights is None:
        if len(features) != 1:
            raise ValueError(
                "Default fidelity cost weights require exactly one fidelity feature. "
                "Provide fidelity_weights explicitly for multiple fidelities."
            )
        return {features[0]: 1.0}

    weights: dict[int, float] = {}
    for raw_index, value in config.fidelity_weights.items():
        index = _resolve_index(int(raw_index), d=d)
        if index in weights:
            raise ValueError(
                "fidelity_weights contains duplicate indices after negative-index resolution: "
                f"feature {index}."
            )
        weights[index] = float(value)

    unknown = set(weights) - set(features)
    if unknown:
        raise ValueError(
            "fidelity_weights may reference only model fidelity features; "
            f"unknown indices: {sorted(unknown)}."
        )
    return weights


def _minimum_cost_objective(min_cost: float) -> GenericMCObjective:
    """Return a cost objective that guarantees strictly positive costs."""

    return GenericMCObjective(
        lambda samples, X=None: samples.squeeze(-1).clamp_min(min_cost)
    )


def _normalize_callable_cost(
    value: Any,
    *,
    X: torch.Tensor,
) -> torch.Tensor:
    """Normalize a known Python cost callable to ``batch x q x 1``."""

    cost = torch.as_tensor(value, dtype=X.dtype, device=X.device)
    target_shape = X.shape[:-1]
    if cost.ndim == 0:
        cost = cost.expand(target_shape).unsqueeze(-1)
    elif cost.shape == target_shape:
        cost = cost.unsqueeze(-1)
    elif cost.shape != (*target_shape, 1):
        raise ValueError(
            "cost_callable must return a scalar, batch_shape x q, or "
            "batch_shape x q x 1 tensor."
        )
    if not bool(torch.isfinite(cost).all()):
        raise ValueError("cost_callable must return only finite costs.")
    return cost


def build_fidelity_cost_utility(
    config: FidelityCostConfig,
    *,
    fidelity_features: tuple[int, ...],
    d: int | None = None,
) -> tuple[Any, InverseCostWeightedUtility]:
    """Build a known cost model and inverse-cost utility for MF acquisition."""

    if not isinstance(config, FidelityCostConfig):
        raise TypeError("cost_config must be a FidelityCostConfig instance.")

    if config.kind == "affine":
        weights = _resolved_weights(
            config,
            fidelity_features=fidelity_features,
            d=d,
        )
        cost_model: Any = AffineFidelityCostModel(
            fidelity_weights=weights,
            fixed_cost=config.fixed_cost,
        )
    elif config.kind == "fixed":
        fixed_cost = config.fixed_cost
        cost_model = GenericDeterministicModel(
            f=lambda X: X[..., :1] * 0.0 + fixed_cost,
            num_outputs=1,
        )
    else:
        cost_callable = config.cost_callable
        if cost_callable is None:  # guarded by the dataclass, defensive for typing
            raise ValueError("kind='callable' requires cost_callable.")

        def known_cost(X: torch.Tensor) -> torch.Tensor:
            return _normalize_callable_cost(cost_callable(X), X=X)

        cost_model = GenericDeterministicModel(f=known_cost, num_outputs=1)

    utility = InverseCostWeightedUtility(
        cost_model=cost_model,
        cost_objective=_minimum_cost_objective(config.min_cost),
    )
    return cost_model, utility


__all__ = [
    "FidelityCostCallable",
    "FidelityCostConfig",
    "FidelityCostKind",
    "build_fidelity_cost_utility",
]
