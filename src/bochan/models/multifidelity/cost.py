"""Cost models and cost-aware utilities for multi-fidelity BO."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.models.cost import AffineFidelityCostModel

FidelityCostKind = Literal["affine"]


@dataclass(frozen=True)
class FidelityCostConfig:
    """Configuration for a known fidelity-dependent evaluation cost.

    ``fixed_cost`` represents the fidelity-independent portion of one
    evaluation. ``fidelity_weights`` defines the linear contribution of each
    fidelity feature to cost. Negative feature indices are accepted when the
    model dimension is available, matching ``FidelitySpec`` / ``ModelConfig``.
    """

    kind: FidelityCostKind | str = "affine"
    fixed_cost: float = 1.0
    fidelity_weights: Mapping[int, float] | None = None
    min_cost: float = 1e-2

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind != "affine":
            raise ValueError("Gaussian Multi-Fidelity v1 supports FidelityCostConfig.kind='affine' only.")
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
            object.__setattr__(self, "fidelity_weights", normalized)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "fixed_cost", fixed_cost)
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


def build_fidelity_cost_utility(
    config: FidelityCostConfig,
    *,
    fidelity_features: tuple[int, ...],
    d: int | None = None,
) -> tuple[Any, InverseCostWeightedUtility]:
    """Build a BoTorch affine fidelity-cost model and inverse-cost utility."""

    if not isinstance(config, FidelityCostConfig):
        raise TypeError("cost_config must be a FidelityCostConfig instance.")
    weights = _resolved_weights(
        config,
        fidelity_features=fidelity_features,
        d=d,
    )
    cost_model = AffineFidelityCostModel(
        fidelity_weights=weights,
        fixed_cost=config.fixed_cost,
    )
    utility = InverseCostWeightedUtility(
        cost_model=cost_model,
        min_cost=config.min_cost,
    )
    return cost_model, utility


__all__ = [
    "FidelityCostConfig",
    "FidelityCostKind",
    "build_fidelity_cost_utility",
]
