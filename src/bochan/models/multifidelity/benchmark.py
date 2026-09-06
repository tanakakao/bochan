"""Cost-normalized benchmark metrics for multi-fidelity optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from torch import Tensor


def _vector(value: Any, *, name: str) -> Tensor:
    tensor = torch.as_tensor(value)
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.get_default_dtype())
    if tensor.ndim == 2 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    if tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence.")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def cumulative_cost(costs: Any) -> Tensor:
    """Return cumulative evaluation cost, requiring strictly positive costs."""

    tensor = _vector(costs, name="costs")
    if not bool((tensor > 0).all()):
        raise ValueError("costs must be strictly positive.")
    return tensor.cumsum(dim=0)


def best_objective_trace(values: Any, *, maximize: bool = True) -> Tensor:
    """Return incumbent objective after every observation."""

    tensor = _vector(values, name="values")
    operation = torch.cummax if maximize else torch.cummin
    return operation(tensor, dim=0).values


def hypervolume_trace(Y: Any, *, ref_point: Any) -> Tensor:
    """Return dominated hypervolume after each prefix of multi-objective observations."""

    outcomes = torch.as_tensor(Y)
    if not outcomes.is_floating_point():
        outcomes = outcomes.to(dtype=torch.get_default_dtype())
    if outcomes.ndim != 2 or outcomes.shape[0] == 0 or outcomes.shape[1] < 2:
        raise ValueError("Y must have shape n x m with n >= 1 and m >= 2.")
    if not bool(torch.isfinite(outcomes).all()):
        raise ValueError("Y must contain only finite values.")
    reference = torch.as_tensor(ref_point, dtype=outcomes.dtype, device=outcomes.device).reshape(-1)
    if reference.numel() != outcomes.shape[-1] or not bool(torch.isfinite(reference).all()):
        raise ValueError("ref_point must contain one finite value per objective.")

    hypervolume = Hypervolume(ref_point=reference)
    return torch.stack(
        [hypervolume.compute(outcomes[: index + 1]) for index in range(outcomes.shape[0])]
    ).to(dtype=outcomes.dtype, device=outcomes.device)


def hypervolume_regret_trace(hypervolume: Any, *, reference_hypervolume: float) -> Tensor:
    """Return non-negative hypervolume regret to a known target/reference HV."""

    values = _vector(hypervolume, name="hypervolume")
    target = values.new_tensor(float(reference_hypervolume))
    if not bool(torch.isfinite(target)) or float(target) < 0:
        raise ValueError("reference_hypervolume must be finite and non-negative.")
    return (target - values).clamp_min(0.0)


@dataclass(frozen=True)
class CostNormalizedTrace:
    """One strategy's metric values aligned to cumulative evaluation cost."""

    strategy: str
    cumulative_cost: Tensor
    metric: Tensor
    metric_name: str

    def __post_init__(self) -> None:
        if not str(self.strategy).strip():
            raise ValueError("strategy must not be empty.")
        if not str(self.metric_name).strip():
            raise ValueError("metric_name must not be empty.")
        cost = _vector(self.cumulative_cost, name="cumulative_cost")
        metric = _vector(self.metric, name="metric")
        if cost.shape != metric.shape:
            raise ValueError("cumulative_cost and metric must have the same length.")
        if not bool((cost > 0).all()) or not bool((cost[1:] >= cost[:-1]).all()):
            raise ValueError("cumulative_cost must be positive and non-decreasing.")
        object.__setattr__(self, "strategy", str(self.strategy))
        object.__setattr__(self, "metric_name", str(self.metric_name))
        object.__setattr__(self, "cumulative_cost", cost)
        object.__setattr__(self, "metric", metric.to(dtype=cost.dtype, device=cost.device))


def single_objective_cost_trace(
    *,
    strategy: str,
    values: Any,
    costs: Any,
    maximize: bool = True,
) -> CostNormalizedTrace:
    """Build ``best objective vs cumulative cost`` for a BO strategy."""

    metric = best_objective_trace(values, maximize=maximize)
    cost = cumulative_cost(costs).to(dtype=metric.dtype, device=metric.device)
    if metric.shape != cost.shape:
        raise ValueError("values and costs must have the same length.")
    return CostNormalizedTrace(
        strategy=strategy,
        cumulative_cost=cost,
        metric=metric,
        metric_name="best_objective",
    )


def multi_objective_cost_trace(
    *,
    strategy: str,
    Y: Any,
    costs: Any,
    ref_point: Any,
) -> CostNormalizedTrace:
    """Build ``hypervolume vs cumulative cost`` for a BO strategy."""

    metric = hypervolume_trace(Y, ref_point=ref_point)
    cost = cumulative_cost(costs).to(dtype=metric.dtype, device=metric.device)
    if metric.shape != cost.shape:
        raise ValueError("Y and costs must have the same number of observations.")
    return CostNormalizedTrace(
        strategy=strategy,
        cumulative_cost=cost,
        metric=metric,
        metric_name="hypervolume",
    )


def inference_hv_regret_cost_trace(
    *,
    strategy: str,
    hypervolume: Any,
    costs: Any,
    reference_hypervolume: float,
) -> CostNormalizedTrace:
    """Build ``inference HV regret vs cumulative cost`` for a strategy."""

    metric = hypervolume_regret_trace(
        hypervolume,
        reference_hypervolume=reference_hypervolume,
    )
    cost = cumulative_cost(costs).to(dtype=metric.dtype, device=metric.device)
    if metric.shape != cost.shape:
        raise ValueError("hypervolume and costs must have the same length.")
    return CostNormalizedTrace(
        strategy=strategy,
        cumulative_cost=cost,
        metric=metric,
        metric_name="inference_hypervolume_regret",
    )


__all__ = [
    "CostNormalizedTrace",
    "best_objective_trace",
    "cumulative_cost",
    "hypervolume_regret_trace",
    "hypervolume_trace",
    "inference_hv_regret_cost_trace",
    "multi_objective_cost_trace",
    "single_objective_cost_trace",
]
