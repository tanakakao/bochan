"""Synthetic benchmark problems for multi-fidelity optimization experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class SyntheticMultiFidelityProblem:
    """A synthetic objective with one explicit fidelity coordinate.

    The problem always follows BoTorch's maximization convention. The fidelity
    coordinate is part of ``X`` and ``target_fidelity`` is the expensive source
    against which optimization quality is measured.
    """

    name: str
    bounds: Tensor
    fidelity_feature: int
    fidelity_values: tuple[float, ...]
    target_fidelity: float
    num_objectives: int
    evaluate_fn: Callable[[Tensor], Tensor]
    cost_fn: Callable[[Tensor], Tensor]
    cost_config: Mapping[str, Any] | None = None
    ref_point: Tensor | None = None

    def __post_init__(self) -> None:
        bounds = torch.as_tensor(self.bounds)
        if not bounds.is_floating_point():
            bounds = bounds.to(dtype=torch.get_default_dtype())
        if bounds.ndim != 2 or bounds.shape[0] != 2 or bounds.shape[1] < 2:
            raise ValueError("bounds must have shape 2 x d with d >= 2.")
        if not bool(torch.isfinite(bounds).all()) or bool((bounds[0] >= bounds[1]).any()):
            raise ValueError("bounds must be finite with lower < upper for every feature.")

        d = int(bounds.shape[-1])
        feature = int(self.fidelity_feature)
        if feature < 0:
            feature += d
        if feature < 0 or feature >= d:
            raise ValueError("fidelity_feature is outside the problem dimension.")

        values = tuple(float(value) for value in self.fidelity_values)
        if not values or len(set(values)) != len(values):
            raise ValueError("fidelity_values must be non-empty and contain unique values.")
        if any(not torch.isfinite(bounds.new_tensor(value)) for value in values):
            raise ValueError("fidelity_values must be finite.")
        lower = float(bounds[0, feature])
        upper = float(bounds[1, feature])
        if any(value < lower or value > upper for value in values):
            raise ValueError("Every fidelity value must lie inside the fidelity bounds.")

        target = float(self.target_fidelity)
        if target not in values:
            raise ValueError("target_fidelity must be one of fidelity_values.")
        if int(self.num_objectives) < 1:
            raise ValueError("num_objectives must be positive.")

        ref_point = self.ref_point
        if ref_point is not None:
            ref_point = torch.as_tensor(ref_point, dtype=bounds.dtype, device=bounds.device).reshape(-1)
            if int(self.num_objectives) < 2:
                raise ValueError("ref_point is only valid for multi-objective problems.")
            if ref_point.numel() != int(self.num_objectives) or not bool(torch.isfinite(ref_point).all()):
                raise ValueError("ref_point must contain one finite value per objective.")

        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "fidelity_feature", feature)
        object.__setattr__(self, "fidelity_values", values)
        object.__setattr__(self, "target_fidelity", target)
        object.__setattr__(self, "num_objectives", int(self.num_objectives))
        object.__setattr__(self, "ref_point", ref_point)

    @property
    def dim(self) -> int:
        return int(self.bounds.shape[-1])

    def evaluate(self, X: Any) -> Tensor:
        """Evaluate objective values and normalize them to ``n x m``."""

        tensor = torch.as_tensor(X, dtype=self.bounds.dtype, device=self.bounds.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[-1] != self.dim:
            raise ValueError(f"X must have shape n x {self.dim}.")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("X must contain only finite values.")
        if bool(((tensor < self.bounds[0]) | (tensor > self.bounds[1])).any()):
            raise ValueError("X contains values outside problem bounds.")
        result = torch.as_tensor(
            self.evaluate_fn(tensor),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        if result.ndim == 1:
            result = result.unsqueeze(-1)
        if result.ndim != 2 or result.shape != (tensor.shape[0], self.num_objectives):
            raise ValueError(
                "evaluate_fn must return shape n or n x num_objectives; "
                f"received {tuple(result.shape)}."
            )
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Synthetic objective returned non-finite values.")
        return result

    def cost(self, X: Any) -> Tensor:
        """Return one strictly positive cost per evaluation."""

        tensor = torch.as_tensor(X, dtype=self.bounds.dtype, device=self.bounds.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[-1] != self.dim:
            raise ValueError(f"X must have shape n x {self.dim}.")
        values = torch.as_tensor(self.cost_fn(tensor), dtype=tensor.dtype, device=tensor.device)
        if values.ndim == 2 and values.shape[-1] == 1:
            values = values.squeeze(-1)
        if values.ndim == 0:
            values = values.expand(tensor.shape[0])
        if values.shape != (tensor.shape[0],):
            raise ValueError("cost_fn must return scalar, n, or n x 1 costs.")
        if not bool(torch.isfinite(values).all()) or not bool((values > 0).all()):
            raise ValueError("Synthetic evaluation costs must be finite and strictly positive.")
        return values

    def is_target_fidelity(self, X: Any, *, atol: float = 1e-8) -> Tensor:
        tensor = torch.as_tensor(X, dtype=self.bounds.dtype, device=self.bounds.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        target = tensor.new_tensor(self.target_fidelity)
        return torch.isclose(tensor[..., self.fidelity_feature], target, atol=atol, rtol=0.0)


def augmented_branin_problem(
    *,
    fidelity_values: Sequence[float] = (0.5, 0.75, 1.0),
    fixed_cost: float = 0.25,
    dtype: torch.dtype = torch.double,
    device: torch.device | str | None = None,
) -> SyntheticMultiFidelityProblem:
    """Return the maximization form of BoTorch's Augmented Branin problem."""

    from botorch.test_functions.multi_fidelity import AugmentedBranin

    device = torch.device(device or "cpu")
    objective = AugmentedBranin(negate=True).to(dtype=dtype, device=device)
    bounds = torch.tensor(
        [[-5.0, 0.0, 0.0], [10.0, 15.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    fixed_cost = float(fixed_cost)
    if fixed_cost <= 0:
        raise ValueError("fixed_cost must be positive.")

    return SyntheticMultiFidelityProblem(
        name="augmented_branin",
        bounds=bounds,
        fidelity_feature=2,
        fidelity_values=tuple(float(value) for value in fidelity_values),
        target_fidelity=1.0,
        num_objectives=1,
        evaluate_fn=objective,
        cost_fn=lambda X: fixed_cost + X[..., 2],
        cost_config={
            "kind": "affine",
            "fixed_cost": fixed_cost,
            "fidelity_weights": {2: 1.0},
        },
    )


def augmented_hartmann_problem(
    *,
    fidelity_values: Sequence[float] = (0.5, 0.75, 1.0),
    fixed_cost: float = 0.25,
    dtype: torch.dtype = torch.double,
    device: torch.device | str | None = None,
) -> SyntheticMultiFidelityProblem:
    """Return the maximization form of BoTorch's Augmented Hartmann problem."""

    from botorch.test_functions.multi_fidelity import AugmentedHartmann

    device = torch.device(device or "cpu")
    objective = AugmentedHartmann(negate=True).to(dtype=dtype, device=device)
    bounds = torch.stack(
        [
            torch.zeros(7, dtype=dtype, device=device),
            torch.ones(7, dtype=dtype, device=device),
        ]
    )
    fixed_cost = float(fixed_cost)
    if fixed_cost <= 0:
        raise ValueError("fixed_cost must be positive.")

    return SyntheticMultiFidelityProblem(
        name="augmented_hartmann",
        bounds=bounds,
        fidelity_feature=6,
        fidelity_values=tuple(float(value) for value in fidelity_values),
        target_fidelity=1.0,
        num_objectives=1,
        evaluate_fn=objective,
        cost_fn=lambda X: fixed_cost + X[..., 6],
        cost_config={
            "kind": "affine",
            "fixed_cost": fixed_cost,
            "fidelity_weights": {6: 1.0},
        },
    )


def momf_branin_currin_problem(
    *,
    fidelity_values: Sequence[float] = (0.5, 0.75, 1.0),
    dtype: torch.dtype = torch.double,
    device: torch.device | str | None = None,
) -> SyntheticMultiFidelityProblem:
    """Return BoTorch's bi-objective multi-fidelity Branin-Currin problem."""

    from botorch.test_functions.multi_objective_multi_fidelity import MOMFBraninCurrin

    device = torch.device(device or "cpu")
    objective = MOMFBraninCurrin(negate=True).to(dtype=dtype, device=device)
    bounds = torch.as_tensor(objective.bounds, dtype=dtype, device=device)
    feature = int(objective.dim - 1)

    def exponential_cost(X: Tensor) -> Tensor:
        return torch.exp(4.8 * X[..., feature])

    return SyntheticMultiFidelityProblem(
        name="momf_branin_currin",
        bounds=bounds,
        fidelity_feature=feature,
        fidelity_values=tuple(float(value) for value in fidelity_values),
        target_fidelity=1.0,
        num_objectives=2,
        evaluate_fn=objective,
        cost_fn=exponential_cost,
        cost_config={
            "kind": "callable",
            "cost_callable": exponential_cost,
        },
        ref_point=torch.zeros(2, dtype=dtype, device=device),
    )


__all__ = [
    "SyntheticMultiFidelityProblem",
    "augmented_branin_problem",
    "augmented_hartmann_problem",
    "momf_branin_currin_problem",
]
