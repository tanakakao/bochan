"""Compatibility fallback for NSGA-II input constraints on older BoTorch.

Some supported BoTorch releases expose ``optimize_with_nsgaii`` without an
``inequality_constraints`` parameter.  Bochan's candidate repair pipeline can
still enforce those constraints.  This module therefore omits the unsupported
keyword, applies the configured post-processing function to the returned Pareto
candidates, re-evaluates their objective values, and verifies feasibility.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor

from . import nsgaii as _base
from .nsgaii_adapter import (
    _accepted_parameters,
    _apply_discrete_choices,
    _evaluate_objectives,
)

_APPLIED = False


def _supports_parameter(function: Callable[..., Any], name: str) -> bool:
    accepted, accepts_var_kwargs = _accepted_parameters(function)
    return accepts_var_kwargs or name in accepted


def _max_linear_constraint_violation(
    X: Tensor,
    constraints: Sequence[_base.LinearConstraint] | None,
) -> float:
    """Return the maximum violation for ``a^T x >= rhs`` constraints."""

    if not constraints:
        return 0.0

    violations: list[Tensor] = []
    for indices, coefficients, rhs in constraints:
        indices_t = torch.as_tensor(indices, dtype=torch.long, device=X.device).reshape(-1)
        coefficients_t = torch.as_tensor(
            coefficients,
            dtype=X.dtype,
            device=X.device,
        ).reshape(-1)
        if indices_t.numel() != coefficients_t.numel():
            raise ValueError(
                "indices and coefficients must have the same length. "
                f"Got {indices_t.numel()} and {coefficients_t.numel()}."
            )
        lhs = (X[..., indices_t] * coefficients_t).sum(dim=-1)
        violations.append(torch.clamp(float(rhs) - lhs, min=0.0))

    return float(torch.stack(violations, dim=-1).max().detach().cpu())


def _make_linear_constraint_compatible_optimizer(
    function: Callable[..., Any],
) -> Callable[..., Any]:
    """Wrap an NSGA-II optimizer with a repair-and-validate fallback."""

    raw_function = getattr(function, "_bochan_original", function)

    def compatible_optimize_with_nsgaii(*args: Any, **kwargs: Any):
        inequality_constraints = kwargs.get("inequality_constraints")
        if not inequality_constraints or _supports_parameter(
            raw_function,
            "inequality_constraints",
        ):
            return function(*args, **kwargs)

        filtered_kwargs = {
            name: value
            for name, value in kwargs.items()
            if _supports_parameter(raw_function, name)
        }

        with torch.no_grad():
            X, Y = raw_function(*args, **filtered_kwargs)

        changed = False
        discrete_choices = kwargs.get("discrete_choices")
        if discrete_choices and not _supports_parameter(raw_function, "discrete_choices"):
            X = _apply_discrete_choices(X, discrete_choices)
            changed = True

        post_processing_func = kwargs.get("post_processing_func")
        if post_processing_func is not None:
            # Apply once at the public boundary even when the older BoTorch
            # implementation accepted the function. This guarantees that the
            # final returned candidates, rather than only intermediate
            # population members, satisfy bochan's repair contract.
            X = post_processing_func(X)
            changed = True

        if changed:
            acq_function = kwargs.get("acq_function")
            if acq_function is None:
                raise RuntimeError(
                    "Could not re-evaluate repaired NSGA-II candidates because "
                    "acq_function was not provided as a keyword argument."
                )
            Y = _evaluate_objectives(
                acq_function=acq_function,
                X=X,
                objective=kwargs.get("objective"),
            )

        max_violation = _max_linear_constraint_violation(X, inequality_constraints)
        tolerance = float(kwargs.get("constraint_validation_tol", 1e-6))
        if max_violation > tolerance:
            raise RuntimeError(
                "The installed BoTorch optimize_with_nsgaii does not natively "
                "support inequality_constraints, and the returned candidates "
                "remain infeasible after compatibility post-processing. "
                f"max_violation={max_violation:.6g}, tolerance={tolerance:.6g}. "
                "Provide a CandidateRepairConfig / post_processing_func that "
                "repairs the configured linear constraints."
            )

        return X, Y

    compatible_optimize_with_nsgaii._bochan_original = raw_function  # type: ignore[attr-defined]
    return compatible_optimize_with_nsgaii


def apply_legacy_nsgaii_linear_constraint_compat() -> None:
    """Install the fallback exactly once."""

    global _APPLIED
    if _APPLIED:
        return
    _base.optimize_with_nsgaii = _make_linear_constraint_compatible_optimizer(
        _base.optimize_with_nsgaii
    )
    _APPLIED = True


__all__ = ["apply_legacy_nsgaii_linear_constraint_compat"]
