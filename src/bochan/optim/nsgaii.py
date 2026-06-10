"""NSGA-II optimizer wrapper for multi-output acquisition functions.

This module intentionally keeps the wrapper thin.  The primary use case is
optimizing ``MultiOutputPosteriorMean`` to extract predicted Pareto candidates,
rather than replacing EHVI / NEHVI style Bayesian optimization acquisitions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor

from botorch.utils.multi_objective.optimize import optimize_with_nsgaii

LinearConstraint = tuple[Sequence[int] | Tensor, Sequence[float] | Tensor, float]
OutcomeConstraint = Callable[[Tensor], Tensor]


def _constraint_to_tensor_tuple(
    constraint: LinearConstraint,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, float]:
    """Convert a sparse linear constraint to BoTorch tensor format.

    Args:
        constraint: ``(indices, coefficients, rhs)``.
        device: Target device.
        dtype: Target floating dtype for coefficients.

    Returns:
        Constraint tuple accepted by BoTorch optimizers.
    """
    indices, coefficients, rhs = constraint
    if isinstance(indices, Tensor):
        indices_t = indices.to(device=device, dtype=torch.long).reshape(-1)
    else:
        indices_t = torch.as_tensor(list(indices), device=device, dtype=torch.long)

    if isinstance(coefficients, Tensor):
        coefficients_t = coefficients.to(device=device, dtype=dtype).reshape(-1)
    else:
        coefficients_t = torch.as_tensor(list(coefficients), device=device, dtype=dtype)

    if indices_t.numel() != coefficients_t.numel():
        raise ValueError(
            "indices and coefficients must have the same length. "
            f"Got {indices_t.numel()} and {coefficients_t.numel()}."
        )
    return indices_t, coefficients_t, float(rhs)


def equality_constraints_to_inequality_constraints(
    equality_constraints: Sequence[LinearConstraint] | None,
    *,
    eps: float = 1e-6,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> list[tuple[Tensor, Tensor, float]]:
    """Approximate equality constraints by pairs of inequality constraints.

    ``optimize_with_nsgaii`` does not directly support equality constraints.  This
    helper converts

    ``sum_i X[..., indices[i]] * coefficients[i] = rhs``

    into

    ``sum_i X[..., indices[i]] * coefficients[i] >= rhs - eps``
    ``-sum_i X[..., indices[i]] * coefficients[i] >= -(rhs + eps)``.

    Args:
        equality_constraints: Equality constraints in sparse BoTorch format.
        eps: Feasibility tolerance around the equality target.
        device: Optional target device.  If omitted, CPU is used unless a tensor
            constraint provides a device.
        dtype: Optional target dtype.  If omitted, ``torch.double`` is used unless
            a tensor constraint provides a dtype.

    Returns:
        Inequality constraints in ``>=`` format.
    """
    if equality_constraints is None:
        return []
    if eps < 0:
        raise ValueError("eps must be non-negative.")

    resolved_device = device
    resolved_dtype = dtype
    if resolved_device is None or resolved_dtype is None:
        for indices, coefficients, _ in equality_constraints:
            if isinstance(indices, Tensor) and resolved_device is None:
                resolved_device = indices.device
            if isinstance(coefficients, Tensor):
                if resolved_device is None:
                    resolved_device = coefficients.device
                if resolved_dtype is None and coefficients.is_floating_point():
                    resolved_dtype = coefficients.dtype
            if resolved_device is not None and resolved_dtype is not None:
                break

    resolved_device = resolved_device or torch.device("cpu")
    resolved_dtype = resolved_dtype or torch.double

    converted: list[tuple[Tensor, Tensor, float]] = []
    for constraint in equality_constraints:
        indices_t, coefficients_t, rhs = _constraint_to_tensor_tuple(
            constraint,
            device=resolved_device,
            dtype=resolved_dtype,
        )
        converted.append((indices_t, coefficients_t, rhs - float(eps)))
        converted.append((indices_t, -coefficients_t, -(rhs + float(eps))))
    return converted


def _merge_inequality_constraints(
    inequality_constraints: Sequence[LinearConstraint] | None,
    equality_constraints: Sequence[LinearConstraint] | None,
    *,
    equality_tol: float,
    bounds: Tensor,
) -> list[tuple[Tensor, Tensor, float]] | None:
    """Merge direct inequality constraints and equality-derived constraints."""
    device = bounds.device
    dtype = bounds.dtype
    merged: list[tuple[Tensor, Tensor, float]] = []

    if inequality_constraints is not None:
        merged.extend(
            _constraint_to_tensor_tuple(
                constraint,
                device=device,
                dtype=dtype,
            )
            for constraint in inequality_constraints
        )

    merged.extend(
        equality_constraints_to_inequality_constraints(
            equality_constraints,
            eps=equality_tol,
            device=device,
            dtype=dtype,
        )
    )
    return merged or None


def _infer_num_objectives(
    acq_function: Any,
    *,
    num_objectives: int | None = None,
    ref_point: Tensor | Sequence[float] | None = None,
) -> int:
    """Infer the number of objectives used by NSGA-II."""
    if num_objectives is not None:
        return int(num_objectives)

    if ref_point is not None:
        if isinstance(ref_point, Tensor):
            return int(ref_point.numel())
        return len(list(ref_point))

    weights = getattr(acq_function, "weights", None)
    if isinstance(weights, Tensor):
        return int(weights.numel())
    if weights is not None:
        try:
            return len(list(weights))
        except TypeError:
            pass

    model = getattr(acq_function, "model", None)
    for attr in ("num_outputs", "_num_outputs", "num_objectives"):
        value = getattr(model, attr, None)
        if value is None:
            continue
        if callable(value):
            value = value()
        if value is not None:
            return int(value)

    raise ValueError(
        "num_objectives could not be inferred. Pass num_objectives explicitly "
        "via OptimizeConfig.optimizer_kwargs."
    )


def validate_discrete_choices(
    X: Tensor,
    discrete_choices: dict[int, Sequence[float] | Tensor] | None,
    *,
    atol: float = 1e-8,
) -> None:
    """Validate that returned candidates use only allowed discrete values.

    Args:
        X: Candidate tensor with shape ``n x d`` or ``... x d``.
        discrete_choices: Mapping from dimension index to allowed values.
        atol: Absolute tolerance for floating point comparison.

    Raises:
        ValueError: If any discrete dimension contains a value outside the
            corresponding allowed set.
    """
    if not discrete_choices:
        return

    for dim, choices in discrete_choices.items():
        dim_i = int(dim)
        values = X[..., dim_i]
        if isinstance(choices, Tensor):
            choices_t = choices.to(device=X.device, dtype=X.dtype).reshape(-1)
        else:
            choices_t = torch.as_tensor(list(choices), device=X.device, dtype=X.dtype)
        is_allowed = torch.isclose(values.unsqueeze(-1), choices_t, atol=atol, rtol=0.0).any(dim=-1)
        if not bool(is_allowed.all()):
            bad_values = values[~is_allowed].detach().cpu().unique().tolist()
            raise ValueError(
                f"NSGA-II returned values outside discrete_choices for dim={dim_i}: "
                f"bad_values={bad_values}, allowed_values={choices_t.detach().cpu().tolist()}"
            )


def optimize_acqf_nsgaii(
    acq_function: Any,
    bounds: Tensor,
    *,
    q: int | None = 10,
    num_objectives: int | None = None,
    ref_point: Tensor | Sequence[float] | None = None,
    objective: Any | None = None,
    constraints: Sequence[OutcomeConstraint] | None = None,
    inequality_constraints: Sequence[LinearConstraint] | None = None,
    equality_constraints: Sequence[LinearConstraint] | None = None,
    equality_tol: float = 1e-6,
    fixed_features: dict[int, float] | None = None,
    discrete_choices: dict[int, Sequence[float] | Tensor] | None = None,
    post_processing_func: Callable[[Tensor], Tensor] | None = None,
    population_size: int = 250,
    max_gen: int | None = 200,
    seed: int | None = None,
    max_attempts: int = 2,
    validate_output: bool = True,
    validate_discrete: bool = True,
    sequential: bool = False,
    **kwargs: Any,
) -> tuple[Tensor, Tensor]:
    """Optimize a multi-output acquisition function with NSGA-II.

    This is a compatibility wrapper around BoTorch's
    ``optimize_with_nsgaii``.  It is intended for acquisition functions that
    return multiple objective values, especially ``MultiOutputPosteriorMean``.

    Args:
        acq_function: Multi-output acquisition function.
        bounds: Search bounds with shape ``2 x d``.
        q: Number of Pareto candidates to return.  ``None`` returns the final
            population from BoTorch.
        num_objectives: Number of objectives.  Inferred from ``ref_point``,
            ``acq_function.weights``, or ``acq_function.model.num_outputs`` when
            omitted.
        ref_point: Optional outcome-space reference point / lower bound.
        objective: Optional objective transform passed to BoTorch.
        constraints: Outcome-space constraints.  Negative values are feasible.
        inequality_constraints: Input-space constraints in ``>=`` sparse format.
        equality_constraints: Equality constraints.  These are approximated as
            two inequalities using ``equality_tol``.
        equality_tol: Tolerance used when converting equality constraints.
        fixed_features: Fixed input dimensions.
        discrete_choices: Allowed values for discrete / integer / categorical
            dimensions.
        post_processing_func: Final candidate repair function.
        population_size: NSGA-II population size.
        max_gen: Number of NSGA-II generations.
        seed: Random seed.
        max_attempts: Number of attempts used by BoTorch when feasible points are
            not found.
        validate_output: If True, check finite output tensors.
        validate_discrete: If True, check that returned discrete dimensions are
            within ``discrete_choices``.
        sequential: Accepted for API compatibility.  NSGA-II is population-based
            and does not use sequential greedy optimization.
        **kwargs: Extra keyword arguments are ignored for compatibility with the
            existing optimizer dispatch layer.

    Returns:
        ``(X_pareto, Y_pareto)`` where ``X_pareto`` is the predicted Pareto set
        and ``Y_pareto`` is the corresponding acquisition/objective value.
    """
    if sequential:
        raise NotImplementedError("sequential=True is not supported for optimize_acqf_nsgaii.")
    if bounds.shape[0] != 2:
        raise ValueError(f"bounds must have shape 2 x d. Got shape={tuple(bounds.shape)}.")

    n_obj = _infer_num_objectives(
        acq_function,
        num_objectives=num_objectives,
        ref_point=ref_point,
    )

    merged_inequality_constraints = _merge_inequality_constraints(
        inequality_constraints,
        equality_constraints,
        equality_tol=equality_tol,
        bounds=bounds,
    )

    X_pareto, Y_pareto = optimize_with_nsgaii(
        acq_function=acq_function,
        bounds=bounds,
        num_objectives=n_obj,
        q=q,
        ref_point=ref_point,
        objective=objective,
        constraints=list(constraints) if constraints is not None else None,
        inequality_constraints=merged_inequality_constraints,
        population_size=population_size,
        max_gen=max_gen,
        seed=seed,
        fixed_features=fixed_features,
        max_attempts=max_attempts,
        discrete_choices=discrete_choices,
        post_processing_func=post_processing_func,
    )

    if validate_output:
        if not torch.isfinite(X_pareto).all():
            raise RuntimeError("NSGA-II returned non-finite candidate values.")
        if not torch.isfinite(Y_pareto).all():
            raise RuntimeError("NSGA-II returned non-finite objective values.")
    if validate_discrete:
        validate_discrete_choices(X_pareto, discrete_choices)

    return X_pareto, Y_pareto


__all__ = [
    "equality_constraints_to_inequality_constraints",
    "optimize_acqf_nsgaii",
    "validate_discrete_choices",
]
