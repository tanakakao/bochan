"""Linear-constraint repair and validation utilities for BoTorch candidates.

This module is intentionally optimizer-agnostic.  All public repair factories
return ``Callable[[Tensor], Tensor]`` so they can be used as BoTorch
``post_processing_func`` objects or composed with other candidate repairs.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

LinearConstraint = Tuple[Union[Sequence[int], Tensor], Union[Sequence[float], Tensor], float]
InequalitySense = Literal["ge", "le"]


def to_1d_long_tensor(indices: Union[Sequence[int], Tensor], *, device: torch.device) -> Tensor:
    """Convert indices to a flattened long tensor."""
    if isinstance(indices, Tensor):
        return indices.to(device=device, dtype=torch.long).reshape(-1)
    return torch.as_tensor(list(indices), device=device, dtype=torch.long).reshape(-1)


def to_1d_value_tensor(
    values: Union[Sequence[float], Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Convert numeric values to a flattened tensor with the requested dtype."""
    if isinstance(values, Tensor):
        return values.to(device=device, dtype=dtype).reshape(-1)
    return torch.as_tensor(list(values), device=device, dtype=dtype).reshape(-1)


def normalize_bounds(bounds: Tensor, *, d: int, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
    """Return lower / upper bounds with shape ``(d,)``.

    Args:
        bounds: BoTorch-style bounds with shape ``(2, d)``.
        d: Expected feature dimension.
        device: Destination device.
        dtype: Destination dtype.

    Returns:
        ``(lower, upper)`` tensors.
    """
    bounds_t = bounds.to(device=device, dtype=dtype)
    if bounds_t.ndim < 2 or bounds_t.shape[0] != 2:
        raise ValueError(f"bounds must have shape (2, d). Got {tuple(bounds.shape)}.")
    lower = bounds_t[0].reshape(-1)[-d:]
    upper = bounds_t[1].reshape(-1)[-d:]
    if lower.numel() != d or upper.numel() != d:
        raise ValueError(f"bounds last dimension is incompatible with d={d}.")
    if torch.any(lower > upper):
        raise ValueError("bounds lower must be <= upper for every dimension.")
    return lower, upper


def normalize_constraints(
    constraints: Optional[Sequence[LinearConstraint]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Tuple[Tensor, Tensor, float]]:
    """Normalize linear constraints to tensor tuples.

    Each constraint is ``(indices, coefficients, rhs)``.
    """
    if constraints is None:
        return []

    out: List[Tuple[Tensor, Tensor, float]] = []
    for indices, coefficients, rhs in constraints:
        idx = to_1d_long_tensor(indices, device=device)
        coef = to_1d_value_tensor(coefficients, device=device, dtype=dtype)
        if idx.numel() != coef.numel():
            raise ValueError("Each constraint requires len(indices) == len(coefficients).")
        out.append((idx, coef, float(rhs)))
    return out


def dense_constraint_vector(indices: Tensor, coefficients: Tensor, *, d: int) -> Tensor:
    """Create a dense vector ``a`` where ``a[indices] = coefficients``."""
    a = torch.zeros(d, device=indices.device, dtype=coefficients.dtype)
    if torch.any(indices < 0) or torch.any(indices >= d):
        raise ValueError(f"constraint indices out of range for d={d}.")
    a[indices] = coefficients
    return a


def _apply_fixed_features(Xf: Tensor, fixed_features: Optional[Dict[int, float]]) -> Tensor:
    if not fixed_features:
        return Xf
    for j, value in fixed_features.items():
        Xf[:, int(j)] = torch.as_tensor(value, device=Xf.device, dtype=Xf.dtype)
    return Xf


def _make_allowed_mask(
    *,
    n: int,
    d: int,
    device: torch.device,
    dtype: torch.dtype,
    fixed_features: Optional[Dict[int, float]] = None,
    adjustable_mask: Optional[Tensor] = None,
) -> Tensor:
    allowed = torch.ones(n, d, device=device, dtype=dtype)
    if adjustable_mask is not None:
        mask = adjustable_mask.to(device=device, dtype=torch.bool).reshape(-1)
        if mask.numel() != d:
            raise ValueError(f"adjustable_mask must have length {d}.")
        allowed = allowed * mask.to(dtype=dtype).unsqueeze(0)
    if fixed_features:
        for j in fixed_features:
            allowed[:, int(j)] = 0.0
    return allowed


def project_linear_constraints(
    X: Tensor,
    *,
    bounds: Tensor,
    equality_constraints: Optional[Sequence[LinearConstraint]] = None,
    inequality_constraints: Optional[Sequence[LinearConstraint]] = None,
    inequality_sense: InequalitySense = "ge",
    fixed_features: Optional[Dict[int, float]] = None,
    adjustable_mask: Optional[Tensor] = None,
    max_iters: int = 10,
    clamp_each_iter: bool = True,
) -> Tensor:
    """Approximately project candidates onto box, fixed-feature, and linear constraints.

    This is a continuous projection.  If it is used after grid rounding, the final
    values may no longer lie exactly on the grid.  Use it before a final discrete
    rounding step if grid exactness is more important than exact linear repair.

    Args:
        X: Candidate tensor with shape ``(..., d)``.
        bounds: BoTorch-style bounds with shape ``(2, d)``.
        equality_constraints: ``sum(coeff * x[idx]) == rhs`` constraints.
        inequality_constraints: Linear inequality constraints.
        inequality_sense: ``"ge"`` means ``a^T x >= rhs``. ``"le"`` means
            ``a^T x <= rhs``.
        fixed_features: Dimensions that must be fixed to specified values.
        adjustable_mask: Boolean mask of dimensions that are allowed to move.
        max_iters: Number of projection passes.
        clamp_each_iter: Whether to clamp to bounds during each pass.

    Returns:
        Repaired tensor with the same shape as ``X``.
    """
    if X.ndim < 1:
        raise ValueError("X must have shape (..., d).")
    if inequality_sense not in ("ge", "le"):
        raise ValueError("inequality_sense must be 'ge' or 'le'.")

    orig_shape = X.shape
    d = int(orig_shape[-1])
    device, dtype = X.device, X.dtype
    lower, upper = normalize_bounds(bounds, d=d, device=device, dtype=dtype)

    eq_constraints = normalize_constraints(equality_constraints, device=device, dtype=dtype)
    ineq_constraints = normalize_constraints(inequality_constraints, device=device, dtype=dtype)

    Xf = X.reshape(-1, d).clone()
    n = Xf.shape[0]
    allowed_base = _make_allowed_mask(
        n=n,
        d=d,
        device=device,
        dtype=dtype,
        fixed_features=fixed_features,
        adjustable_mask=adjustable_mask,
    )

    Xf = Xf.clamp(min=lower, max=upper)
    Xf = _apply_fixed_features(Xf, fixed_features)

    for _ in range(max_iters):
        if clamp_each_iter:
            Xf = Xf.clamp(min=lower, max=upper)
            Xf = _apply_fixed_features(Xf, fixed_features)

        # Equality projection: x <- x + ((rhs - a^T x) / ||a_allowed||^2) a_allowed
        for idxs, coeffs, rhs in eq_constraints:
            a = dense_constraint_vector(idxs, coeffs, d=d)
            a_allowed = allowed_base * a.unsqueeze(0)
            norm2 = (a_allowed * a_allowed).sum(dim=1)
            can = norm2 > 1e-12
            if not can.any():
                continue
            resid = torch.as_tensor(rhs, device=device, dtype=dtype) - (Xf * a).sum(dim=1)
            Xf[can] = Xf[can] + (resid[can] / norm2[can]).unsqueeze(1) * a_allowed[can]

        # Half-space projection.
        for idxs, coeffs, rhs in ineq_constraints:
            a = dense_constraint_vector(idxs, coeffs, d=d)
            lhs = (Xf * a).sum(dim=1)
            rhs_t = torch.as_tensor(rhs, device=device, dtype=dtype)

            if inequality_sense == "ge":
                violation = rhs_t - lhs
                bad = violation > 0
                direction = a
            else:
                violation = lhs - rhs_t
                bad = violation > 0
                direction = -a

            if not bad.any():
                continue

            a_allowed = allowed_base * direction.unsqueeze(0)
            norm2 = (a_allowed * a_allowed).sum(dim=1)
            can = bad & (norm2 > 1e-12)
            if not can.any():
                continue
            Xf[can] = Xf[can] + (violation[can] / norm2[can]).unsqueeze(1) * a_allowed[can]

    Xf = Xf.clamp(min=lower, max=upper)
    Xf = _apply_fixed_features(Xf, fixed_features)
    return Xf.reshape(orig_shape)


def make_linear_constraint_repair_func(
    *,
    bounds: Tensor,
    equality_constraints: Optional[Sequence[LinearConstraint]] = None,
    inequality_constraints: Optional[Sequence[LinearConstraint]] = None,
    inequality_sense: InequalitySense = "ge",
    fixed_features: Optional[Dict[int, float]] = None,
    adjustable_mask: Optional[Tensor] = None,
    max_iters: int = 10,
) -> Callable[[Tensor], Tensor]:
    """Create a BoTorch-compatible linear-constraint post-processing function."""

    def repair(X: Tensor) -> Tensor:
        return project_linear_constraints(
            X,
            bounds=bounds,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
            inequality_sense=inequality_sense,
            fixed_features=fixed_features,
            adjustable_mask=adjustable_mask,
            max_iters=max_iters,
        )

    return repair


def linear_constraint_violations(
    X: Tensor,
    *,
    equality_constraints: Optional[Sequence[LinearConstraint]] = None,
    inequality_constraints: Optional[Sequence[LinearConstraint]] = None,
    inequality_sense: InequalitySense = "ge",
) -> Dict[str, Tensor]:
    """Return per-candidate linear-constraint violation magnitudes.

    Returns:
        Dictionary with ``eq`` and ``ineq`` tensors.  Each tensor has shape
        ``X.shape[:-1] + (num_constraints,)``.  Missing constraint groups are
        returned as empty tensors on the same device/dtype.
    """
    if X.ndim < 1:
        raise ValueError("X must have shape (..., d).")
    d = X.shape[-1]
    device, dtype = X.device, X.dtype
    Xf = X.reshape(-1, d)

    eq_constraints = normalize_constraints(equality_constraints, device=device, dtype=dtype)
    ineq_constraints = normalize_constraints(inequality_constraints, device=device, dtype=dtype)

    eq_vals: List[Tensor] = []
    for idxs, coeffs, rhs in eq_constraints:
        a = dense_constraint_vector(idxs, coeffs, d=d)
        lhs = (Xf * a).sum(dim=1)
        eq_vals.append((lhs - torch.as_tensor(rhs, device=device, dtype=dtype)).abs())

    ineq_vals: List[Tensor] = []
    for idxs, coeffs, rhs in ineq_constraints:
        a = dense_constraint_vector(idxs, coeffs, d=d)
        lhs = (Xf * a).sum(dim=1)
        rhs_t = torch.as_tensor(rhs, device=device, dtype=dtype)
        if inequality_sense == "ge":
            ineq_vals.append(torch.clamp(rhs_t - lhs, min=0.0))
        elif inequality_sense == "le":
            ineq_vals.append(torch.clamp(lhs - rhs_t, min=0.0))
        else:
            raise ValueError("inequality_sense must be 'ge' or 'le'.")

    out_shape = X.shape[:-1]
    eq = torch.stack(eq_vals, dim=-1).reshape(out_shape + (len(eq_vals),)) if eq_vals else torch.empty(out_shape + (0,), device=device, dtype=dtype)
    ineq = torch.stack(ineq_vals, dim=-1).reshape(out_shape + (len(ineq_vals),)) if ineq_vals else torch.empty(out_shape + (0,), device=device, dtype=dtype)
    return {"eq": eq, "ineq": ineq}


def convert_legacy_constraints(
    constraint_indices: Sequence[Sequence[int]],
    constraint_coeffs: Sequence[Sequence[float]],
    constraint_targets: Sequence[float],
    constraint_ops: Sequence[str],
    *,
    inequality_sense: InequalitySense = "ge",
) -> Tuple[List[LinearConstraint], List[LinearConstraint]]:
    """Convert legacy ``idx/coefs/target/op`` constraints to normalized lists.

    Args:
        constraint_indices: List of index lists.
        constraint_coeffs: List of coefficient lists.
        constraint_targets: RHS values.
        constraint_ops: Operators: ``=``, ``==``, ``>=`` or ``<=``.
        inequality_sense: Desired sense for the returned inequality list.

    Returns:
        ``(equality_constraints, inequality_constraints)``.
    """
    equality_constraints: List[LinearConstraint] = []
    inequality_constraints: List[LinearConstraint] = []

    for idxs, coeffs, target, op in zip(
        constraint_indices,
        constraint_coeffs,
        constraint_targets,
        constraint_ops,
    ):
        idx_list = [int(i) for i in idxs]
        coef_list = [float(c) for c in coeffs]
        target_f = float(target)

        if op in ("=", "=="):
            equality_constraints.append((idx_list, coef_list, target_f))
        elif op == ">=":
            if inequality_sense == "ge":
                inequality_constraints.append((idx_list, coef_list, target_f))
            else:
                inequality_constraints.append((idx_list, [-c for c in coef_list], -target_f))
        elif op == "<=":
            if inequality_sense == "le":
                inequality_constraints.append((idx_list, coef_list, target_f))
            else:
                inequality_constraints.append((idx_list, [-c for c in coef_list], -target_f))
        else:
            raise ValueError(f"Unsupported constraint op: {op}")

    return equality_constraints, inequality_constraints
