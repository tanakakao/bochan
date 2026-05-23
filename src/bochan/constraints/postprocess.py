"""Composition helpers for BoTorch candidate post-processing pipelines."""

from __future__ import annotations

from typing import Callable, Dict, List, Literal, Optional, Sequence

import torch
from torch import Tensor

from .constraints import LinearConstraint, linear_constraint_violations, normalize_bounds
from .rounding import grid_residual, make_grid_rounding_post_processing_func
from .ksparse import make_k_sparse_post_processing_func

FinalPriority = Literal["grid", "constraints"]


def compose_post_processing_funcs(*funcs: Optional[Callable[[Tensor], Tensor]]) -> Callable[[Tensor], Tensor]:
    """Compose multiple ``Tensor -> Tensor`` post-processing functions.

    ``None`` entries are ignored, so optional pipeline parts can be assembled
    without extra branching.
    """
    active_funcs = [f for f in funcs if f is not None]

    def post_process(X: Tensor) -> Tensor:
        Xp = X
        for func in active_funcs:
            Xp = func(Xp)
        return Xp

    return post_process


def make_grid_k_sparse_post_processing_func(
    *,
    bounds: Tensor,
    steps: Optional[Tensor] = None,
    comp_idx: Optional[Sequence[int]] = None,
    k: int = 0,
    numeric_indices: Optional[Sequence[int]] = None,
    equality_constraints: Optional[List[LinearConstraint]] = None,
    inequality_constraints: Optional[List[LinearConstraint]] = None,
    inequality_sense: Literal["le", "ge"] = "le",
    fixed_features: Optional[Dict[int, float]] = None,
    final_sum_constraint: Optional[tuple[Sequence[int], float]] = None,
    diversify: bool = False,
    diversify_kwargs: Optional[Dict] = None,
    score: Literal["abs", "value"] = "abs",
    support_selection: Literal["topk", "sample"] = "topk",
    sample_tau: float = 0.2,
    sample_eps: float = 0.05,
    generator: Optional[torch.Generator] = None,
    max_iters: int = 12,
    num_alternations: int = 2,
    final_priority: FinalPriority = "grid",
    support_eps: float = 0.0,
) -> Callable[[Tensor], Tensor]:
    """Create a combined k-sparse + grid-rounding post-processing function.

    Args:
        bounds: BoTorch-style bounds with shape ``(2, d)``.
        steps: Grid step sizes with shape ``(d,)`` or ``(len(numeric_indices),)``.
            If ``None``, grid rounding is disabled while k-sparse / linear
            repairs remain active.
        comp_idx: Dimensions subject to k-sparse support.
        k: Maximum number of active components inside ``comp_idx``.  The current
            k-sparse repair keeps exactly up to ``k`` selected support entries.
        numeric_indices: Dimensions to round.  ``None`` means all dimensions.
        equality_constraints: Linear equality constraints.
        inequality_constraints: Linear inequality constraints.
        inequality_sense: Sense used by the k-sparse repair.  ``"le"`` means
            ``a^T x <= rhs``; ``"ge"`` means ``a^T x >= rhs``.
        fixed_features: Fixed feature dictionary.
        final_sum_constraint: Optional ``(indices, rhs)`` sum-on-active-support
            constraint.
        diversify: Whether to perturb duplicate q-batch points and re-repair.
        diversify_kwargs: Extra args for ``diversify_within_q``.
        final_priority: ``"grid"`` ends with grid rounding; ``"constraints"``
            ends with k-sparse / linear repair.

    Returns:
        BoTorch-compatible ``post_processing_func``.
    """
    comp_idx = [] if comp_idx is None else list(comp_idx)
    
    k_post = make_k_sparse_post_processing_func(
        bounds=bounds,
        comp_idx=comp_idx,
        k=k,
        score=score,
        equality_constraints=equality_constraints,
        inequality_constraints=inequality_constraints,
        inequality_sense=inequality_sense,
        fixed_features=fixed_features,
        final_sum_constraint=final_sum_constraint,
        diversify=diversify,
        diversify_kwargs=diversify_kwargs,
        support_selection=support_selection,
        sample_tau=sample_tau,
        sample_eps=sample_eps,
        generator=generator,
        max_iters=max_iters,
    )

    grid_post = make_grid_rounding_post_processing_func(
        steps=steps,
        bounds=bounds,
        base=bounds[0],
        numeric_indices=numeric_indices,
        sparse_indices=comp_idx,
        support_eps=support_eps,
    )

    def post_process(X: Tensor) -> Tensor:
        Xp = X
        for _ in range(max(1, num_alternations)):
            Xp = k_post(Xp)
            Xp = grid_post(Xp)

        if final_priority == "grid":
            Xp = grid_post(k_post(Xp))
        elif final_priority == "constraints":
            Xp = k_post(grid_post(Xp))
        else:
            raise ValueError("final_priority must be 'grid' or 'constraints'.")
        return Xp

    return post_process


def validate_post_processed_candidates(
    X: Tensor,
    *,
    bounds: Tensor,
    steps: Optional[Tensor] = None,
    numeric_indices: Optional[Sequence[int]] = None,
    base: Optional[Tensor] = None,
    comp_idx: Optional[Sequence[int]] = None,
    k: Optional[int] = None,
    equality_constraints: Optional[Sequence[LinearConstraint]] = None,
    inequality_constraints: Optional[Sequence[LinearConstraint]] = None,
    inequality_sense: Literal["ge", "le"] = "ge",
    tol: float = 1e-6,
) -> Dict[str, object]:
    """Validate bounds, grid, k-sparse, and linear constraints.

    Returns a small dictionary of scalar diagnostics.  This function is useful
    after post-processing because some combinations of grid + sparse + equality
    constraints can be infeasible.
    """
    if X.ndim < 1:
        raise ValueError("X must have shape (..., d).")
    d = int(X.shape[-1])
    device, dtype = X.device, X.dtype
    lower, upper = normalize_bounds(bounds, d=d, device=device, dtype=dtype)

    lower_violation = torch.clamp(lower - X, min=0.0).max().item()
    upper_violation = torch.clamp(X - upper, min=0.0).max().item()

    result: Dict[str, object] = {
        "lower_violation": float(lower_violation),
        "upper_violation": float(upper_violation),
        "is_bounds_ok": lower_violation <= tol and upper_violation <= tol,
    }

    if steps is not None:
        gres = grid_residual(
            X,
            steps=steps,
            bounds=bounds,
            base=base,
            numeric_indices=numeric_indices,
        )
        max_grid_error = gres.abs().max().item() if gres.numel() > 0 else 0.0
        result.update({
            "max_grid_error": float(max_grid_error),
            "is_grid_ok": max_grid_error <= tol,
        })

    if comp_idx is not None and k is not None:
        idx = torch.as_tensor(list(comp_idx), device=device, dtype=torch.long)
        comp = X[..., idx]
        active_count = (comp.abs() > tol).sum(dim=-1)
        max_active = int(active_count.max().item()) if active_count.numel() > 0 else 0
        min_active = int(active_count.min().item()) if active_count.numel() > 0 else 0
        result.update({
            "min_active_count": min_active,
            "max_active_count": max_active,
            "is_k_sparse_ok": max_active <= int(k),
        })

    if equality_constraints or inequality_constraints:
        viol = linear_constraint_violations(
            X,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
            inequality_sense=inequality_sense,
        )
        max_eq = viol["eq"].max().item() if viol["eq"].numel() > 0 else 0.0
        max_ineq = viol["ineq"].max().item() if viol["ineq"].numel() > 0 else 0.0
        result.update({
            "max_equality_violation": float(max_eq),
            "max_inequality_violation": float(max_ineq),
            "is_linear_constraints_ok": max(max_eq, max_ineq) <= tol,
        })

    return result
