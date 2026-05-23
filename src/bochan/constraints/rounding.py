"""Grid-rounding utilities for BoTorch candidate tensors.

The functions in this module are non-differentiable and are intended for
candidate post-processing, not for use inside acquisition-function forward
passes.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional, Sequence, Union

import torch
from torch import Tensor

from .constraints import LinearConstraint, make_linear_constraint_repair_func, normalize_bounds

FinalPriority = Literal["grid", "constraints"]


def identity_post_processing_func(X: Tensor) -> Tensor:
    """Return ``X`` unchanged.

    This is useful when an optional post-processing step is disabled, e.g.
    ``steps=None`` for grid rounding.
    """
    return X


def _as_1d_long_tensor(
    x: Optional[Union[Sequence[int], Tensor]],
    *,
    device: torch.device,
) -> Optional[Tensor]:
    if x is None:
        return None
    return torch.as_tensor(x, dtype=torch.long, device=device).reshape(-1)


def resolve_step_full(
    steps: Tensor,
    *,
    d: int,
    numeric_indices: Optional[Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Resolve step sizes to a full ``(d,)`` tensor."""
    steps_t = steps.to(device=device, dtype=dtype).reshape(-1)
    if numeric_indices is None:
        if steps_t.numel() != d:
            raise ValueError(f"steps must have length {d} when numeric_indices is None.")
        step_full = steps_t.clone()
    else:
        if steps_t.numel() == d:
            step_full = steps_t.clone()
        elif steps_t.numel() == numeric_indices.numel():
            step_full = torch.ones(d, device=device, dtype=dtype)
            step_full[numeric_indices] = steps_t
        else:
            raise ValueError(
                "steps must have length d or len(numeric_indices). "
                f"Got steps={steps_t.numel()}, d={d}, len(numeric_indices)={numeric_indices.numel()}."
            )
    if torch.any(step_full <= 0):
        raise ValueError("All steps must be positive.")
    return step_full


def round_numeric(
    X: Tensor,
    *,
    steps: Optional[Tensor] = None,
    bounds: Tensor,
    base: Optional[Tensor] = None,
    numeric_indices: Optional[Union[Sequence[int], Tensor]] = None,
    clamp: bool = True,
) -> Tensor:
    """Round selected numeric dimensions to a regular grid.

    Args:
        X: Candidate tensor with shape ``(..., d)``.
        steps: Grid step sizes with shape ``(d,)`` or ``(len(numeric_indices),)``.
            If ``None``, no rounding or clamping is performed and ``X`` is
            returned unchanged.
        bounds: BoTorch-style bounds with shape ``(2, d)``.
        base: Grid origin.  Defaults to ``bounds[0]``.
        numeric_indices: Dimensions to round.  ``None`` means all dimensions.
        clamp: Whether to clamp the final tensor to bounds.

    Returns:
        Rounded tensor with the same shape as ``X``.
    """
    if X.ndim < 1:
        raise ValueError("X must have shape (..., d).")

    if steps is None:
        return X

    d = int(X.shape[-1])
    device, dtype = X.device, X.dtype
    lower, upper = normalize_bounds(bounds, d=d, device=device, dtype=dtype)
    base_t = lower if base is None else base.to(device=device, dtype=dtype).reshape(-1)
    if base_t.numel() != d:
        raise ValueError(f"base must have length {d}.")

    numeric_idx = _as_1d_long_tensor(numeric_indices, device=device)
    if numeric_idx is None:
        numeric_idx = torch.arange(d, device=device, dtype=torch.long)

    step_full = resolve_step_full(
        steps,
        d=d,
        numeric_indices=numeric_idx,
        device=device,
        dtype=dtype,
    )

    Xp = X.clone()
    X_num = Xp[..., numeric_idx]
    Xp[..., numeric_idx] = (
        torch.round((X_num - base_t[numeric_idx]) / step_full[numeric_idx])
        * step_full[numeric_idx]
        + base_t[numeric_idx]
    )
    if clamp:
        Xp = torch.maximum(Xp, lower)
        Xp = torch.minimum(Xp, upper)
    return Xp


def round_numeric_preserve_sparse_support(
    X: Tensor,
    *,
    steps: Optional[Tensor] = None,
    bounds: Tensor,
    base: Optional[Tensor] = None,
    numeric_indices: Optional[Union[Sequence[int], Tensor]] = None,
    sparse_indices: Optional[Union[Sequence[int], Tensor]] = None,
    support_eps: float = 0.0,
) -> Tensor:
    """Round numeric dimensions while preserving inactive sparse components.

    ``sparse_indices`` are inspected before rounding.  Entries whose absolute
    value is <= ``support_eps`` are considered inactive and are forced back to
    zero after rounding / clamping.

    If ``steps`` is ``None``, this function is an identity transform.
    """
    if steps is None:
        return X

    device = X.device
    sparse_idx = _as_1d_long_tensor(sparse_indices, device=device)
    support_mask = None
    if sparse_idx is not None and sparse_idx.numel() > 0:
        support_mask = X[..., sparse_idx].abs() > support_eps

    Xp = round_numeric(
        X,
        steps=steps,
        bounds=bounds,
        base=base,
        numeric_indices=numeric_indices,
        clamp=True,
    )

    if sparse_idx is not None and sparse_idx.numel() > 0:
        group = Xp[..., sparse_idx]
        Xp[..., sparse_idx] = torch.where(support_mask, group, torch.zeros_like(group))
    return Xp


def make_grid_rounding_post_processing_func(
    *,
    steps: Optional[Tensor] = None,
    bounds: Tensor,
    base: Optional[Tensor] = None,
    numeric_indices: Optional[Union[Sequence[int], Tensor]] = None,
    sparse_indices: Optional[Union[Sequence[int], Tensor]] = None,
    support_eps: float = 0.0,
) -> Callable[[Tensor], Tensor]:
    """Create a BoTorch-compatible grid-rounding post-processing function.

    If ``steps`` is ``None``, no grid rounding is applied and an identity
    function is returned.  This makes optional grid rounding easy to compose
    with k-sparse or linear-constraint repairs.
    """

    if steps is None:
        return identity_post_processing_func

    def post_process(X: Tensor) -> Tensor:
        if sparse_indices is None:
            return round_numeric(
                X,
                steps=steps,
                bounds=bounds,
                base=base,
                numeric_indices=numeric_indices,
            )
        return round_numeric_preserve_sparse_support(
            X,
            steps=steps,
            bounds=bounds,
            base=base,
            numeric_indices=numeric_indices,
            sparse_indices=sparse_indices,
            support_eps=support_eps,
        )

    return post_process


def make_grid_rounding_with_linear_repair_func(
    *,
    steps: Optional[Tensor] = None,
    bounds: Tensor,
    base: Optional[Tensor] = None,
    numeric_indices: Optional[Union[Sequence[int], Tensor]] = None,
    sparse_indices: Optional[Union[Sequence[int], Tensor]] = None,
    equality_constraints: Optional[Sequence[LinearConstraint]] = None,
    inequality_constraints: Optional[Sequence[LinearConstraint]] = None,
    inequality_sense: Literal["ge", "le"] = "ge",
    fixed_features: Optional[dict[int, float]] = None,
    max_iters: int = 8,
    num_alternations: int = 2,
    final_priority: FinalPriority = "grid",
    support_eps: float = 0.0,
) -> Callable[[Tensor], Tensor]:
    """Create a grid-rounding + continuous linear-repair post-processing function.

    Notes:
        - If ``steps`` is ``None``, only linear repair is applied.
        - ``final_priority='grid'`` guarantees final grid rounding but may leave
          small linear-constraint residuals.
        - ``final_priority='constraints'`` prioritizes the linear projection but
          may move values slightly off-grid.
    """
    grid_post = make_grid_rounding_post_processing_func(
        steps=steps,
        bounds=bounds,
        base=base,
        numeric_indices=numeric_indices,
        sparse_indices=sparse_indices,
        support_eps=support_eps,
    )

    repair_post = make_linear_constraint_repair_func(
        bounds=bounds,
        equality_constraints=equality_constraints,
        inequality_constraints=inequality_constraints,
        inequality_sense=inequality_sense,
        fixed_features=fixed_features,
        max_iters=max_iters,
    )

    def post_process(X: Tensor) -> Tensor:
        Xp = X
        for _ in range(max(1, num_alternations)):
            Xp = grid_post(Xp)
            Xp = repair_post(Xp)
        if final_priority == "grid":
            Xp = grid_post(Xp)
        elif final_priority == "constraints":
            Xp = repair_post(Xp)
        else:
            raise ValueError("final_priority must be 'grid' or 'constraints'.")
        return Xp

    return post_process


def grid_residual(
    X: Tensor,
    *,
    steps: Optional[Tensor] = None,
    bounds: Tensor,
    base: Optional[Tensor] = None,
    numeric_indices: Optional[Union[Sequence[int], Tensor]] = None,
) -> Tensor:
    """Return distance from the nearest grid point in step units.

    If ``steps`` is ``None``, returns an empty tensor with shape
    ``X.shape[:-1] + (0,)``.
    """
    if steps is None:
        return torch.empty(X.shape[:-1] + (0,), device=X.device, dtype=X.dtype)

    d = int(X.shape[-1])
    device, dtype = X.device, X.dtype
    lower, _ = normalize_bounds(bounds, d=d, device=device, dtype=dtype)
    base_t = lower if base is None else base.to(device=device, dtype=dtype).reshape(-1)
    numeric_idx = _as_1d_long_tensor(numeric_indices, device=device)
    if numeric_idx is None:
        numeric_idx = torch.arange(d, device=device, dtype=torch.long)
    step_full = resolve_step_full(steps, d=d, numeric_indices=numeric_idx, device=device, dtype=dtype)
    z = (X[..., numeric_idx] - base_t[numeric_idx]) / step_full[numeric_idx]
    return z - torch.round(z)
