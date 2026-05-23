"""Wrappers around BoTorch standard optimizers with k-sparse repair.

The k-sparse logic is imported from ``constraints.k_sparse``.  This file should
only contain optimizer-facing glue code.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf, optimize_acqf_mixed

try:  # package import
    from ..constraints.k_sparse import (
        LinearConstraint,
        expand_categorical_features,
        generate_k_sparse_initial_conditions,
        make_k_sparse_post_processing_func,
    )
except ImportError:  # flat-file fallback
    from constraints.k_sparse import (  # type: ignore
        LinearConstraint,
        expand_categorical_features,
        generate_k_sparse_initial_conditions,
        make_k_sparse_post_processing_func,
    )


def _compose_post_processing(
    first: Optional[Callable[[Tensor], Tensor]],
    second: Optional[Callable[[Tensor], Tensor]],
) -> Optional[Callable[[Tensor], Tensor]]:
    """Compose two post-processing functions as ``second(first(X))``."""
    if first is None:
        return second
    if second is None:
        return first

    def composed(X: Tensor) -> Tensor:
        return second(first(X))

    return composed


def _get_X_pending(acq_function: AcquisitionFunction) -> Optional[Tensor]:
    return getattr(acq_function, "X_pending", None)


def _set_X_pending(acq_function: AcquisitionFunction, X_pending: Optional[Tensor]) -> None:
    if hasattr(acq_function, "set_X_pending"):
        acq_function.set_X_pending(X_pending)
    elif hasattr(acq_function, "X_pending"):
        acq_function.X_pending = X_pending  # type: ignore[attr-defined]


def _pending_with_selected(
    base_X_pending: Optional[Tensor],
    selected: List[Tensor],
) -> Optional[Tensor]:
    if not selected:
        return base_X_pending
    X_selected = torch.cat(selected, dim=-2)  # (i, d)
    if base_X_pending is None:
        return X_selected
    return torch.cat([base_X_pending, X_selected], dim=-2)


def _select_sequential_ic(
    batch_initial_conditions: Optional[Tensor],
    step: int,
) -> Optional[Tensor]:
    """Return q=1 initial conditions for one sequential step.

    BoTorch does not allow ``batch_initial_conditions`` together with
    ``sequential=True``.  The wrappers therefore run their own sequential loop
    and call BoTorch with ``q=1``.  This helper adapts initial conditions to that
    per-step call.
    """
    if batch_initial_conditions is None:
        return None
    if batch_initial_conditions.ndim != 3:
        raise ValueError(
            "batch_initial_conditions must have shape (num_restarts, q, d). "
            f"Got shape={tuple(batch_initial_conditions.shape)}."
        )
    q_ic = batch_initial_conditions.shape[-2]
    if q_ic == 1:
        return batch_initial_conditions
    if step < q_ic:
        return batch_initial_conditions[:, step : step + 1, :]
    return batch_initial_conditions[:, -1:, :]


def optimize_acqf_k_sparse(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    *,
    q: int = 1,
    num_restarts: int = 10,
    raw_samples: int = 512,
    comp_idx: Sequence[int],
    k: int,
    inequality_constraints: Optional[List[LinearConstraint]] = None,
    equality_constraints: Optional[List[LinearConstraint]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
    batch_initial_conditions: Optional[Tensor] = None,
    return_best_only: bool = True,
    sequential: bool = False,
    options: Optional[Dict] = None,
    final_sum_constraint: Optional[Tuple[Sequence[int], float]] = None,
    diversify: bool = False,
    diversify_kwargs: Optional[Dict] = None,
    generate_initial_conditions: bool = True,
    support_selection: str = "topk",
    inequality_sense: str = "le",
    **kwargs,
):
    """Run ``optimize_acqf`` with k-sparse post-processing and initial conditions.

    Important behavior:
      - ``post_processing_func`` is always composed with k-sparse repair.
      - When ``generate_initial_conditions=True`` and no initial conditions are
        supplied, sparse initial conditions are generated and passed to BoTorch.
      - For ``sequential=True`` with ``q>1``, this wrapper runs a manual greedy
        sequential loop. This is necessary because BoTorch's standard
        ``optimize_acqf`` does not support ``batch_initial_conditions`` together
        with ``sequential=True``.

    This wrapper is useful when you want the final candidate to satisfy k-sparse
    constraints while still using BoTorch's gradient-based optimizer.
    """
    options = dict(options or {})
    fixed_features = dict(fixed_features or {})

    k_sparse_post = make_k_sparse_post_processing_func(
        bounds=bounds,
        comp_idx=comp_idx,
        k=k,
        equality_constraints=equality_constraints,
        inequality_constraints=inequality_constraints,
        inequality_sense=inequality_sense,  # type: ignore[arg-type]
        fixed_features=fixed_features,
        final_sum_constraint=final_sum_constraint,
        diversify=diversify,
        diversify_kwargs=diversify_kwargs,
        support_selection=support_selection,  # type: ignore[arg-type]
    )
    post = _compose_post_processing(post_processing_func, k_sparse_post)

    if batch_initial_conditions is None and generate_initial_conditions:
        ic_q = 1 if (sequential and q > 1) else q
        batch_initial_conditions = generate_k_sparse_initial_conditions(
            bounds=bounds,
            num_restarts=num_restarts,
            q=ic_q,
            comp_idx=comp_idx,
            k=k,
            fixed_features=fixed_features,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
            inequality_sense=inequality_sense,  # type: ignore[arg-type]
            final_sum_constraint=final_sum_constraint,
            support_selection=support_selection,  # type: ignore[arg-type]
        )
    elif batch_initial_conditions is not None:
        batch_initial_conditions = post(batch_initial_conditions) if post is not None else batch_initial_conditions

    # BoTorch does not allow batch_initial_conditions when sequential=True.
    # Since this wrapper intentionally generates sparse initial conditions, we
    # handle the sequential loop explicitly and call BoTorch with q=1.
    if sequential and q > 1:
        if not return_best_only:
            raise NotImplementedError("return_best_only=False is not supported for manual sequential optimization.")

        base_X_pending = _get_X_pending(acq_function)
        selected: List[Tensor] = []
        values: List[Tensor] = []

        try:
            for step in range(q):
                _set_X_pending(acq_function, _pending_with_selected(base_X_pending, selected))
                bic_step = _select_sequential_ic(batch_initial_conditions, step)

                cand_i, val_i = optimize_acqf(
                    acq_function=acq_function,
                    bounds=bounds,
                    q=1,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    inequality_constraints=inequality_constraints,
                    equality_constraints=equality_constraints,
                    fixed_features=fixed_features,
                    post_processing_func=post,
                    batch_initial_conditions=bic_step,
                    return_best_only=True,
                    sequential=False,
                    options=options,
                    **kwargs,
                )
                if post is not None:
                    cand_i = post(cand_i)
                selected.append(cand_i.detach())  # (1, d)
                values.append(val_i.reshape(-1)[0].detach())
        finally:
            _set_X_pending(acq_function, base_X_pending)

        candidate = torch.cat(selected, dim=-2)  # (q, d)
        acq_value = torch.stack(values)  # (q,)
        if post is not None:
            candidate = post(candidate)
        return candidate, acq_value

    candidate, acq_value = optimize_acqf(
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        fixed_features=fixed_features,
        post_processing_func=post,
        batch_initial_conditions=batch_initial_conditions,
        return_best_only=return_best_only,
        sequential=sequential,
        options=options,
        **kwargs,
    )

    # Defensive final repair.  This is intentionally repeated because some BoTorch
    # paths may apply transforms before / after fixed features.
    if post is not None:
        candidate = post(candidate)
    return candidate, acq_value


def optimize_acqf_mixed_k_sparse(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    *,
    q: int = 1,
    num_restarts: int = 10,
    raw_samples: int = 512,
    comp_idx: Sequence[int],
    k: int,
    fixed_features_list: Optional[List[Dict[int, float]]] = None,
    categorical_features: Optional[Dict[int, Sequence[float]]] = None,
    inequality_constraints: Optional[List[LinearConstraint]] = None,
    equality_constraints: Optional[List[LinearConstraint]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
    batch_initial_conditions: Optional[Tensor] = None,
    return_best_only: bool = True,
    sequential: bool = False,
    options: Optional[Dict] = None,
    final_sum_constraint: Optional[Tuple[Sequence[int], float]] = None,
    diversify: bool = False,
    diversify_kwargs: Optional[Dict] = None,
    generate_initial_conditions: bool = True,
    support_selection: str = "topk",
    inequality_sense: str = "le",
    allow_sparse_on_fixed_features: bool = False,
    **kwargs,
):
    """Run ``optimize_acqf_mixed`` with k-sparse repair.

    Use ``fixed_features_list`` directly, or provide ``categorical_features`` as
    ``{dim: [values...]}`` to auto-expand it.

    Recommendation: keep categorical dimensions out of ``comp_idx``.  k-sparse
    repair zeros / projects dimensions, which is usually not meaningful for
    categorical features.
    """
    options = dict(options or {})
    fixed_features = dict(fixed_features or {})

    if fixed_features_list is None:
        fixed_features_list = expand_categorical_features(
            categorical_features or {},
            base_fixed_features=fixed_features,
        )
    else:
        fixed_features_list = [dict(item) for item in fixed_features_list]
        if fixed_features:
            fixed_features_list = [{**fixed_features, **item} for item in fixed_features_list]

    if not fixed_features_list:
        raise ValueError("fixed_features_list is empty. Provide at least one category combination.")

    fixed_dims = set()
    for item in fixed_features_list:
        fixed_dims.update(int(k_) for k_ in item.keys())
    overlap = set(int(i) for i in comp_idx) & fixed_dims
    if overlap and not allow_sparse_on_fixed_features:
        raise ValueError(
            "comp_idx overlaps with fixed / categorical dimensions. "
            f"overlap={sorted(overlap)}. Remove categorical dims from comp_idx, "
            "or pass allow_sparse_on_fixed_features=True."
        )

    # The post-processing repair should respect common fixed features only.  The
    # category-specific fixed features are enforced internally by optimize_acqf_mixed.
    k_sparse_post = make_k_sparse_post_processing_func(
        bounds=bounds,
        comp_idx=comp_idx,
        k=k,
        equality_constraints=equality_constraints,
        inequality_constraints=inequality_constraints,
        inequality_sense=inequality_sense,  # type: ignore[arg-type]
        fixed_features=fixed_features,
        final_sum_constraint=final_sum_constraint,
        diversify=diversify,
        diversify_kwargs=diversify_kwargs,
        support_selection=support_selection,  # type: ignore[arg-type]
    )
    post = _compose_post_processing(post_processing_func, k_sparse_post)

    if batch_initial_conditions is None and generate_initial_conditions:
        ic_q = 1 if (sequential and q > 1) else q
        batch_initial_conditions = generate_k_sparse_initial_conditions(
            bounds=bounds,
            num_restarts=num_restarts,
            q=ic_q,
            comp_idx=comp_idx,
            k=k,
            fixed_features=fixed_features,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
            inequality_sense=inequality_sense,  # type: ignore[arg-type]
            final_sum_constraint=final_sum_constraint,
            support_selection=support_selection,  # type: ignore[arg-type]
        )
    elif batch_initial_conditions is not None:
        batch_initial_conditions = post(batch_initial_conditions) if post is not None else batch_initial_conditions

    # Same reason as in optimize_acqf_k_sparse: BoTorch's sequential path rejects
    # batch_initial_conditions. Run a manual greedy loop so sparse initial
    # conditions can still be used.
    if sequential and q > 1:
        if not return_best_only:
            raise NotImplementedError("return_best_only=False is not supported for manual sequential optimization.")

        base_X_pending = _get_X_pending(acq_function)
        selected: List[Tensor] = []
        values: List[Tensor] = []

        try:
            for step in range(q):
                _set_X_pending(acq_function, _pending_with_selected(base_X_pending, selected))
                bic_step = _select_sequential_ic(batch_initial_conditions, step)

                cand_i, val_i = optimize_acqf_mixed(
                    acq_function=acq_function,
                    bounds=bounds,
                    q=1,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    fixed_features_list=fixed_features_list,
                    inequality_constraints=inequality_constraints,
                    equality_constraints=equality_constraints,
                    post_processing_func=post,
                    batch_initial_conditions=bic_step,
                    return_best_only=True,
                    sequential=False,
                    options=options,
                    **kwargs,
                )
                if post is not None:
                    cand_i = post(cand_i)
                selected.append(cand_i.detach())
                values.append(val_i.reshape(-1)[0].detach())
        finally:
            _set_X_pending(acq_function, base_X_pending)

        candidate = torch.cat(selected, dim=-2)
        acq_value = torch.stack(values)
        if post is not None:
            candidate = post(candidate)
        return candidate, acq_value

    candidate, acq_value = optimize_acqf_mixed(
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        fixed_features_list=fixed_features_list,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        post_processing_func=post,
        batch_initial_conditions=batch_initial_conditions,
        return_best_only=return_best_only,
        sequential=sequential,
        options=options,
        **kwargs,
    )

    if post is not None:
        candidate = post(candidate)
    return candidate, acq_value
