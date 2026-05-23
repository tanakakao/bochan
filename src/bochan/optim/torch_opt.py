"""Torch-optim based acquisition optimizers.

This module provides gradient-based acquisition-function optimizers implemented
with ``torch.optim``.  The API intentionally mirrors a useful subset of
``botorch.optim.optimize_acqf`` and ``botorch.optim.optimize_acqf_mixed`` while
keeping hard constraint / k-sparse logic in ``constraints.k_sparse``.

Typical use cases:
    - stochastic MC acquisition functions where repeated base-sample resampling
      makes quasi-Newton methods less attractive;
    - quick custom projected-gradient acquisition optimization;
    - hybrid workflows where initial candidates are generated / repaired by
      k-sparse utilities and then refined with torch optimizers.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import torch
from torch import Tensor
from botorch.acquisition.acquisition import AcquisitionFunction

try:
    from ..constraints.k_sparse import (
        LinearConstraint,
        ScoreMode,
        SupportSelection,
        expand_categorical_features,
        generate_k_sparse_initial_conditions,
        make_k_sparse_post_processing_func,
    )
except ImportError:
    from constraints.k_sparse import (  # type: ignore
        LinearConstraint,
        ScoreMode,
        SupportSelection,
        expand_categorical_features,
        generate_k_sparse_initial_conditions,
        make_k_sparse_post_processing_func,
    )

TorchOptimizerName = Literal["adam", "adamw", "sgd", "rmsprop", "lbfgs"]
InequalitySense = Literal["ge", "le"]


def _set_X_pending_on_acqf(
    acq_function: AcquisitionFunction,
    X_pending: Optional[Tensor],
) -> None:
    """Set ``X_pending`` when the acquisition function supports it."""
    if hasattr(acq_function, "set_X_pending"):
        acq_function.set_X_pending(X_pending)
    elif hasattr(acq_function, "X_pending"):
        acq_function.X_pending = X_pending  # type: ignore[attr-defined]


def _compose_post_processing(
    first: Optional[Callable[[Tensor], Tensor]],
    second: Optional[Callable[[Tensor], Tensor]],
) -> Optional[Callable[[Tensor], Tensor]]:
    if first is None:
        return second
    if second is None:
        return first

    def composed(X: Tensor) -> Tensor:
        return second(first(X))

    return composed


def _merge_fixed_features(
    *items: Optional[Dict[int, float]],
) -> Dict[int, float]:
    merged: Dict[int, float] = {}
    for item in items:
        if not item:
            continue
        for k, v in item.items():
            k_i = int(k)
            v_f = float(v)
            if k_i in merged and merged[k_i] != v_f:
                raise ValueError(
                    f"Conflicting fixed feature value for dim={k_i}: "
                    f"{merged[k_i]} vs {v_f}."
                )
            merged[k_i] = v_f
    return merged


def _apply_fixed_features(X: Tensor, fixed_features: Optional[Dict[int, float]]) -> Tensor:
    if not fixed_features:
        return X
    X_new = X.clone()
    for dim, value in fixed_features.items():
        X_new[..., int(dim)] = torch.as_tensor(value, device=X.device, dtype=X.dtype)
    return X_new


def _project_to_box_and_fixed_(
    X: Tensor,
    *,
    bounds: Tensor,
    fixed_features: Optional[Dict[int, float]],
) -> None:
    """In-place projection onto box bounds and fixed features."""
    lower = bounds[0].to(device=X.device, dtype=X.dtype)
    upper = bounds[1].to(device=X.device, dtype=X.dtype)
    X.clamp_(min=lower, max=upper)
    if fixed_features:
        for dim, value in fixed_features.items():
            X[..., int(dim)] = torch.as_tensor(value, device=X.device, dtype=X.dtype)


def _linear_constraints_penalty(
    X: Tensor,
    *,
    inequality_constraints: Optional[List[LinearConstraint]],
    equality_constraints: Optional[List[LinearConstraint]],
    inequality_sense: InequalitySense,
    penalty_factor: float,
) -> Tensor:
    """Return differentiable linear-constraint penalty with shape ``(N,)``.

    Args:
        X: Candidate tensor with shape ``(N, q, d)``.
        inequality_sense: ``"ge"`` means ``a^T x >= rhs`` (BoTorch-style);
            ``"le"`` means ``a^T x <= rhs``.
    """
    device, dtype = X.device, X.dtype
    penalty = torch.zeros(X.shape[0], device=device, dtype=dtype)

    for idxs, coeffs, rhs in inequality_constraints or []:
        idx_t = torch.as_tensor(list(idxs), device=device, dtype=torch.long)
        coeff_t = torch.as_tensor(list(coeffs), device=device, dtype=dtype)
        vals = (X[..., idx_t] * coeff_t).sum(dim=-1)  # (N, q)
        rhs_t = torch.as_tensor(rhs, device=device, dtype=dtype)
        if inequality_sense == "ge":
            violation = torch.clamp(rhs_t - vals, min=0.0)
        elif inequality_sense == "le":
            violation = torch.clamp(vals - rhs_t, min=0.0)
        else:
            raise ValueError(f"Unknown inequality_sense: {inequality_sense}")
        penalty = penalty + violation.max(dim=-1).values

    for idxs, coeffs, rhs in equality_constraints or []:
        idx_t = torch.as_tensor(list(idxs), device=device, dtype=torch.long)
        coeff_t = torch.as_tensor(list(coeffs), device=device, dtype=dtype)
        vals = (X[..., idx_t] * coeff_t).sum(dim=-1)  # (N, q)
        rhs_t = torch.as_tensor(rhs, device=device, dtype=dtype)
        penalty = penalty + (vals - rhs_t).abs().max(dim=-1).values

    return penalty_factor * penalty


def _evaluate_acq_values(
    acq_function: AcquisitionFunction,
    X: Tensor,
) -> Tensor:
    values = acq_function(X)
    if values.ndim == 0:
        values = values.reshape(1).repeat(X.shape[0])
    elif values.ndim > 1:
        values = values.reshape(values.shape[0], -1).mean(dim=-1)
    return values


def _make_optimizer(
    method: TorchOptimizerName,
    params: Sequence[Tensor],
    options: Dict,
) -> torch.optim.Optimizer:
    method_l = method.lower()
    lr = float(options.get("lr", 0.01))
    weight_decay = float(options.get("weight_decay", 0.0))

    if method_l == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if method_l == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if method_l == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(options.get("momentum", 0.0)),
            nesterov=bool(options.get("nesterov", False)),
            weight_decay=weight_decay,
        )
    if method_l == "rmsprop":
        return torch.optim.RMSprop(
            params,
            lr=lr,
            momentum=float(options.get("momentum", 0.0)),
            alpha=float(options.get("alpha", 0.99)),
            weight_decay=weight_decay,
        )
    if method_l == "lbfgs":
        return torch.optim.LBFGS(
            params,
            lr=lr,
            max_iter=int(options.get("lbfgs_max_iter", 20)),
            max_eval=options.get("lbfgs_max_eval", None),
            tolerance_grad=float(options.get("lbfgs_tolerance_grad", 1e-7)),
            tolerance_change=float(options.get("lbfgs_tolerance_change", 1e-9)),
            history_size=int(options.get("lbfgs_history_size", 10)),
            line_search_fn=options.get("lbfgs_line_search_fn", None),
        )
    raise ValueError(f"Unknown torch optimizer method: {method}")


def _prepare_candidate_for_eval(
    X: Tensor,
    *,
    candidate_transform: Optional[Callable[[Tensor], Tensor]],
    post_processing_func: Optional[Callable[[Tensor], Tensor]],
    apply_post_processing_during_eval: bool,
) -> Tensor:
    X_eval = X
    if candidate_transform is not None:
        X_eval = candidate_transform(X_eval)
    if post_processing_func is not None and apply_post_processing_during_eval:
        X_eval = post_processing_func(X_eval)
    return X_eval


def _score_candidates_no_grad(
    acq_function: AcquisitionFunction,
    X: Tensor,
    *,
    candidate_transform: Optional[Callable[[Tensor], Tensor]],
    post_processing_func: Optional[Callable[[Tensor], Tensor]],
    apply_post_processing_during_eval: bool,
    inequality_constraints: Optional[List[LinearConstraint]],
    equality_constraints: Optional[List[LinearConstraint]],
    inequality_sense: InequalitySense,
    penalty_factor: float,
) -> Tensor:
    with torch.no_grad():
        X_eval = _prepare_candidate_for_eval(
            X,
            candidate_transform=candidate_transform,
            post_processing_func=post_processing_func,
            apply_post_processing_during_eval=apply_post_processing_during_eval,
        )
        values = _evaluate_acq_values(acq_function, X_eval)
        penalty = _linear_constraints_penalty(
            X_eval,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            inequality_sense=inequality_sense,
            penalty_factor=penalty_factor,
        )
        score = values - penalty
        score = torch.where(torch.isnan(score), torch.full_like(score, -float("inf")), score)
    return score


def _make_random_initial_conditions(
    acq_function: AcquisitionFunction,
    *,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: Optional[int],
    fixed_features: Optional[Dict[int, float]],
    candidate_transform: Optional[Callable[[Tensor], Tensor]],
    post_processing_func: Optional[Callable[[Tensor], Tensor]],
    repair_initial_conditions: bool,
    inequality_constraints: Optional[List[LinearConstraint]],
    equality_constraints: Optional[List[LinearConstraint]],
    inequality_sense: InequalitySense,
    penalty_factor: float,
    options: Dict,
) -> Tensor:
    """Generate random initial conditions and downselect by acquisition value."""
    device, dtype = bounds.device, bounds.dtype
    d = bounds.shape[-1]
    n_raw = int(raw_samples or max(num_restarts, 1))
    n_raw = max(n_raw, num_restarts)

    generator = options.get("generator", None)
    lower = bounds[0].to(device=device, dtype=dtype)
    upper = bounds[1].to(device=device, dtype=dtype)
    Xraw = lower + torch.rand(n_raw, q, d, device=device, dtype=dtype, generator=generator) * (upper - lower)
    Xraw = _apply_fixed_features(Xraw, fixed_features)
    if post_processing_func is not None and repair_initial_conditions:
        Xraw = post_processing_func(Xraw)
        Xraw = _apply_fixed_features(Xraw, fixed_features)
        Xraw = Xraw.clamp(min=lower, max=upper)

    batch_limit = int(options.get("init_batch_limit", min(1024, n_raw)))
    scores = []
    for chunk in Xraw.split(batch_limit, dim=0):
        scores.append(
            _score_candidates_no_grad(
                acq_function,
                chunk,
                candidate_transform=candidate_transform,
                post_processing_func=post_processing_func,
                apply_post_processing_during_eval=False,
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
                inequality_sense=inequality_sense,
                penalty_factor=penalty_factor,
            )
        )
    score = torch.cat(scores, dim=0)
    if torch.isfinite(score).any():
        topk = min(num_restarts, Xraw.shape[0])
        idx = torch.topk(score, k=topk).indices
    else:
        idx = torch.arange(min(num_restarts, Xraw.shape[0]), device=device)
    return Xraw[idx].detach().clone()


def _normalize_batch_initial_conditions(
    batch_initial_conditions: Tensor,
    *,
    bounds: Tensor,
    q: int,
    fixed_features: Optional[Dict[int, float]],
    post_processing_func: Optional[Callable[[Tensor], Tensor]],
    repair_initial_conditions: bool,
) -> Tensor:
    X0 = batch_initial_conditions.to(device=bounds.device, dtype=bounds.dtype)
    if X0.ndim == 2:
        X0 = X0.unsqueeze(0)
    if X0.ndim != 3 or X0.shape[-2] != q or X0.shape[-1] != bounds.shape[-1]:
        raise ValueError(
            f"batch_initial_conditions must have shape (N, q={q}, d={bounds.shape[-1]}) "
            f"or (q, d). Got {tuple(X0.shape)}."
        )
    X0 = _apply_fixed_features(X0, fixed_features)
    if post_processing_func is not None and repair_initial_conditions:
        X0 = post_processing_func(X0)
        X0 = _apply_fixed_features(X0, fixed_features)
    return X0.clamp(min=bounds[0], max=bounds[1]).detach().clone()


def _optimize_acqf_torch_batch(
    *,
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    method: TorchOptimizerName,
    num_restarts: int,
    raw_samples: Optional[int],
    inequality_constraints: Optional[List[LinearConstraint]],
    equality_constraints: Optional[List[LinearConstraint]],
    inequality_sense: InequalitySense,
    fixed_features: Optional[Dict[int, float]],
    post_processing_func: Optional[Callable[[Tensor], Tensor]],
    candidate_transform: Optional[Callable[[Tensor], Tensor]],
    batch_initial_conditions: Optional[Tensor],
    options: Dict,
) -> Tuple[Tensor, Tensor]:
    """Optimize a q-batch with torch.optim and return best candidate."""
    fixed_features = {int(k): float(v) for k, v in (fixed_features or {}).items()}
    num_steps = int(options.get("num_steps", 100))
    penalty_factor = float(options.get("penalty_factor", 1e3))
    grad_clip_norm = options.get("grad_clip_norm", None)
    repair_initial_conditions = bool(options.get("repair_initial_conditions", True))
    apply_post_processing_during_eval = bool(options.get("apply_post_processing_during_eval", False))
    apply_post_processing_after_step = bool(options.get("apply_post_processing_after_step", False))
    track_best = bool(options.get("track_best", True))

    if batch_initial_conditions is not None:
        X0 = _normalize_batch_initial_conditions(
            batch_initial_conditions,
            bounds=bounds,
            q=q,
            fixed_features=fixed_features,
            post_processing_func=post_processing_func,
            repair_initial_conditions=repair_initial_conditions,
        )
    else:
        X0 = _make_random_initial_conditions(
            acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            fixed_features=fixed_features,
            candidate_transform=candidate_transform,
            post_processing_func=post_processing_func,
            repair_initial_conditions=repair_initial_conditions,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            inequality_sense=inequality_sense,
            penalty_factor=penalty_factor,
            options=options,
        )

    X = X0.detach().clone().requires_grad_(True)
    optimizer = _make_optimizer(method, [X], options)

    with torch.no_grad():
        _project_to_box_and_fixed_(X, bounds=bounds, fixed_features=fixed_features)

    best_X = X.detach().clone()
    best_score = _score_candidates_no_grad(
        acq_function,
        best_X,
        candidate_transform=candidate_transform,
        post_processing_func=post_processing_func,
        apply_post_processing_during_eval=apply_post_processing_during_eval,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        inequality_sense=inequality_sense,
        penalty_factor=penalty_factor,
    )

    def closure() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        X_eval = _prepare_candidate_for_eval(
            X,
            candidate_transform=candidate_transform,
            post_processing_func=post_processing_func,
            apply_post_processing_during_eval=apply_post_processing_during_eval,
        )
        values = _evaluate_acq_values(acq_function, X_eval)
        penalty = _linear_constraints_penalty(
            X_eval,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            inequality_sense=inequality_sense,
            penalty_factor=penalty_factor,
        )
        objective = values - penalty
        loss = -objective.sum()
        if not torch.isfinite(loss):
            loss = torch.nan_to_num(loss, nan=1e30, posinf=1e30, neginf=-1e30)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_([X], float(grad_clip_norm))
        return loss

    method_l = method.lower()
    for _ in range(num_steps):
        if method_l == "lbfgs":
            optimizer.step(closure)
        else:
            loss = closure()
            if not torch.isfinite(loss):
                break
            optimizer.step()

        with torch.no_grad():
            _project_to_box_and_fixed_(X, bounds=bounds, fixed_features=fixed_features)
            if post_processing_func is not None and apply_post_processing_after_step:
                X.copy_(post_processing_func(X))
                _project_to_box_and_fixed_(X, bounds=bounds, fixed_features=fixed_features)

            if track_best:
                score = _score_candidates_no_grad(
                    acq_function,
                    X,
                    candidate_transform=candidate_transform,
                    post_processing_func=post_processing_func,
                    apply_post_processing_during_eval=apply_post_processing_during_eval,
                    inequality_constraints=inequality_constraints,
                    equality_constraints=equality_constraints,
                    inequality_sense=inequality_sense,
                    penalty_factor=penalty_factor,
                )
                better = score > best_score
                if better.any():
                    best_X[better] = X.detach()[better]
                    best_score[better] = score[better]

    final_pool = best_X if track_best else X.detach().clone()
    if post_processing_func is not None:
        final_pool = post_processing_func(final_pool)
        final_pool = _apply_fixed_features(final_pool, fixed_features)
        final_pool = final_pool.clamp(min=bounds[0], max=bounds[1])

    final_score = _score_candidates_no_grad(
        acq_function,
        final_pool,
        candidate_transform=candidate_transform,
        post_processing_func=None,
        apply_post_processing_during_eval=False,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        inequality_sense=inequality_sense,
        penalty_factor=penalty_factor,
    )
    best_idx = torch.argmax(final_score)
    return final_pool[best_idx].detach(), final_score[best_idx].reshape(1).detach()


def optimize_acqf_torch(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int = 1,
    method: TorchOptimizerName = "adam",
    num_restarts: int = 10,
    raw_samples: Optional[int] = 512,
    inequality_constraints: Optional[List[LinearConstraint]] = None,
    equality_constraints: Optional[List[LinearConstraint]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
    batch_initial_conditions: Optional[Tensor] = None,
    return_best_only: bool = True,
    sequential: bool = False,
    options: Optional[Dict] = None,
    candidate_transform: Optional[Callable[[Tensor], Tensor]] = None,
    X_pending: Optional[Tensor] = None,
    inequality_sense: InequalitySense = "le",
) -> Tuple[Tensor, Tensor]:
    """Optimize an acquisition function with ``torch.optim``.

    The interface follows the common arguments of BoTorch's ``optimize_acqf``.
    The optimizer maximizes ``acq_function`` by minimizing the negative
    acquisition value.  Box constraints are enforced by projected gradient steps.

    Args:
        acq_function: BoTorch acquisition function.
        bounds: Tensor with shape ``(2, d)``.
        q: Number of candidates.
        method: One of ``"adam"``, ``"adamw"``, ``"sgd"``, ``"rmsprop"``,
            ``"lbfgs"``.
        num_restarts: Number of parallel restarts.
        raw_samples: Number of random q-batches used to pick initial conditions.
        inequality_constraints: Linear constraints in BoTorch tuple format.
        equality_constraints: Linear equality constraints.
        fixed_features: Fixed features ``{dim: value}``.
        post_processing_func: Optional final / projected repair function.
        batch_initial_conditions: Optional initial conditions of shape
            ``(N, q, d)`` or ``(q, d)``.
        sequential: If True and ``q > 1``, generate candidates one at a time and
            update ``X_pending``.
        options: Torch optimizer options. Common keys include ``lr``,
            ``num_steps``, ``penalty_factor``, ``repair_initial_conditions``,
            ``apply_post_processing_during_eval``, ``apply_post_processing_after_step``.
        candidate_transform: Optional transform applied before acquisition eval.
        X_pending: Optional pending points.
        inequality_sense: ``"ge"`` matches BoTorch's linear-constraint
            convention; use ``"le"`` for ``a^T x <= rhs``.
    """
    if not return_best_only:
        raise NotImplementedError("return_best_only=False is not supported for optimize_acqf_torch.")
    if bounds.shape[0] != 2:
        raise ValueError(f"bounds must have shape (2, d). Got {tuple(bounds.shape)}.")

    options = dict(options or {})
    bounds = bounds.to(dtype=bounds.dtype)
    base_X_pending = X_pending
    if base_X_pending is None and hasattr(acq_function, "X_pending"):
        base_X_pending = getattr(acq_function, "X_pending")

    def _run_single(q_local: int, X_pending_local: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        _set_X_pending_on_acqf(acq_function, X_pending_local)
        bic = batch_initial_conditions
        if bic is not None and q_local != q:
            # Sequential optimization uses q_local=1.  Reuse only if compatible.
            bic_t = bic.to(device=bounds.device, dtype=bounds.dtype)
            if bic_t.ndim == 3 and bic_t.shape[-2] >= 1:
                bic = bic_t[:, :1, :]
            elif bic_t.ndim == 2 and bic_t.shape[-2] >= 1:
                bic = bic_t[:1, :]
            else:
                bic = None
        return _optimize_acqf_torch_batch(
            acq_function=acq_function,
            bounds=bounds,
            q=q_local,
            method=method,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            inequality_sense=inequality_sense,
            fixed_features=fixed_features,
            post_processing_func=post_processing_func,
            candidate_transform=candidate_transform,
            batch_initial_conditions=bic,
            options=options,
        )

    if (not sequential) or q == 1:
        return _run_single(q, base_X_pending)

    selected: List[Tensor] = []
    values: List[Tensor] = []
    for _ in range(q):
        cur_pending = base_X_pending
        if selected:
            X_sel = torch.stack(selected, dim=0)
            cur_pending = X_sel if cur_pending is None else torch.cat([cur_pending, X_sel], dim=-2)
        X_i, v_i = _run_single(1, cur_pending)
        selected.append(X_i.squeeze(0).detach())
        values.append(v_i.reshape(-1)[0].detach())

    return torch.stack(selected, dim=0), torch.stack(values).view(q, 1)


def optimize_acqf_torch_mixed(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int = 1,
    method: TorchOptimizerName = "adam",
    fixed_features_list: Optional[List[Dict[int, float]]] = None,
    categorical_features: Optional[Dict[int, Sequence[float]]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    num_restarts: int = 10,
    raw_samples: Optional[int] = 512,
    inequality_constraints: Optional[List[LinearConstraint]] = None,
    equality_constraints: Optional[List[LinearConstraint]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
    batch_initial_conditions: Optional[Tensor] = None,
    return_best_only: bool = True,
    sequential: bool = False,
    options: Optional[Dict] = None,
    candidate_transform: Optional[Callable[[Tensor], Tensor]] = None,
    X_pending: Optional[Tensor] = None,
    inequality_sense: InequalitySense = "le",
) -> Tuple[Tensor, Tensor]:
    """Mixed-variable torch optimizer.

    This mirrors the role of BoTorch's ``optimize_acqf_mixed``: each entry in
    ``fixed_features_list`` defines one categorical / discrete assignment, and
    continuous free dimensions are optimized with ``torch.optim``.  You may pass
    either ``fixed_features_list`` directly or a compact ``categorical_features``
    spec such as ``{3: [0, 1, 2]}``.
    """
    if not return_best_only:
        raise NotImplementedError("return_best_only=False is not supported for optimize_acqf_mixed_torch.")

    options = dict(options or {})
    base_fixed = {int(k): float(v) for k, v in (fixed_features or {}).items()}
    if fixed_features_list is None:
        fixed_features_list = expand_categorical_features(
            categorical_features or {},
            base_fixed_features=base_fixed,
        )
    else:
        fixed_features_list = [_merge_fixed_features(base_fixed, item) for item in fixed_features_list]
    if not fixed_features_list:
        raise ValueError("fixed_features_list must contain at least one assignment.")

    base_X_pending = X_pending
    if base_X_pending is None and hasattr(acq_function, "X_pending"):
        base_X_pending = getattr(acq_function, "X_pending")

    def _best_for_assignments(q_local: int, X_pending_local: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        candidates: List[Tensor] = []
        values: List[Tensor] = []
        for ff in fixed_features_list or []:
            X_i, v_i = optimize_acqf_torch(
                acq_function=acq_function,
                bounds=bounds,
                q=q_local,
                method=method,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
                fixed_features=ff,
                post_processing_func=post_processing_func,
                batch_initial_conditions=batch_initial_conditions,
                return_best_only=True,
                sequential=False,
                options=options,
                candidate_transform=candidate_transform,
                X_pending=X_pending_local,
                inequality_sense=inequality_sense,
            )
            candidates.append(X_i)
            values.append(v_i.reshape(-1)[0])
        vals = torch.stack(values)
        best = torch.argmax(vals)
        return candidates[int(best)].detach(), vals[int(best)].reshape(1).detach()

    if (not sequential) or q == 1:
        return _best_for_assignments(q, base_X_pending)

    selected: List[Tensor] = []
    values: List[Tensor] = []
    for _ in range(q):
        cur_pending = base_X_pending
        if selected:
            X_sel = torch.stack(selected, dim=0)
            cur_pending = X_sel if cur_pending is None else torch.cat([cur_pending, X_sel], dim=-2)
        X_i, v_i = _best_for_assignments(1, cur_pending)
        selected.append(X_i.squeeze(0).detach())
        values.append(v_i.reshape(-1)[0].detach())
    return torch.stack(selected, dim=0), torch.stack(values).view(q, 1)


def optimize_acqf_torch_k_sparse(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int = 1,
    method: TorchOptimizerName = "adam",
    num_restarts: int = 10,
    raw_samples: Optional[int] = 512,
    *,
    comp_idx: Sequence[int],
    k: int,
    score: ScoreMode = "abs",
    equality_constraints: Optional[List[LinearConstraint]] = None,
    inequality_constraints: Optional[List[LinearConstraint]] = None,
    inequality_sense: InequalitySense = "le",
    fixed_features: Optional[Dict[int, float]] = None,
    final_sum_constraint: Optional[Tuple[Sequence[int], float]] = None,
    diversify: bool = False,
    diversify_kwargs: Optional[Dict] = None,
    support_selection: SupportSelection = "topk",
    batch_initial_conditions: Optional[Tensor] = None,
    return_best_only: bool = True,
    sequential: bool = False,
    options: Optional[Dict] = None,
    candidate_transform: Optional[Callable[[Tensor], Tensor]] = None,
    X_pending: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """k-sparse convenience wrapper around :func:`optimize_acqf_torch`."""
    options = dict(options or {})
    post = make_k_sparse_post_processing_func(
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
    )

    if batch_initial_conditions is None:
        batch_initial_conditions = generate_k_sparse_initial_conditions(
            bounds,
            num_restarts=num_restarts,
            q=q,
            comp_idx=comp_idx,
            k=k,
            fixed_features=fixed_features,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
            inequality_sense=inequality_sense,
            final_sum_constraint=final_sum_constraint,
            score=score,
            support_selection=support_selection,
            dtype=bounds.dtype,
            device=bounds.device,
        )
    else:
        batch_initial_conditions = post(batch_initial_conditions.to(device=bounds.device, dtype=bounds.dtype))

    return optimize_acqf_torch(
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        method=method,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        inequality_sense=inequality_sense,
        fixed_features=fixed_features,
        post_processing_func=post,
        batch_initial_conditions=batch_initial_conditions,
        return_best_only=return_best_only,
        sequential=sequential,
        options=options,
        candidate_transform=candidate_transform,
        X_pending=X_pending,
    )


def optimize_acqf_torch_mixed_k_sparse(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int = 1,
    method: TorchOptimizerName = "adam",
    fixed_features_list: Optional[List[Dict[int, float]]] = None,
    categorical_features: Optional[Dict[int, Sequence[float]]] = None,
    num_restarts: int = 10,
    raw_samples: Optional[int] = 512,
    *,
    comp_idx: Sequence[int],
    k: int,
    score: ScoreMode = "abs",
    equality_constraints: Optional[List[LinearConstraint]] = None,
    inequality_constraints: Optional[List[LinearConstraint]] = None,
    inequality_sense: InequalitySense = "le",
    fixed_features: Optional[Dict[int, float]] = None,
    final_sum_constraint: Optional[Tuple[Sequence[int], float]] = None,
    diversify: bool = False,
    diversify_kwargs: Optional[Dict] = None,
    support_selection: SupportSelection = "topk",
    allow_sparse_on_fixed_features: bool = False,
    batch_initial_conditions: Optional[Tensor] = None,
    return_best_only: bool = True,
    sequential: bool = False,
    options: Optional[Dict] = None,
    candidate_transform: Optional[Callable[[Tensor], Tensor]] = None,
    X_pending: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Mixed-variable k-sparse convenience wrapper around torch optimizer."""
    base_fixed = {int(k_): float(v) for k_, v in (fixed_features or {}).items()}
    if fixed_features_list is None:
        fixed_features_list = expand_categorical_features(
            categorical_features or {},
            base_fixed_features=base_fixed,
        )
    else:
        fixed_features_list = [_merge_fixed_features(base_fixed, item) for item in fixed_features_list]

    if not allow_sparse_on_fixed_features:
        sparse_set = {int(i) for i in comp_idx}
        fixed_set = set().union(*(set(ff.keys()) for ff in fixed_features_list)) if fixed_features_list else set()
        overlap = sparse_set & fixed_set
        if overlap:
            raise ValueError(
                f"comp_idx overlaps with mixed fixed dimensions: {sorted(overlap)}. "
                "Use continuous composition dimensions for comp_idx, or set "
                "allow_sparse_on_fixed_features=True intentionally."
            )

    post = make_k_sparse_post_processing_func(
        bounds=bounds,
        comp_idx=comp_idx,
        k=k,
        score=score,
        equality_constraints=equality_constraints,
        inequality_constraints=inequality_constraints,
        inequality_sense=inequality_sense,
        fixed_features=base_fixed,
        final_sum_constraint=final_sum_constraint,
        diversify=diversify,
        diversify_kwargs=diversify_kwargs,
        support_selection=support_selection,
    )

    if batch_initial_conditions is None:
        batch_initial_conditions = generate_k_sparse_initial_conditions(
            bounds,
            num_restarts=num_restarts,
            q=q,
            comp_idx=comp_idx,
            k=k,
            fixed_features=base_fixed,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
            inequality_sense=inequality_sense,
            final_sum_constraint=final_sum_constraint,
            score=score,
            support_selection=support_selection,
            dtype=bounds.dtype,
            device=bounds.device,
        )
    else:
        batch_initial_conditions = post(batch_initial_conditions.to(device=bounds.device, dtype=bounds.dtype))

    return optimize_acqf_mixed_torch(
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        method=method,
        fixed_features_list=fixed_features_list,
        fixed_features=base_fixed,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        inequality_sense=inequality_sense,
        post_processing_func=post,
        batch_initial_conditions=batch_initial_conditions,
        return_best_only=return_best_only,
        sequential=sequential,
        options=options,
        candidate_transform=candidate_transform,
        X_pending=X_pending,
    )


__all__ = [
    "TorchOptimizerName",
    "optimize_acqf_torch",
    "optimize_acqf_torch_mixed",
    "optimize_acqf_torch_k_sparse",
    "optimize_acqf_torch_mixed_k_sparse",
]
