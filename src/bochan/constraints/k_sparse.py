"""k-sparse constraints and repair utilities.

This module is optimizer-agnostic.  Standard BoTorch optimizers and custom
non-gradient optimizers should import repair / post-processing functions from
here rather than duplicating k-sparse logic.
"""

from __future__ import annotations

from itertools import product
from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import torch
from torch import Tensor

LinearConstraint = Tuple[Sequence[int], Sequence[float], float]
ScoreMode = Literal["abs", "value"]
SupportSelection = Literal["topk", "sample"]


def _to_1d_long_tensor(indices: Sequence[int] | Tensor, *, device: torch.device) -> Tensor:
    if isinstance(indices, Tensor):
        return indices.to(device=device, dtype=torch.long).reshape(-1)
    return torch.as_tensor(list(indices), device=device, dtype=torch.long)


def _to_1d_value_tensor(values: Sequence[float] | Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    if isinstance(values, Tensor):
        return values.to(device=device, dtype=dtype).reshape(-1)
    return torch.as_tensor(list(values), device=device, dtype=dtype)


def _normalize_bounds(bounds: Tensor, *, d: int, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
    """Return flattened lower / upper bounds with shape ``(d,)``."""
    if bounds.shape[0] != 2:
        raise ValueError(f"bounds must have first dimension 2. Got shape={tuple(bounds.shape)}.")
    lower = bounds[0].to(device=device, dtype=dtype).reshape(-1)[-d:]
    upper = bounds[1].to(device=device, dtype=dtype).reshape(-1)[-d:]
    if lower.numel() != d or upper.numel() != d:
        raise ValueError(f"bounds last dimension is incompatible with d={d}.")
    return lower, upper


def k_exact_sparse_transform_factory(
    comp_idx: Sequence[int],
    k: int,
    score: ScoreMode = "abs",
    min_active: float = 0.0,
) -> Callable[[Tensor], Tensor]:
    """Create a transform that keeps exactly top-``k`` components in ``comp_idx``.

    This transform only zeros inactive entries.  It does not enforce a sum-to-one
    constraint; combine it with :func:`enforce_sum_on_support` or
    :func:`make_k_sparse_linear_constraints_repair` when a composition sum is
    required.

    Args:
        comp_idx: Candidate dimensions to sparsify.
        k: Number of active components.  ``k <= 0`` zeros all ``comp_idx`` dims.
        score: ``"abs"`` selects by absolute value; ``"value"`` selects by raw value.
        min_active: Optional lower clamp for selected entries.

    Returns:
        A transform ``X -> X_new`` supporting ``(..., q, d)`` and ``(..., d)``.
    """
    comp_idx = [int(i) for i in comp_idx]

    def transform(X: Tensor) -> Tensor:
        if not comp_idx:
            return X

        X_new = X.clone()
        idx_t = torch.as_tensor(comp_idx, device=X.device, dtype=torch.long)
        group = X_new.index_select(dim=-1, index=idx_t)
        m = group.shape[-1]

        if k <= 0:
            X_new[..., idx_t] = 0.0
            return X_new
        if k > m:
            raise ValueError(f"exact-k requires len(comp_idx) >= k. Got len={m}, k={k}.")

        scores = group.abs() if score == "abs" else group
        topk_idx = scores.topk(k, dim=-1).indices
        mask = torch.zeros_like(group, dtype=torch.bool).scatter(-1, topk_idx, True)
        sparse_group = torch.where(mask, group, torch.zeros_like(group))
        if min_active > 0:
            sparse_group = torch.where(mask, sparse_group.clamp_min(min_active), sparse_group)

        X_new[..., idx_t] = sparse_group
        return X_new

    return transform


def sample_k_without_replacement(
    scores: Tensor,
    k: int,
    *,
    tau: float = 0.2,
    eps: float = 0.05,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Sample ``k`` indices without replacement from score-derived probabilities.

    Args:
        scores: Tensor with shape ``(..., d)``.
        k: Number of indices to sample.
        tau: Softmax temperature.  Smaller values are closer to top-k selection.
        eps: Uniform mixture weight to keep exploration probability.
        generator: Optional torch generator.

    Returns:
        Long tensor with shape ``(..., min(k, d))``.
    """
    if k < 1:
        raise ValueError("k must be >= 1.")
    d = scores.shape[-1]
    k_eff = min(k, d)

    scaled = scores / max(float(tau), 1e-12)
    scaled = scaled - scaled.max(dim=-1, keepdim=True).values
    probs = torch.softmax(scaled, dim=-1)

    if eps > 0:
        probs = (1.0 - eps) * probs + eps / d

    flat = probs.reshape(-1, d)
    idx = torch.multinomial(flat, num_samples=k_eff, replacement=False, generator=generator)
    return idx.reshape(scores.shape[:-1] + (k_eff,))


def _select_support_mask(
    group: Tensor,
    *,
    k: int,
    score: ScoreMode,
    support_selection: SupportSelection,
    sample_tau: float,
    sample_eps: float,
    generator: Optional[torch.Generator],
) -> Tensor:
    """Return boolean support mask for ``group`` with shape ``(N, m)``."""
    m = group.shape[-1]
    if k <= 0:
        return torch.zeros_like(group, dtype=torch.bool)
    k_eff = min(k, m)
    scores = group.abs() if score == "abs" else group
    if support_selection == "topk":
        idx = scores.topk(k_eff, dim=-1).indices
    elif support_selection == "sample":
        idx = sample_k_without_replacement(
            scores=scores,
            k=k_eff,
            tau=sample_tau,
            eps=sample_eps,
            generator=generator,
        )
    else:
        raise ValueError(f"Unknown support_selection: {support_selection}")
    return torch.zeros_like(group, dtype=torch.bool).scatter(-1, idx, True)


def make_k_sparse_linear_constraints_repair(
    bounds: Tensor,
    comp_idx: Sequence[int],
    k: int,
    *,
    score: ScoreMode = "abs",
    equality_constraints: Optional[List[LinearConstraint]] = None,
    inequality_constraints: Optional[List[LinearConstraint]] = None,
    inequality_sense: Literal["le", "ge"] = "le",
    fixed_features: Optional[Dict[int, float]] = None,
    max_iters: int = 12,
    support_selection: SupportSelection = "topk",
    sample_tau: float = 0.2,
    sample_eps: float = 0.05,
    generator: Optional[torch.Generator] = None,
) -> Callable[[Tensor], Tensor]:
    """Create a k-sparse repair function with box / fixed / linear constraints.

    The returned function works on ``(..., q, d)`` and ``(..., d)`` tensors.  It
    first fixes a k-sparse support inside ``comp_idx`` and then performs repeated
    approximate projections onto equality / inequality constraints without
    activating inactive sparse components.

    Args:
        bounds: Tensor of shape ``(2, d)``.
        comp_idx: Dimensions controlled by k-sparse support.
        k: Number of active dimensions inside ``comp_idx``.  ``k <= 0`` zeros all.
        score: Score used for support selection.
        equality_constraints: Constraints ``(idxs, coeffs, rhs)`` representing
            ``sum_j coeffs[j] * x[idxs[j]] = rhs``.
        inequality_constraints: Constraints ``(idxs, coeffs, rhs)``.
        inequality_sense: ``"le"`` means ``a^T x <= rhs``; ``"ge"`` means
            ``a^T x >= rhs``.
        fixed_features: Fixed feature values ``{dim: value}``.
        max_iters: Number of projection passes.
        support_selection: ``"topk"`` for deterministic support, ``"sample"`` for
            stochastic support.
        sample_tau: Sampling softmax temperature for ``support_selection="sample"``.
        sample_eps: Uniform mixture weight for stochastic support.
        generator: Optional torch generator for stochastic support.
    """
    comp_idx = [int(i) for i in comp_idx]
    equality_constraints = equality_constraints or []
    inequality_constraints = inequality_constraints or []
    fixed_features = {int(k_): float(v) for k_, v in (fixed_features or {}).items()}

    if bounds.ndim < 2 or bounds.shape[0] != 2:
        raise ValueError(f"bounds must have shape (2, d). Got {tuple(bounds.shape)}.")
    d = int(bounds.shape[-1])
    if any(i < 0 or i >= d for i in comp_idx):
        raise ValueError(f"comp_idx contains an out-of-range index for d={d}: {comp_idx}")
    if any(i < 0 or i >= d for i in fixed_features):
        raise ValueError(f"fixed_features contains an out-of-range index for d={d}: {fixed_features}")

    def repair(X: Tensor) -> Tensor:
        orig_shape = X.shape
        if orig_shape[-1] != d:
            raise ValueError(f"X last dim {orig_shape[-1]} does not match bounds d={d}.")

        device, dtype = X.device, X.dtype
        lower, upper = _normalize_bounds(bounds, d=d, device=device, dtype=dtype)
        Xf = X.reshape(-1, d).clone()

        # Initial clamp.
        Xf = Xf.clamp(min=lower, max=upper)

        idx_t: Optional[Tensor]
        support_mask: Optional[Tensor]
        if comp_idx:
            idx_t = torch.as_tensor(comp_idx, device=device, dtype=torch.long)
            group = Xf[:, idx_t]
            support_mask = _select_support_mask(
                group,
                k=k,
                score=score,
                support_selection=support_selection,
                sample_tau=sample_tau,
                sample_eps=sample_eps,
                generator=generator,
            )
            Xf[:, idx_t] = torch.where(support_mask, group, torch.zeros_like(group))
        else:
            idx_t = None
            support_mask = None

        fixed_idx = None
        if fixed_features:
            fixed_idx = torch.as_tensor(list(fixed_features.keys()), device=device, dtype=torch.long)
            for j, value in fixed_features.items():
                Xf[:, j] = torch.as_tensor(value, device=device, dtype=dtype)

        def allowed_coefficients(a: Tensor) -> Tensor:
            a_allowed = a.expand(Xf.shape[0], -1).clone()
            if fixed_idx is not None and fixed_idx.numel() > 0:
                a_allowed[:, fixed_idx] = 0.0
            if idx_t is not None and support_mask is not None:
                a_allowed[:, idx_t] = a_allowed[:, idx_t] * support_mask.to(dtype=dtype)
            return a_allowed

        for _ in range(max_iters):
            Xf = Xf.clamp(min=lower, max=upper)

            if fixed_features:
                for j, value in fixed_features.items():
                    Xf[:, j] = torch.as_tensor(value, device=device, dtype=dtype)

            if idx_t is not None and support_mask is not None:
                cur = Xf[:, idx_t]
                Xf[:, idx_t] = torch.where(support_mask, cur, torch.zeros_like(cur))

            # Equality projection: x <- x + ((rhs - a^T x) / ||a_allowed||^2) a_allowed
            for idxs, coeffs, rhs in equality_constraints:
                idxs_t = _to_1d_long_tensor(idxs, device=device)
                coeffs_t = _to_1d_value_tensor(coeffs, device=device, dtype=dtype)
                if idxs_t.numel() != coeffs_t.numel():
                    raise ValueError("Each equality constraint requires len(idxs) == len(coeffs).")
                a = torch.zeros(d, device=device, dtype=dtype)
                a[idxs_t] = coeffs_t
                a_allowed = allowed_coefficients(a)
                norm2 = (a_allowed * a_allowed).sum(dim=1)
                can = norm2 > 1e-12
                if not can.any():
                    continue
                resid = torch.as_tensor(rhs, device=device, dtype=dtype) - (Xf * a).sum(dim=1)
                step = torch.zeros_like(Xf)
                step[can] = (resid[can] / norm2[can]).unsqueeze(1) * a_allowed[can]
                Xf = Xf + step

            # Inequality halfspace projection.
            for idxs, coeffs, rhs in inequality_constraints:
                idxs_t = _to_1d_long_tensor(idxs, device=device)
                coeffs_t = _to_1d_value_tensor(coeffs, device=device, dtype=dtype)
                if idxs_t.numel() != coeffs_t.numel():
                    raise ValueError("Each inequality constraint requires len(idxs) == len(coeffs).")
                a = torch.zeros(d, device=device, dtype=dtype)
                a[idxs_t] = coeffs_t
                lhs = (Xf * a).sum(dim=1)
                rhs_t = torch.as_tensor(rhs, device=device, dtype=dtype)
                viol = lhs - rhs_t if inequality_sense == "le" else rhs_t - lhs
                bad = viol > 0
                if not bad.any():
                    continue
                a_allowed = allowed_coefficients(a)
                norm2 = (a_allowed * a_allowed).sum(dim=1)
                can = bad & (norm2 > 1e-12)
                if not can.any():
                    continue
                direction = a_allowed if inequality_sense == "le" else -a_allowed
                Xf[can] = Xf[can] - (viol[can] / norm2[can]).unsqueeze(1) * direction[can]

        Xf = Xf.clamp(min=lower, max=upper)
        if fixed_features:
            for j, value in fixed_features.items():
                Xf[:, j] = torch.as_tensor(value, device=device, dtype=dtype)
        if idx_t is not None and support_mask is not None:
            cur = Xf[:, idx_t]
            Xf[:, idx_t] = torch.where(support_mask, cur, torch.zeros_like(cur))

        return Xf.reshape(orig_shape)

    return repair


def _project_sum_box_1d(
    x: Tensor,
    lo: Tensor,
    hi: Tensor,
    rhs: float,
    *,
    n_bisect: int = 60,
) -> Tensor:
    """Euclidean projection onto ``sum(y)=rhs, lo<=y<=hi`` for one vector."""
    rhs_eff = min(max(float(rhs), float(lo.sum().item())), float(hi.sum().item()))
    lam_lo = float((lo - x).min().item())
    lam_hi = float((hi - x).max().item())

    for _ in range(n_bisect):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        y = torch.clamp(x + lam_mid, lo, hi)
        if float(y.sum().item()) < rhs_eff:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid
    return torch.clamp(x + lam_hi, lo, hi)


def enforce_sum_on_support(
    X: Tensor,
    sum_idx: Sequence[int],
    rhs: float,
    bounds: Tensor,
    *,
    support_eps: float = 1e-12,
) -> Tensor:
    """Adjust active dimensions inside ``sum_idx`` so their sum equals ``rhs``.

    Zero / inactive entries remain zero, so k-sparse support is preserved.
    Supports ``(d,)``, ``(q, d)``, and ``(..., q, d)`` tensors.
    """
    if not sum_idx:
        return X

    orig_shape = X.shape
    squeeze_1d = X.ndim == 1
    X_work = X.reshape(1, 1, -1) if squeeze_1d else X.reshape(-1, orig_shape[-2], orig_shape[-1])
    device, dtype = X_work.device, X_work.dtype
    d = X_work.shape[-1]
    lower, upper = _normalize_bounds(bounds, d=d, device=device, dtype=dtype)
    idx_t = torch.as_tensor([int(i) for i in sum_idx], device=device, dtype=torch.long)

    lo_all = lower[idx_t]
    hi_all = upper[idx_t]
    X_out = X_work.clone()

    for b in range(X_out.shape[0]):
        for i in range(X_out.shape[1]):
            group = X_out[b, i, idx_t]
            active = group.abs() > support_eps
            if not active.any():
                continue
            rhs_eff = float(rhs - group[~active].sum().item())
            projected = _project_sum_box_1d(group[active], lo_all[active], hi_all[active], rhs_eff)
            new_group = torch.zeros_like(group)
            new_group[active] = projected
            X_out[b, i, idx_t] = new_group

    if squeeze_1d:
        return X_out.reshape(-1)
    return X_out.reshape(orig_shape)


def diversify_within_q(
    X: Tensor,
    repair: Callable[[Tensor], Tensor],
    *,
    bounds: Optional[Tensor] = None,
    tol: Optional[float] = None,
    step: Optional[float] = None,
    mode: Literal["deterministic", "random"] = "deterministic",
    frozen_idx: Sequence[int] = (),
    comp_idx: Sequence[int] = (),
    active_eps: float = 0.0,
    max_tries: int = 3,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Diversify nearly duplicated candidates inside the q-batch and re-repair.

    This is useful after hard repair steps that can map multiple q-batch points to
    the same sparse support and values.
    """
    Xr = repair(X)
    if Xr.ndim < 2 or Xr.shape[-2] <= 1:
        return Xr

    d = Xr.shape[-1]
    q = Xr.shape[-2]
    orig_shape = Xr.shape
    Xb = Xr.reshape(-1, q, d)

    if bounds is not None:
        lower, upper = _normalize_bounds(bounds, d=d, device=X.device, dtype=X.dtype)
        span = (upper - lower).abs().clamp_min(torch.finfo(X.dtype).eps)
        scale = float(span.mean().item())
        tol = 1e-12 * max(scale, 1.0) if tol is None else tol
        step = 1e-6 * max(scale, 1.0) if step is None else step
    else:
        lower = upper = None
        tol = 1e-12 if tol is None else tol
        step = 1e-6 if step is None else step

    frozen = {int(i) for i in frozen_idx}
    allowed_dims = [j for j in range(d) if j not in frozen]
    if not allowed_dims:
        return Xr
    comp_list = [int(i) for i in comp_idx]
    eye = torch.eye(q, device=X.device, dtype=X.dtype).unsqueeze(0)

    for t in range(max_tries):
        dist = torch.cdist(Xb, Xb) + eye * 1e9
        dup = torch.tril(dist < tol, diagonal=-1).any(dim=-1)
        if not dup.any():
            return Xb.reshape(orig_shape)

        delta = torch.zeros_like(Xb)
        for i in range(1, q):
            rows_mask = dup[:, i]
            if not rows_mask.any():
                continue
            rows = torch.nonzero(rows_mask, as_tuple=False).squeeze(-1)
            nb = rows.numel()

            dim_choice: Optional[Tensor] = None
            if comp_list:
                comp_tensor = torch.as_tensor(comp_list, device=X.device, dtype=torch.long)
                vals = Xb[rows, i][:, comp_tensor].abs()
                active = vals > active_eps
                if active.any(dim=-1).all():
                    first_active = active.float().argmax(dim=-1)
                    chosen = comp_tensor[first_active]
                    if not any(int(c) in frozen for c in chosen.tolist()):
                        dim_choice = chosen
            if dim_choice is None:
                dim = allowed_dims[i % len(allowed_dims)]
                dim_choice = torch.full((nb,), dim, device=X.device, dtype=torch.long)

            if mode == "random":
                gen = generator or torch.Generator(device=X.device)
                if generator is None:
                    gen.manual_seed(0)
                step_vec = step * torch.randn((nb,), device=X.device, dtype=X.dtype, generator=gen)
            else:
                sign = -1.0 if (i % 2 == 0) else 1.0
                step_vec = torch.full((nb,), sign * step * (1.0 + 0.25 * t), device=X.device, dtype=X.dtype)
            delta[rows, i, dim_choice] = step_vec

        Xb = Xb + delta
        if lower is not None and upper is not None:
            Xb = Xb.clamp(min=lower, max=upper)
        Xb = repair(Xb.reshape(orig_shape)).reshape(-1, q, d)

    return Xb.reshape(orig_shape)


def make_k_sparse_post_processing_func(
    bounds: Tensor,
    comp_idx: Sequence[int],
    k: int,
    *,
    score: ScoreMode = "abs",
    equality_constraints: Optional[List[LinearConstraint]] = None,
    inequality_constraints: Optional[List[LinearConstraint]] = None,
    inequality_sense: Literal["le", "ge"] = "le",
    fixed_features: Optional[Dict[int, float]] = None,
    final_sum_constraint: Optional[Tuple[Sequence[int], float]] = None,
    diversify: bool = False,
    diversify_kwargs: Optional[Dict] = None,
    support_selection: SupportSelection = "topk",
    sample_tau: float = 0.2,
    sample_eps: float = 0.05,
    generator: Optional[torch.Generator] = None,
    max_iters: int = 12,
) -> Callable[[Tensor], Tensor]:
    """Create a complete post-processing function for optimizer wrappers."""
    repair = make_k_sparse_linear_constraints_repair(
        bounds=bounds,
        comp_idx=comp_idx,
        k=k,
        score=score,
        equality_constraints=equality_constraints,
        inequality_constraints=inequality_constraints,
        inequality_sense=inequality_sense,
        fixed_features=fixed_features,
        max_iters=max_iters,
        support_selection=support_selection,
        sample_tau=sample_tau,
        sample_eps=sample_eps,
        generator=generator,
    )
    diversify_kwargs = dict(diversify_kwargs or {})

    def post_process(X: Tensor) -> Tensor:
        Xp = repair(X)
        if diversify:
            Xp = diversify_within_q(
                Xp,
                repair=repair,
                bounds=bounds,
                frozen_idx=list((fixed_features or {}).keys()),
                comp_idx=comp_idx,
                **diversify_kwargs,
            )
        if final_sum_constraint is not None:
            sum_idx, rhs = final_sum_constraint
            Xp = enforce_sum_on_support(Xp, sum_idx=sum_idx, rhs=rhs, bounds=bounds)
            Xp = repair(Xp)
        return Xp

    return post_process


def generate_k_sparse_initial_conditions(
    bounds: Tensor,
    *,
    num_restarts: int,
    q: int,
    comp_idx: Sequence[int],
    k: int,
    fixed_features: Optional[Dict[int, float]] = None,
    equality_constraints: Optional[List[LinearConstraint]] = None,
    inequality_constraints: Optional[List[LinearConstraint]] = None,
    inequality_sense: Literal["le", "ge"] = "le",
    final_sum_constraint: Optional[Tuple[Sequence[int], float]] = None,
    score: ScoreMode = "abs",
    support_selection: SupportSelection = "topk",
    generator: Optional[torch.Generator] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Generate repaired initial conditions with shape ``(num_restarts, q, d)``."""
    device = device or bounds.device
    dtype = dtype or bounds.dtype
    bounds_t = bounds.to(device=device, dtype=dtype)
    d = bounds_t.shape[-1]
    lower, upper = _normalize_bounds(bounds_t, d=d, device=device, dtype=dtype)
    rand = torch.rand(num_restarts, q, d, device=device, dtype=dtype, generator=generator)
    X0 = lower + rand * (upper - lower)

    post = make_k_sparse_post_processing_func(
        bounds=bounds_t,
        comp_idx=comp_idx,
        k=k,
        score=score,
        equality_constraints=equality_constraints,
        inequality_constraints=inequality_constraints,
        inequality_sense=inequality_sense,
        fixed_features=fixed_features,
        final_sum_constraint=final_sum_constraint,
        support_selection=support_selection,
        generator=generator,
    )
    return post(X0)


def expand_categorical_features(
    categorical_features: Dict[int, Sequence[float]],
    *,
    base_fixed_features: Optional[Dict[int, float]] = None,
) -> List[Dict[int, float]]:
    """Expand ``{dim: values}`` categorical spec into BoTorch fixed_features_list."""
    base = {int(k): float(v) for k, v in (base_fixed_features or {}).items()}
    if not categorical_features:
        return [base]

    dims = [int(dim) for dim in categorical_features.keys()]
    values = [list(vs) for vs in categorical_features.values()]
    if any(len(vs) == 0 for vs in values):
        raise ValueError("categorical_features must not contain an empty value list.")

    fixed_features_list: List[Dict[int, float]] = []
    for combo in product(*values):
        item = dict(base)
        for dim, value in zip(dims, combo):
            item[dim] = float(value)
        fixed_features_list.append(item)
    return fixed_features_list
