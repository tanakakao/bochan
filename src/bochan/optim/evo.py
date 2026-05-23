"""Non-gradient acquisition optimizers and k-sparse wrappers.

This module contains GA / PSO / SA / CMA-ES style optimizers.  Constraint logic
is intentionally imported from ``constraints.k_sparse`` and is not duplicated
here.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import torch
from torch import Tensor
from botorch.acquisition.acquisition import AcquisitionFunction

try:
    from ..constraints.k_sparse import (
        LinearConstraint,
        expand_categorical_features,
        make_k_sparse_post_processing_func,
    )
except ImportError:
    from constraints.k_sparse import (  # type: ignore
        LinearConstraint,
        expand_categorical_features,
        make_k_sparse_post_processing_func,
    )

MethodName = Literal["ga", "pso", "sa", "cmaes"]


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


def candidate_transform_mixed_factory(
    categorical_features: Dict[int, Sequence[float]],
    bounds: Tensor,
    *,
    base_transform: Optional[Callable[[Tensor], Tensor]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
) -> Callable[[Tensor], Tensor]:
    """Create a transform that rounds specified dimensions to category values.

    This is an evo-optimizer analogue of BoTorch's ``fixed_features_list`` mixed
    optimization.  It is useful when the optimizer operates in a continuous box
    but candidates must be evaluated at categorical values.
    """
    fixed_features = {int(k): float(v) for k, v in (fixed_features or {}).items()}
    d = bounds.shape[-1]
    cat_info: Dict[int, Tensor] = {}
    for dim, values in categorical_features.items():
        dim_i = int(dim)
        if dim_i < 0 or dim_i >= d:
            raise ValueError(f"categorical dim={dim_i} is out of range for d={d}.")
        vals = torch.as_tensor(list(values), dtype=bounds.dtype)
        if vals.numel() == 0:
            raise ValueError(f"categorical dim={dim_i} has no values.")
        cat_info[dim_i] = vals

    def transform(X: Tensor) -> Tensor:
        if base_transform is not None:
            X = base_transform(X)
        if not cat_info:
            return X

        X_new = X.clone()
        device, dtype = X_new.device, X_new.dtype
        for dim, vals_cpu in cat_info.items():
            if dim in fixed_features:
                continue
            vals = vals_cpu.to(device=device, dtype=dtype)
            raw = X_new[..., dim].unsqueeze(-1)  # (..., 1)
            dist = (raw - vals.view(*([1] * (raw.ndim - 1)), -1)).abs()
            idx = dist.argmin(dim=-1)
            X_new[..., dim] = vals[idx]
        return X_new

    return transform


def _linear_constraints_penalty(
    X: Tensor,
    *,
    inequality_constraints: Optional[List[LinearConstraint]],
    equality_constraints: Optional[List[LinearConstraint]],
    inequality_sense: Literal["le", "ge"],
    penalty_factor: float,
) -> Tensor:
    """Penalty for linear constraints, returned with shape ``(N,)``.

    X is expected to have shape ``(N, q, d)``.
    """
    device, dtype = X.device, X.dtype
    penalty = torch.zeros(X.shape[0], device=device, dtype=dtype)

    for idxs, coeffs, rhs in inequality_constraints or []:
        idx_t = torch.as_tensor(list(idxs), device=device, dtype=torch.long)
        coeff_t = torch.as_tensor(list(coeffs), device=device, dtype=dtype)
        vals = (X[..., idx_t] * coeff_t).sum(dim=-1)  # (N, q)
        if inequality_sense == "le":
            viol = torch.clamp(vals - rhs, min=0.0)
        else:
            viol = torch.clamp(rhs - vals, min=0.0)
        penalty = penalty + viol.max(dim=-1).values

    for idxs, coeffs, rhs in equality_constraints or []:
        idx_t = torch.as_tensor(list(idxs), device=device, dtype=torch.long)
        coeff_t = torch.as_tensor(list(coeffs), device=device, dtype=dtype)
        vals = (X[..., idx_t] * coeff_t).sum(dim=-1)  # (N, q)
        penalty = penalty + (vals - rhs).abs().max(dim=-1).values

    return penalty_factor * penalty


def _build_decode_and_evaluate(
    *,
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    fixed_features: Optional[Dict[int, float]],
    inequality_constraints: Optional[List[LinearConstraint]],
    equality_constraints: Optional[List[LinearConstraint]],
    inequality_sense: Literal["le", "ge"],
    candidate_transform: Optional[Callable[[Tensor], Tensor]],
    post_processing_func: Optional[Callable[[Tensor], Tensor]],
    apply_post_processing_during_eval: bool,
    penalty_factor: float,
):
    """Build decode and fitness evaluation functions shared by evo backends."""
    device, dtype = bounds.device, bounds.dtype
    d_total = bounds.shape[-1]
    fixed_features = {int(k): float(v) for k, v in (fixed_features or {}).items()}
    free_idx = [i for i in range(d_total) if i not in fixed_features]
    if not free_idx:
        raise ValueError("All features are fixed; nothing to optimize.")
    d_free = len(free_idx)
    lower_free = bounds[0, free_idx].to(device=device, dtype=dtype)
    upper_free = bounds[1, free_idx].to(device=device, dtype=dtype)

    def decode(z: Tensor) -> Tensor:
        z = z.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        x_free = lower_free + z * (upper_free - lower_free)
        *batch_shape, q_local, _ = z.shape
        X = torch.empty(*batch_shape, q_local, d_total, device=device, dtype=dtype)
        X[..., free_idx] = x_free
        for dim, value in fixed_features.items():
            X[..., dim] = torch.as_tensor(value, device=device, dtype=dtype)
        if candidate_transform is not None:
            X = candidate_transform(X)
        return X

    def evaluate_population(pop: Tensor) -> Tensor:
        X = decode(pop)  # (N, q, d)
        if post_processing_func is not None and apply_post_processing_during_eval:
            X = post_processing_func(X)

        with torch.no_grad():
            values = acq_function(X)
        if values.ndim == 0:
            values = values.reshape(1).repeat(pop.shape[0])
        elif values.ndim > 1:
            values = values.reshape(values.shape[0], -1).mean(dim=-1)

        penalty = _linear_constraints_penalty(
            X,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            inequality_sense=inequality_sense,
            penalty_factor=penalty_factor,
        )
        fitness = values - penalty
        fitness = torch.where(torch.isnan(fitness), torch.full_like(fitness, -float("inf")), fitness)
        return fitness

    return free_idx, decode, evaluate_population


def _normalize_initial_population(
    batch_initial_conditions: Tensor,
    *,
    q: int,
    d_free: int,
    bounds: Tensor,
) -> Tensor:
    """Validate normalized evo initial population ``(N, q, d_free)``."""
    z = batch_initial_conditions.to(device=bounds.device, dtype=bounds.dtype)
    if z.ndim == 2:
        z = z.unsqueeze(0)
    if z.ndim != 3 or z.shape[-2] != q or z.shape[-1] != d_free:
        raise ValueError(
            f"batch_initial_conditions for evo must be normalized z with shape "
            f"(N, q={q}, d_free={d_free}). Got {tuple(z.shape)}."
        )
    return z.clamp(0.0, 1.0)


def _optimize_acqf_ga_core(
    evaluate_population: Callable[[Tensor], Tensor],
    *,
    q: int,
    d_free: int,
    bounds: Tensor,
    batch_initial_conditions: Optional[Tensor],
    options: Dict,
) -> Tuple[Tensor, Tensor]:
    device, dtype = bounds.device, bounds.dtype
    pop_size = int(options.get("pop_size", 64))
    num_generations = int(options.get("num_generations", 100))
    elite_frac = float(options.get("elite_frac", 0.1))
    mutation_prob = float(options.get("mutation_prob", 0.1))
    mutation_std = float(options.get("mutation_std", 0.1))

    if batch_initial_conditions is not None:
        pop = _normalize_initial_population(batch_initial_conditions, q=q, d_free=d_free, bounds=bounds)
        pop_size = pop.shape[0]
    else:
        pop = torch.rand(pop_size, q, d_free, device=device, dtype=dtype)
    n_elite = max(1, int(pop_size * elite_frac))

    best_z = pop[0].detach().clone()
    best_val = torch.tensor(-float("inf"), device=device, dtype=dtype)

    for _ in range(num_generations):
        fitness = evaluate_population(pop)
        gen_best_val, gen_best_idx = fitness.max(dim=0)
        if gen_best_val > best_val:
            best_val = gen_best_val.detach().clone()
            best_z = pop[gen_best_idx].detach().clone()

        elite = pop[torch.topk(fitness, n_elite).indices]
        n_offspring = pop_size - n_elite
        idx = torch.randint(0, pop_size, (n_offspring * 2, 3), device=device)
        fit = fitness[idx]
        selected = pop[idx[torch.arange(n_offspring * 2, device=device), fit.argmax(dim=1)]]
        parents = selected.view(n_offspring, 2, q, d_free)
        mask = torch.rand(n_offspring, q, d_free, device=device, dtype=dtype) < 0.5
        offspring = torch.where(mask, parents[:, 0], parents[:, 1])
        mut_mask = torch.rand_like(offspring) < mutation_prob
        offspring = (offspring + mut_mask * torch.randn_like(offspring) * mutation_std).clamp(0.0, 1.0)
        pop = torch.cat([elite, offspring], dim=0)

    return best_z, best_val.view(1)


def _optimize_acqf_pso_core(
    evaluate_population: Callable[[Tensor], Tensor],
    *,
    q: int,
    d_free: int,
    bounds: Tensor,
    batch_initial_conditions: Optional[Tensor],
    options: Dict,
) -> Tuple[Tensor, Tensor]:
    device, dtype = bounds.device, bounds.dtype
    swarm_size = int(options.get("swarm_size", 64))
    num_iterations = int(options.get("num_iterations", 100))
    inertia = float(options.get("inertia", 0.7))
    c1 = float(options.get("c1", 1.5))
    c2 = float(options.get("c2", 1.5))

    if batch_initial_conditions is not None:
        pos = _normalize_initial_population(batch_initial_conditions, q=q, d_free=d_free, bounds=bounds)
        swarm_size = pos.shape[0]
    else:
        pos = torch.rand(swarm_size, q, d_free, device=device, dtype=dtype)
    vel = torch.zeros_like(pos)

    fitness = evaluate_population(pos)
    pbest_pos = pos.clone()
    pbest_fit = fitness.clone()
    gbest_fit, gbest_idx = fitness.max(dim=0)
    gbest_pos = pos[gbest_idx].clone()

    for _ in range(num_iterations):
        r1 = torch.rand_like(pos)
        r2 = torch.rand_like(pos)
        vel = inertia * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos.unsqueeze(0) - pos)
        pos = (pos + vel).clamp(0.0, 1.0)
        fitness = evaluate_population(pos)
        better = fitness > pbest_fit
        pbest_pos[better] = pos[better]
        pbest_fit[better] = fitness[better]
        cur_best_fit, cur_best_idx = fitness.max(dim=0)
        if cur_best_fit > gbest_fit:
            gbest_fit = cur_best_fit.detach().clone()
            gbest_pos = pos[cur_best_idx].detach().clone()

    return gbest_pos.detach().clone(), gbest_fit.detach().view(1)


def _optimize_acqf_sa_core(
    evaluate_population: Callable[[Tensor], Tensor],
    *,
    q: int,
    d_free: int,
    bounds: Tensor,
    batch_initial_conditions: Optional[Tensor],
    options: Dict,
) -> Tuple[Tensor, Tensor]:
    device, dtype = bounds.device, bounds.dtype
    n_steps = int(options.get("sa_steps", 500))
    init_temp = float(options.get("sa_init_temp", 1.0))
    final_temp = float(options.get("sa_final_temp", 1e-2))
    step_size = float(options.get("sa_step_size", 0.1))

    if batch_initial_conditions is not None:
        z_cur = _normalize_initial_population(batch_initial_conditions, q=q, d_free=d_free, bounds=bounds)[:1]
    else:
        z_cur = torch.rand(1, q, d_free, device=device, dtype=dtype)
    f_cur = evaluate_population(z_cur)[0]
    z_best = z_cur.clone()
    f_best = f_cur.clone()

    for step in range(n_steps):
        if n_steps <= 1:
            temp = final_temp
        else:
            temp = init_temp * (final_temp / init_temp) ** (step / (n_steps - 1))
        z_new = (z_cur + torch.randn_like(z_cur) * step_size).clamp(0.0, 1.0)
        f_new = evaluate_population(z_new)[0]
        delta = f_new - f_cur
        accept = bool(delta >= 0 or torch.rand((), device=device) < torch.exp(delta / max(temp, 1e-12)))
        if accept:
            z_cur, f_cur = z_new, f_new
        if f_new > f_best:
            z_best, f_best = z_new.clone(), f_new.clone()

    return z_best[0].detach().clone(), f_best.detach().view(1)


def _optimize_acqf_cmaes_core(
    evaluate_population: Callable[[Tensor], Tensor],
    *,
    q: int,
    d_free: int,
    bounds: Tensor,
    batch_initial_conditions: Optional[Tensor],
    options: Dict,
) -> Tuple[Tensor, Tensor]:
    if q != 1:
        raise NotImplementedError("CMA-ES backend supports q=1 only. Use sequential=True for q>1.")
    try:
        import cma
    except ImportError as exc:
        raise ImportError("CMA-ES backend requires `pip install cma`.") from exc

    device, dtype = bounds.device, bounds.dtype
    sigma0 = float(options.get("sigma0", 0.3))
    maxiter = int(options.get("maxiter", 200))

    if batch_initial_conditions is not None:
        z0 = _normalize_initial_population(batch_initial_conditions, q=1, d_free=d_free, bounds=bounds)[0, 0]
        x0 = z0.detach().cpu().tolist()
    else:
        x0 = [0.5] * d_free

    def objective(x_flat):
        z = torch.as_tensor(x_flat, device=device, dtype=dtype).view(1, 1, d_free).clamp(0.0, 1.0)
        return -float(evaluate_population(z)[0].item())

    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        {"bounds": [0.0, 1.0], "maxiter": maxiter, "verb_disp": 0},
    )
    while not es.stop():
        xs = es.ask()
        es.tell(xs, [objective(x) for x in xs])

    z_best = torch.as_tensor(es.best.x, device=device, dtype=dtype).view(1, 1, d_free).clamp(0.0, 1.0)
    best_val = evaluate_population(z_best)[0:1].detach().clone()
    return z_best[0].detach().clone(), best_val


def optimize_acqf_evo(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    *,
    q: int = 1,
    method: MethodName = "ga",
    inequality_constraints: Optional[List[LinearConstraint]] = None,
    equality_constraints: Optional[List[LinearConstraint]] = None,
    inequality_sense: Literal["le", "ge"] = "le",
    fixed_features: Optional[Dict[int, float]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
    candidate_transform: Optional[Callable[[Tensor], Tensor]] = None,
    batch_initial_conditions: Optional[Tensor] = None,
    return_best_only: bool = True,
    sequential: bool = False,
    options: Optional[Dict] = None,
    X_pending: Optional[Tensor] = None,
    apply_post_processing_during_eval: bool = True,
    repair_final_candidate: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Optimize an acquisition function using a non-gradient backend.

    ``batch_initial_conditions`` are normalized internal variables ``z`` with
    shape ``(N, q, d_free)`` in ``[0, 1]``.  This differs from BoTorch's standard
    initial condition convention.
    """
    if not return_best_only:
        raise NotImplementedError("return_best_only=False is not supported.")
    options = dict(options or {})
    method_l = method.lower()
    penalty_factor = float(options.get("penalty_factor", 1e3))

    base_X_pending = X_pending
    if base_X_pending is None and hasattr(acq_function, "X_pending"):
        base_X_pending = getattr(acq_function, "X_pending")

    def _run_single(q_local: int, X_pending_local: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        _set_X_pending_on_acqf(acq_function, X_pending_local)
        free_idx, decode, evaluate_population = _build_decode_and_evaluate(
            acq_function=acq_function,
            bounds=bounds,
            q=q_local,
            fixed_features=fixed_features,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            inequality_sense=inequality_sense,
            candidate_transform=candidate_transform,
            post_processing_func=post_processing_func,
            apply_post_processing_during_eval=apply_post_processing_during_eval,
            penalty_factor=penalty_factor,
        )
        d_free = len(free_idx)
        bic = batch_initial_conditions
        if bic is not None and sequential and q > 1 and q_local == 1 and bic.shape[-2] != 1:
            bic = None  # avoid shape mismatch in sequential one-point calls

        if method_l == "ga":
            best_z, best_val = _optimize_acqf_ga_core(
                evaluate_population, q=q_local, d_free=d_free, bounds=bounds, batch_initial_conditions=bic, options=options
            )
        elif method_l == "pso":
            best_z, best_val = _optimize_acqf_pso_core(
                evaluate_population, q=q_local, d_free=d_free, bounds=bounds, batch_initial_conditions=bic, options=options
            )
        elif method_l == "sa":
            best_z, best_val = _optimize_acqf_sa_core(
                evaluate_population, q=q_local, d_free=d_free, bounds=bounds, batch_initial_conditions=bic, options=options
            )
        elif method_l == "cmaes":
            best_z, best_val = _optimize_acqf_cmaes_core(
                evaluate_population, q=q_local, d_free=d_free, bounds=bounds, batch_initial_conditions=bic, options=options
            )
        else:
            raise ValueError(f"Unknown evo method: {method}")

        X_best = decode(best_z.unsqueeze(0))[0]
        if repair_final_candidate and post_processing_func is not None:
            X_best = post_processing_func(X_best.unsqueeze(0))[0]
        return X_best, best_val

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


def _merge_fixed_features(
    base: Optional[Dict[int, float]],
    extra: Optional[Dict[int, float]],
) -> Dict[int, float]:
    """Merge two fixed-feature dictionaries with ``extra`` taking priority."""
    merged = {int(k): float(v) for k, v in (base or {}).items()}
    for key, value in (extra or {}).items():
        merged[int(key)] = float(value)
    return merged


def _build_mixed_fixed_features_list(
    *,
    categorical_features: Optional[Dict[int, Sequence[float]]],
    fixed_features: Optional[Dict[int, float]],
    fixed_features_list: Optional[List[Dict[int, float]]],
    enumerate_categorical_features: bool,
) -> List[Dict[int, float]]:
    """Build fixed-feature combinations for mixed evo optimization.

    Priority:
      1. ``fixed_features`` are always applied as base fixed values.
      2. ``fixed_features_list`` is interpreted like BoTorch's mixed optimizer:
         each item is one categorical/fixed combination and is merged with the base.
      3. If ``fixed_features_list`` is omitted and ``categorical_features`` is
         supplied, categories are enumerated by default.
    """
    base = {int(k): float(v) for k, v in (fixed_features or {}).items()}
    categorical_features = dict(categorical_features or {})

    if fixed_features_list is not None:
        if len(fixed_features_list) == 0:
            raise ValueError("fixed_features_list must not be empty.")
        return [_merge_fixed_features(base, item) for item in fixed_features_list]

    if categorical_features and enumerate_categorical_features:
        return expand_categorical_features(
            categorical_features,
            base_fixed_features=base,
        )

    return [base]


def optimize_acqf_evo_mixed(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    *,
    q: int = 1,
    method: MethodName = "ga",
    categorical_features: Optional[Dict[int, Sequence[float]]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    fixed_features_list: Optional[List[Dict[int, float]]] = None,
    candidate_transform: Optional[Callable[[Tensor], Tensor]] = None,
    enumerate_categorical_features: bool = True,
    use_categorical_rounding_transform: Optional[bool] = None,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    """Evo optimizer for mixed continuous / categorical inputs.

    This function supports two mixed-input modes.

    1. Exhaustive fixed-feature mode, which mirrors BoTorch's
       ``optimize_acqf_mixed``.  Pass ``fixed_features_list`` directly, or pass
       ``categorical_features`` and keep ``enumerate_categorical_features=True``
       to enumerate all category combinations.

    2. Rounding-transform mode.  Set ``enumerate_categorical_features=False``
       or ``use_categorical_rounding_transform=True`` to optimize categorical
       dimensions as continuous variables and round them during evaluation.
       This is usually faster when the category grid is large, but it is less
       faithful to BoTorch's mixed optimizer.
    """
    categorical_features = dict(categorical_features or {})

    if use_categorical_rounding_transform is None:
        # When categories are enumerated, a rounding transform is unnecessary and
        # can conflict with fixed category values.  When enumeration is disabled,
        # rounding is the mixed-input mechanism.
        use_categorical_rounding_transform = bool(categorical_features) and not enumerate_categorical_features

    fixed_list = _build_mixed_fixed_features_list(
        categorical_features=categorical_features,
        fixed_features=fixed_features,
        fixed_features_list=fixed_features_list,
        enumerate_categorical_features=enumerate_categorical_features,
    )

    best_X: Optional[Tensor] = None
    best_val: Optional[Tensor] = None

    for fixed in fixed_list:
        mixed_transform = candidate_transform
        if use_categorical_rounding_transform and categorical_features:
            mixed_transform = candidate_transform_mixed_factory(
                categorical_features,
                bounds,
                base_transform=candidate_transform,
                fixed_features=fixed,
            )

        X, val = optimize_acqf_evo(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            method=method,
            fixed_features=fixed,
            candidate_transform=mixed_transform,
            **kwargs,
        )
        scalar_val = val.reshape(-1).sum()
        if best_val is None or scalar_val > best_val.reshape(-1).sum():
            best_X, best_val = X, val

    if best_X is None or best_val is None:
        raise RuntimeError("No mixed evo optimization run was executed.")
    return best_X, best_val


def optimize_acqf_evo_k_sparse(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    *,
    q: int = 1,
    method: MethodName = "ga",
    comp_idx: Sequence[int],
    k: int,
    inequality_constraints: Optional[List[LinearConstraint]] = None,
    equality_constraints: Optional[List[LinearConstraint]] = None,
    inequality_sense: Literal["le", "ge"] = "le",
    fixed_features: Optional[Dict[int, float]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
    final_sum_constraint: Optional[Tuple[Sequence[int], float]] = None,
    diversify: bool = False,
    diversify_kwargs: Optional[Dict] = None,
    support_selection: str = "topk",
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    """Evo optimizer with k-sparse repair applied during evaluation and final return."""
    k_sparse_post = make_k_sparse_post_processing_func(
        bounds=bounds,
        comp_idx=comp_idx,
        k=k,
        equality_constraints=equality_constraints,
        inequality_constraints=inequality_constraints,
        inequality_sense=inequality_sense,
        fixed_features=fixed_features,
        final_sum_constraint=final_sum_constraint,
        diversify=diversify,
        diversify_kwargs=diversify_kwargs,
        support_selection=support_selection,  # type: ignore[arg-type]
    )
    post = _compose_post_processing(post_processing_func, k_sparse_post)
    return optimize_acqf_evo(
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        method=method,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        inequality_sense=inequality_sense,
        fixed_features=fixed_features,
        post_processing_func=post,
        apply_post_processing_during_eval=True,
        repair_final_candidate=True,
        **kwargs,
    )


def optimize_acqf_evo_mixed_k_sparse(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    *,
    q: int = 1,
    method: MethodName = "ga",
    comp_idx: Sequence[int],
    k: int,
    categorical_features: Optional[Dict[int, Sequence[float]]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    fixed_features_list: Optional[List[Dict[int, float]]] = None,
    inequality_constraints: Optional[List[LinearConstraint]] = None,
    equality_constraints: Optional[List[LinearConstraint]] = None,
    inequality_sense: Literal["le", "ge"] = "le",
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
    candidate_transform: Optional[Callable[[Tensor], Tensor]] = None,
    final_sum_constraint: Optional[Tuple[Sequence[int], float]] = None,
    diversify: bool = False,
    diversify_kwargs: Optional[Dict] = None,
    support_selection: str = "topk",
    allow_sparse_on_categorical: bool = False,
    enumerate_categorical_features: bool = True,
    use_categorical_rounding_transform: Optional[bool] = None,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    """Mixed evo optimizer with k-sparse repair.

    For evo mixed optimization, ``categorical_features`` is used as a rounding
    transform.  If ``fixed_features_list`` is supplied, this function evaluates
    each fixed category combination and returns the best result.
    """
    fixed_features = dict(fixed_features or {})
    categorical_features = dict(categorical_features or {})

    fixed_dims = set(categorical_features.keys()) | set(fixed_features.keys())
    if fixed_features_list is not None:
        for item in fixed_features_list:
            fixed_dims.update(item.keys())
    overlap = set(int(i) for i in comp_idx) & {int(i) for i in fixed_dims}
    if overlap and not allow_sparse_on_categorical:
        raise ValueError(
            "comp_idx overlaps with categorical/fixed dimensions. "
            f"overlap={sorted(overlap)}. Remove these dims from comp_idx or pass "
            "allow_sparse_on_categorical=True."
        )

    fixed_features_list = _build_mixed_fixed_features_list(
        categorical_features=categorical_features,
        fixed_features=fixed_features,
        fixed_features_list=fixed_features_list,
        enumerate_categorical_features=enumerate_categorical_features,
    )

    best_X: Optional[Tensor] = None
    best_val: Optional[Tensor] = None
    for fixed in fixed_features_list:
        # Build the repair per fixed-feature combination so that categorical /
        # fixed dimensions are preserved even if a user-supplied post-processing
        # function modifies them.
        k_sparse_post = make_k_sparse_post_processing_func(
            bounds=bounds,
            comp_idx=comp_idx,
            k=k,
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
            inequality_sense=inequality_sense,
            fixed_features=fixed,
            final_sum_constraint=final_sum_constraint,
            diversify=diversify,
            diversify_kwargs=diversify_kwargs,
            support_selection=support_selection,  # type: ignore[arg-type]
        )
        post = _compose_post_processing(post_processing_func, k_sparse_post)

        X, val = optimize_acqf_evo_mixed(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            method=method,
            categorical_features=categorical_features,
            fixed_features=fixed,
            candidate_transform=candidate_transform,
            fixed_features_list=[fixed],
            enumerate_categorical_features=False,
            use_categorical_rounding_transform=use_categorical_rounding_transform,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            inequality_sense=inequality_sense,
            post_processing_func=post,
            apply_post_processing_during_eval=True,
            repair_final_candidate=True,
            **kwargs,
        )
        scalar_val = val.reshape(-1).sum()
        if best_val is None or scalar_val > best_val.reshape(-1).sum():
            best_X, best_val = X, val

    if best_X is None or best_val is None:
        raise RuntimeError("No mixed evo optimization run was executed.")
    return best_X, best_val
