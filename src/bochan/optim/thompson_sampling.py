"""Finite-candidate Thompson sampling optimizer wrappers.

These helpers intentionally mirror the public shape of ``optimize_acqf`` and
``optimize_acqf_mixed`` while using BoTorch's ``MaxPosteriorSampling`` instead
of gradient-based acquisition optimization.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import ceil
from typing import Any

import torch
from torch import Tensor

from botorch.generation import MaxPosteriorSampling
from botorch.utils.sampling import draw_sobol_samples

LinearConstraint = tuple[Tensor, Tensor, float]


def _resolve_model(acq_function: Any) -> Any:
    model = getattr(acq_function, "model", None)
    if model is not None:
        return model
    if hasattr(acq_function, "posterior"):
        return acq_function
    raise ValueError(
        "Could not resolve a model. Pass an acquisition function with a "
        "`model` attribute or pass the model itself as `acq_function`."
    )


def _model_device_dtype(model: Any, bounds: Tensor) -> tuple[torch.device, torch.dtype]:
    train_inputs = getattr(model, "train_inputs", None)
    if train_inputs:
        train_X = train_inputs[0]
        if torch.is_tensor(train_X):
            return train_X.device, train_X.dtype
    return bounds.device, bounds.dtype


def _flatten_initial_conditions(X: Tensor, d: int) -> Tensor:
    if X.ndim == 2 and X.shape[-1] == d:
        return X
    if X.ndim == 3 and X.shape[-1] == d:
        return X.reshape(-1, d)
    raise ValueError(
        "batch_initial_conditions must have shape [n, d] or "
        f"[num_restarts, q, d]. Got {tuple(X.shape)}."
    )


def _apply_fixed_features(X: Tensor, fixed_features: dict[int, float] | None) -> Tensor:
    if not fixed_features:
        return X
    X = X.clone()
    for index, value in fixed_features.items():
        X[..., int(index)] = torch.as_tensor(value, dtype=X.dtype, device=X.device)
    return X


def _apply_post_processing(
    X: Tensor,
    post_processing_func: Callable[[Tensor], Tensor] | None,
) -> Tensor:
    if post_processing_func is None:
        return X
    try:
        processed = post_processing_func(X.unsqueeze(-2))
        if processed.ndim == 3 and processed.shape[-2] == 1:
            return processed.squeeze(-2)
        return processed
    except (TypeError, ValueError, RuntimeError, IndexError):
        return post_processing_func(X)


def _filter_bounds(X: Tensor, bounds: Tensor) -> Tensor:
    mask = ((X >= bounds[0]) & (X <= bounds[1])).all(dim=-1)
    return X[mask]


def _filter_linear_constraints(
    X: Tensor,
    inequality_constraints: Sequence[LinearConstraint] | None,
    equality_constraints: Sequence[LinearConstraint] | None,
    tolerance: float,
) -> Tensor:
    if X.shape[0] == 0:
        return X

    mask = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
    for constraints, is_equality in (
        (inequality_constraints or [], False),
        (equality_constraints or [], True),
    ):
        for indices, coefficients, rhs in constraints:
            indices_t = torch.as_tensor(indices, dtype=torch.long, device=X.device)
            if indices_t.ndim != 1:
                raise NotImplementedError(
                    "Only intra-point linear constraints are supported. "
                    "Inter-point constraints use two-dimensional indices."
                )
            coefficients_t = torch.as_tensor(
                coefficients, dtype=X.dtype, device=X.device
            )
            rhs_t = torch.as_tensor(rhs, dtype=X.dtype, device=X.device)
            lhs = (X.index_select(-1, indices_t) * coefficients_t).sum(dim=-1)
            if is_equality:
                mask &= (lhs - rhs_t).abs() <= tolerance
            else:
                # BoTorch linear inequality convention: sum(a_i x_i) >= rhs.
                mask &= lhs >= rhs_t - tolerance
    return X[mask]


def _remove_duplicates(X: Tensor, tolerance: float) -> Tensor:
    if X.shape[0] <= 1:
        return X
    if tolerance <= 0:
        return torch.unique(X, dim=0)
    keys = torch.round(X / tolerance).to(torch.int64)
    seen: set[tuple[int, ...]] = set()
    keep: list[int] = []
    for row_idx, row in enumerate(keys.detach().cpu().tolist()):
        key = tuple(int(value) for value in row)
        if key not in seen:
            seen.add(key)
            keep.append(row_idx)
    return X[torch.as_tensor(keep, dtype=torch.long, device=X.device)]


def _prepare_candidate_pool(
    X: Tensor,
    bounds: Tensor,
    *,
    post_processing_func: Callable[[Tensor], Tensor] | None,
    inequality_constraints: Sequence[LinearConstraint] | None,
    equality_constraints: Sequence[LinearConstraint] | None,
    constraint_tolerance: float,
    duplicate_tolerance: float,
) -> Tensor:
    X = _apply_post_processing(X, post_processing_func)
    X = _filter_bounds(X, bounds)
    X = _filter_linear_constraints(
        X,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        tolerance=constraint_tolerance,
    )
    return _remove_duplicates(X, duplicate_tolerance)


def _select_with_max_posterior_sampling(
    *,
    acq_function: Any,
    X_candidates: Tensor,
    q: int,
    replacement: bool,
    observation_noise: bool | Tensor,
) -> tuple[Tensor, Tensor]:
    model = _resolve_model(acq_function)
    objective = getattr(acq_function, "objective", None)
    posterior_transform = getattr(acq_function, "posterior_transform", None)

    model.eval()
    likelihood = getattr(model, "likelihood", None)
    if likelihood is not None and hasattr(likelihood, "eval"):
        likelihood.eval()

    strategy = MaxPosteriorSampling(
        model=model,
        objective=objective,
        posterior_transform=posterior_transform,
        replacement=replacement,
    )

    with torch.no_grad():
        candidates = strategy(
            X_candidates,
            num_samples=q,
            observation_noise=observation_noise,
        )
        posterior = model.posterior(
            candidates,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
        )
        values = posterior.mean
        if values.ndim >= 2 and values.shape[-1] == 1:
            values = values.squeeze(-1)
    return candidates, values


def optimize_thompson_sampling(
    acq_function: Any,
    bounds: Tensor,
    q: int = 1,
    num_restarts: int = 10,
    raw_samples: int = 256,
    options: dict[str, Any] | None = None,
    inequality_constraints: Sequence[LinearConstraint] | None = None,
    equality_constraints: Sequence[LinearConstraint] | None = None,
    nonlinear_inequality_constraints: Any = None,
    fixed_features: dict[int, float] | None = None,
    post_processing_func: Callable[[Tensor], Tensor] | None = None,
    batch_initial_conditions: Tensor | None = None,
    return_best_only: bool = True,
    gen_candidates: Any = None,
    sequential: bool = False,
    **kwargs: Any,
) -> tuple[Tensor, Tensor]:
    """Select candidates by finite-pool Thompson sampling.

    ``num_restarts`` and ``raw_samples`` determine the default pool size
    ``max(num_restarts * raw_samples, 1024)``. The optional ``candidate_set``
    keyword can provide an explicit ``[n, d]`` pool.
    """
    del gen_candidates, sequential
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported keyword arguments: {unsupported}")
    if not return_best_only:
        raise NotImplementedError("return_best_only=False is not supported.")
    if nonlinear_inequality_constraints:
        raise NotImplementedError("Nonlinear constraints are not supported.")
    if q <= 0:
        raise ValueError(f"q must be positive. Got {q}.")

    options = dict(options or {})
    model = _resolve_model(acq_function)
    device, dtype = _model_device_dtype(model, bounds)
    bounds = torch.as_tensor(bounds, device=device, dtype=dtype)
    if bounds.ndim != 2 or bounds.shape[0] != 2:
        raise ValueError(f"bounds must have shape [2, d]. Got {tuple(bounds.shape)}.")

    candidate_set = options.pop("candidate_set", None)
    seed = options.pop("seed", None)
    replacement = bool(options.pop("replacement", False))
    observation_noise = options.pop("observation_noise", False)
    constraint_tolerance = float(options.pop("constraint_tolerance", 1e-6))
    duplicate_tolerance = float(options.pop("duplicate_tolerance", 1e-10))
    n_candidates = int(
        options.pop("n_candidates", max(int(num_restarts) * int(raw_samples), 1024))
    )
    if options:
        unsupported = ", ".join(sorted(options))
        raise TypeError(f"Unsupported options: {unsupported}")

    if candidate_set is None:
        X_candidates = draw_sobol_samples(
            bounds=bounds, n=n_candidates, q=1, seed=seed
        ).squeeze(-2)
    else:
        X_candidates = torch.as_tensor(candidate_set, device=device, dtype=dtype)
        if X_candidates.ndim != 2 or X_candidates.shape[-1] != bounds.shape[-1]:
            raise ValueError(
                "candidate_set must have shape [n_candidates, d]. "
                f"Got {tuple(X_candidates.shape)}."
            )

    X_candidates = _apply_fixed_features(X_candidates, fixed_features)
    if batch_initial_conditions is not None:
        X_initial = _flatten_initial_conditions(
            torch.as_tensor(batch_initial_conditions, device=device, dtype=dtype),
            bounds.shape[-1],
        )
        X_initial = _apply_fixed_features(X_initial, fixed_features)
        X_candidates = torch.cat([X_candidates, X_initial], dim=0)

    X_candidates = _prepare_candidate_pool(
        X_candidates,
        bounds,
        post_processing_func=post_processing_func,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        constraint_tolerance=constraint_tolerance,
        duplicate_tolerance=duplicate_tolerance,
    )
    if X_candidates.shape[0] < q:
        raise RuntimeError(
            "The valid Thompson candidate pool is smaller than q after repair, "
            f"constraint filtering, and deduplication: {X_candidates.shape[0]} < {q}."
        )

    return _select_with_max_posterior_sampling(
        acq_function=acq_function,
        X_candidates=X_candidates,
        q=q,
        replacement=replacement,
        observation_noise=observation_noise,
    )


def optimize_thompson_sampling_mixed(
    acq_function: Any,
    bounds: Tensor,
    fixed_features_list: list[dict[int, float]],
    q: int = 1,
    num_restarts: int = 10,
    raw_samples: int = 256,
    options: dict[str, Any] | None = None,
    inequality_constraints: Sequence[LinearConstraint] | None = None,
    equality_constraints: Sequence[LinearConstraint] | None = None,
    nonlinear_inequality_constraints: Any = None,
    post_processing_func: Callable[[Tensor], Tensor] | None = None,
    batch_initial_conditions: Tensor | None = None,
    return_best_only: bool = True,
    sequential: bool = False,
    **kwargs: Any,
) -> tuple[Tensor, Tensor]:
    """Mixed-input Thompson sampling over category-fixed Sobol pools.

    One Sobol pool is generated for every entry in ``fixed_features_list``.
    Category values are applied before repair and constraint filtering.
    """
    del sequential
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported keyword arguments: {unsupported}")
    if not fixed_features_list:
        raise ValueError("fixed_features_list must contain at least one combination.")
    if not return_best_only:
        raise NotImplementedError("return_best_only=False is not supported.")
    if nonlinear_inequality_constraints:
        raise NotImplementedError("Nonlinear constraints are not supported.")
    if q <= 0:
        raise ValueError(f"q must be positive. Got {q}.")

    options = dict(options or {})
    model = _resolve_model(acq_function)
    device, dtype = _model_device_dtype(model, bounds)
    bounds = torch.as_tensor(bounds, device=device, dtype=dtype)
    if bounds.ndim != 2 or bounds.shape[0] != 2:
        raise ValueError(f"bounds must have shape [2, d]. Got {tuple(bounds.shape)}.")

    candidate_set = options.pop("candidate_set", None)
    seed = options.pop("seed", None)
    replacement = bool(options.pop("replacement", False))
    observation_noise = options.pop("observation_noise", False)
    constraint_tolerance = float(options.pop("constraint_tolerance", 1e-6))
    duplicate_tolerance = float(options.pop("duplicate_tolerance", 1e-10))
    n_candidates = int(
        options.pop("n_candidates", max(int(num_restarts) * int(raw_samples), 1024))
    )
    if options:
        unsupported = ", ".join(sorted(options))
        raise TypeError(f"Unsupported options: {unsupported}")

    combinations = [dict(features) for features in fixed_features_list]
    if candidate_set is None:
        per_combination = max(1, ceil(n_candidates / len(combinations)))
        pools: list[Tensor] = []
        for combination_index, fixed_features in enumerate(combinations):
            combo_seed = None if seed is None else int(seed) + combination_index
            pool = draw_sobol_samples(
                bounds=bounds,
                n=per_combination,
                q=1,
                seed=combo_seed,
            ).squeeze(-2)
            pools.append(_apply_fixed_features(pool, fixed_features))
        X_candidates = torch.cat(pools, dim=0)
    else:
        base_pool = torch.as_tensor(candidate_set, device=device, dtype=dtype)
        if base_pool.ndim != 2 or base_pool.shape[-1] != bounds.shape[-1]:
            raise ValueError(
                "candidate_set must have shape [n_candidates, d]. "
                f"Got {tuple(base_pool.shape)}."
            )
        X_candidates = torch.cat(
            [_apply_fixed_features(base_pool, features) for features in combinations],
            dim=0,
        )

    if batch_initial_conditions is not None:
        X_initial = _flatten_initial_conditions(
            torch.as_tensor(batch_initial_conditions, device=device, dtype=dtype),
            bounds.shape[-1],
        )
        X_candidates = torch.cat([X_candidates, X_initial], dim=0)

    X_candidates = _prepare_candidate_pool(
        X_candidates,
        bounds,
        post_processing_func=post_processing_func,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        constraint_tolerance=constraint_tolerance,
        duplicate_tolerance=duplicate_tolerance,
    )
    if X_candidates.shape[0] < q:
        raise RuntimeError(
            "The valid mixed Thompson candidate pool is smaller than q after "
            f"repair, constraint filtering, and deduplication: {X_candidates.shape[0]} < {q}."
        )

    return _select_with_max_posterior_sampling(
        acq_function=acq_function,
        X_candidates=X_candidates,
        q=q,
        replacement=replacement,
        observation_noise=observation_noise,
    )


__all__ = [
    "optimize_thompson_sampling",
    "optimize_thompson_sampling_mixed",
]
