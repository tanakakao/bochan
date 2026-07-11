"""Torch optimization strategy for correlated multitask acquisition models."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from .torch_opt import InequalitySense, LinearConstraint, TorchOptimizerName
from .torch_opt import optimize_acqf_torch as _optimize_acqf_torch

_WRAPPED_ACQF_ATTRIBUTES = (
    "base_acqf",
    "base_acquisition",
    "acq_function",
    "acquisition_function",
    "wrapped_acqf",
)


def _iter_acquisition_objects(acq_function: Any) -> Iterable[Any]:
    """Yield an acquisition and its common wrapper layers once each."""
    stack = [acq_function]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for name in _WRAPPED_ACQF_ATTRIBUTES:
            wrapped = getattr(current, name, None)
            if wrapped is not None:
                stack.append(wrapped)


def _is_correlated_multitask_model(model: Any) -> bool:
    """Return whether the model uses a joint task-covariance representation."""
    if model is None:
        return False
    class_name = type(model).__name__.lower()
    if "kronecker" in class_name or "multitask" in class_name:
        return True
    return hasattr(model, "task_covar_module") or hasattr(model, "task_feature")


def _uses_correlated_multitask_model(acq_function: AcquisitionFunction) -> bool:
    return any(
        _is_correlated_multitask_model(getattr(obj, "model", None))
        for obj in _iter_acquisition_objects(acq_function)
    )


def _initial_conditions_for_restart(
    batch_initial_conditions: Tensor | None,
    restart: int,
) -> Tensor | None:
    if batch_initial_conditions is None or batch_initial_conditions.ndim < 3:
        return batch_initial_conditions
    if batch_initial_conditions.shape[0] == 1:
        return batch_initial_conditions
    if restart >= batch_initial_conditions.shape[0]:
        return None
    return batch_initial_conditions[restart : restart + 1]


def optimize_acqf_torch(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int = 1,
    method: TorchOptimizerName = "adam",
    num_restarts: int = 10,
    raw_samples: int | None = 512,
    inequality_constraints: list[LinearConstraint] | None = None,
    equality_constraints: list[LinearConstraint] | None = None,
    fixed_features: dict[int, float] | None = None,
    post_processing_func: Callable[[Tensor], Tensor] | None = None,
    batch_initial_conditions: Tensor | None = None,
    return_best_only: bool = True,
    sequential: bool = False,
    options: dict | None = None,
    candidate_transform: Callable[[Tensor], Tensor] | None = None,
    X_pending: Tensor | None = None,
    inequality_sense: InequalitySense = "le",
) -> tuple[Tensor, Tensor]:
    """Optimize an acquisition, serializing restarts for correlated multitask models.

    Correlated multitask posteriors combine the candidate and task axes inside
    lazy linear-operator computations. Evaluating several optimizer restarts in
    one t-batch can therefore make the restart axis collide with the joint
    ``q * m`` event axis during backward. Each restart is optimized independently
    for these models, while every restart still evaluates its complete q-batch
    jointly.
    """
    if num_restarts <= 1 or not _uses_correlated_multitask_model(acq_function):
        return _optimize_acqf_torch(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            method=method,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            fixed_features=fixed_features,
            post_processing_func=post_processing_func,
            batch_initial_conditions=batch_initial_conditions,
            return_best_only=return_best_only,
            sequential=sequential,
            options=options,
            candidate_transform=candidate_transform,
            X_pending=X_pending,
            inequality_sense=inequality_sense,
        )

    candidates: list[Tensor] = []
    scores: list[Tensor] = []
    for restart in range(int(num_restarts)):
        candidate, value = _optimize_acqf_torch(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            method=method,
            num_restarts=1,
            raw_samples=raw_samples,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            fixed_features=fixed_features,
            post_processing_func=post_processing_func,
            batch_initial_conditions=_initial_conditions_for_restart(
                batch_initial_conditions,
                restart,
            ),
            return_best_only=return_best_only,
            sequential=sequential,
            options=options,
            candidate_transform=candidate_transform,
            X_pending=X_pending,
            inequality_sense=inequality_sense,
        )
        candidates.append(candidate.detach())
        scores.append(value.reshape(-1).mean().detach())

    stacked_scores = torch.stack(scores)
    best = int(torch.argmax(stacked_scores).item())
    return candidates[best], stacked_scores[best].reshape(1)


__all__ = ["optimize_acqf_torch"]
