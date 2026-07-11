"""Serialize torch-optimizer restart batches for correlated multitask models."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

import torch
from torch import Tensor

from . import torch_opt as _torch_opt

_WRAPPED_ACQF_ATTRIBUTES = (
    "base_acqf",
    "base_acquisition",
    "acq_function",
    "acquisition_function",
    "wrapped_acqf",
)


def _iter_acquisition_objects(acq_function: Any) -> Iterable[Any]:
    """Yield an acquisition and common wrapper layers without revisiting objects."""

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
    """Return whether a model uses a joint task covariance representation."""

    if model is None:
        return False
    class_name = type(model).__name__.lower()
    if "kronecker" in class_name or "multitask" in class_name:
        return True
    return hasattr(model, "task_covar_module") or hasattr(model, "task_feature")


def _requires_serialized_restart_batches(acq_function: Any, X: Tensor) -> bool:
    """Detect restart t-batches that can collide with joint task event axes."""

    if X.ndim != 3 or int(X.shape[0]) <= 1:
        return False
    return any(
        _is_correlated_multitask_model(getattr(obj, "model", None))
        for obj in _iter_acquisition_objects(acq_function)
    )


def _evaluate_independent_restarts(
    original: Callable[[Any, Tensor], Tensor],
    acq_function: Any,
    X: Tensor,
) -> Tensor:
    """Evaluate each restart as its own t-batch while preserving X gradients."""

    values = [original(acq_function, X[i : i + 1]) for i in range(int(X.shape[0]))]
    return torch.cat([value.reshape(1) for value in values], dim=0)


def apply_torch_multitask_tbatch_compat() -> None:
    """Patch torch optimization to avoid restart/task-axis autograd collisions.

    Correlated multitask posteriors flatten the ``q`` and task dimensions inside
    lazy linear operators. Evaluating several optimizer restarts in one t-batch
    can make LinearOperator's backward path confuse the restart axis with the
    flattened ``q * m`` event axis. BoTorch's SciPy path commonly limits those
    batches, whereas the custom torch optimizer evaluates all restarts together.

    Only correlated multitask acquisitions are serialized. Independent-output
    and ordinary single-output acquisitions retain the existing batched path.
    """

    current = _torch_opt._evaluate_acq_values
    if getattr(current, "_bochan_serializes_multitask_restarts", False):
        return

    original = current

    def _evaluate_acq_values(acq_function: Any, X: Tensor) -> Tensor:
        if _requires_serialized_restart_batches(acq_function, X):
            return _evaluate_independent_restarts(original, acq_function, X)
        return original(acq_function, X)

    _evaluate_acq_values._bochan_serializes_multitask_restarts = True  # type: ignore[attr-defined]
    _evaluate_acq_values._bochan_original = original  # type: ignore[attr-defined]
    _torch_opt._evaluate_acq_values = _evaluate_acq_values


__all__ = ["apply_torch_multitask_tbatch_compat"]
