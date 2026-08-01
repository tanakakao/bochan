"""Shape support helpers for deterministic NSGA-II evaluation."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def _find_last_contiguous_shape(
    shape: tuple[int, ...],
    target: tuple[int, ...],
) -> int | None:
    """Return the last start index where ``target`` occurs in ``shape``."""

    if len(target) == 0 or len(target) > len(shape):
        return None
    for start in range(len(shape) - len(target), -1, -1):
        if shape[start : start + len(target)] == target:
            return start
    return None


def restore_nsgaii_singleton_q_axis(values: Tensor, X: Tensor | None) -> Tensor:
    """Restore a missing ``q=1`` axis before BoTorch objective validation.

    ``MultiOutputPosteriorMean`` may return deterministic values with shape
    ``batch_shape x m`` for an input with shape ``batch_shape x 1 x d``. BoTorch
    multi-output objectives validate the q-axis before bochan can post-process
    their output, so the missing singleton axis must be restored before calling
    the objective.

    Expanded one-to-many outputs such as ``batch_shape x n_w x m`` are left
    unchanged so their dedicated perturbation objectives can aggregate them.
    """

    values = torch.as_tensor(values)
    if X is None or values.ndim == 0:
        return values

    X = torch.as_tensor(X)
    if X.ndim < 2 or int(X.shape[-2]) != 1:
        return values

    batch_shape = tuple(int(size) for size in X.shape[:-2])
    value_shape = tuple(int(size) for size in values.shape)

    if not batch_shape:
        if values.ndim == 1:
            return values.unsqueeze(-2)
        return values

    start = _find_last_contiguous_shape(value_shape, batch_shape)
    if start is None:
        return values

    q_axis = start + len(batch_shape)
    if len(value_shape) - q_axis != 1:
        return values
    return values.unsqueeze(q_axis)


def reduce_nsgaii_model_sample_axes(values: Tensor, X: Tensor | None) -> Tensor:
    """Return one deterministic objective vector per PyMOO population member.

    DeepGP posterior means can retain a leading likelihood-sample axis, for
    example ``(S, population, q, m)``. Those leading sample axes are averaged.

    BoTorch's NSGA-II bridge evaluates every population member with ``q=1`` but
    expects the objective returned to PyMOO to have shape ``population x m``.
    Therefore, a singleton q-axis used only for inner objective validation is
    removed again after the objective has been applied.
    """

    values = torch.as_tensor(values)
    if X is None or values.ndim == 0:
        return values

    X = torch.as_tensor(X)
    if X.ndim < 2:
        return values

    batch_shape = tuple(int(size) for size in X.shape[:-2])
    q = int(X.shape[-2])
    value_shape = tuple(int(size) for size in values.shape)

    public_shape = batch_shape + (q,)
    start = _find_last_contiguous_shape(value_shape, public_shape)
    if start is not None:
        if start > 0:
            values = values.mean(dim=tuple(range(start)))
        if q == 1:
            q_axis = len(batch_shape)
            if values.ndim > q_axis and int(values.shape[q_axis]) == 1:
                values = values.squeeze(q_axis)
        return values

    # Support deterministic posteriors that omit q, and DeepGP outputs shaped
    # ``sample_shape x batch_shape x m``.
    start = _find_last_contiguous_shape(value_shape, batch_shape)
    if start is None:
        return values
    if start > 0:
        values = values.mean(dim=tuple(range(start)))
    return values


class NSGAIIAcquisitionContextAdapter:
    """Record the X used for an NSGA-II acquisition evaluation."""

    def __init__(self, acq_function: Any) -> None:
        self.acq_function = acq_function
        self.last_X: Tensor | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.acq_function, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        X = kwargs.get("X")
        if X is None and len(args) > 0:
            X = args[0]
        if X is not None:
            self.last_X = torch.as_tensor(X)
        return self.acq_function(*args, **kwargs)


class NSGAIIObjectiveOutputAdapter:
    """Apply an objective and normalize its deterministic PyMOO output shape."""

    def __init__(
        self,
        objective: Any | None,
        acquisition_context: NSGAIIAcquisitionContextAdapter,
    ) -> None:
        self.objective = objective
        self.acquisition_context = acquisition_context
        self._verify_output_shape = False

    def __getattr__(self, name: str) -> Any:
        objective = object.__getattribute__(self, "objective")
        if objective is None:
            raise AttributeError(name)
        return getattr(objective, name)

    def __call__(
        self,
        samples: Tensor,
        X: Tensor | None = None,
    ) -> Tensor:
        eval_X = X
        if eval_X is None:
            eval_X = self.acquisition_context.last_X

        prepared_samples = restore_nsgaii_singleton_q_axis(samples, eval_X)
        if self.objective is None:
            values = prepared_samples
        elif eval_X is not None:
            try:
                values = self.objective(prepared_samples, X=eval_X)
            except TypeError:
                values = self.objective(prepared_samples)
        else:
            values = self.objective(prepared_samples)

        return reduce_nsgaii_model_sample_axes(values, eval_X)


def adapt_nsgaii_outputs(
    acq_function: Any,
    objective: Any | None,
) -> tuple[NSGAIIAcquisitionContextAdapter, NSGAIIObjectiveOutputAdapter]:
    """Return paired acquisition/objective adapters sharing evaluation context."""

    acquisition_context = NSGAIIAcquisitionContextAdapter(acq_function)
    objective_adapter = NSGAIIObjectiveOutputAdapter(
        objective,
        acquisition_context,
    )
    return acquisition_context, objective_adapter


__all__ = [
    "NSGAIIAcquisitionContextAdapter",
    "NSGAIIObjectiveOutputAdapter",
    "adapt_nsgaii_outputs",
    "reduce_nsgaii_model_sample_axes",
    "restore_nsgaii_singleton_q_axis",
]
