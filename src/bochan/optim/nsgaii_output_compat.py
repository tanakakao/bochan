"""Shape compatibility helpers for deterministic NSGA-II evaluation."""

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


def reduce_nsgaii_model_sample_axes(values: Tensor, X: Tensor | None) -> Tensor:
    """Average DeepGP sample axes while preserving X batch and q axes.

    DeepGP posterior means can retain a leading likelihood-sample axis, for
    example ``(S, population, q, m)``. Pymoo requires one deterministic objective
    vector per population member. The raw input X identifies the public batch and
    q axes, so only dimensions before those axes are averaged.

    Some DeepGP / model-list posteriors omit a singleton q axis. When the reduced
    result contains only ``batch_shape + objective`` and ``q == 1``, the missing q
    dimension is restored before BoTorch converts the result for Pymoo.
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
        return values

    # Compatibility path for posteriors that omit q when q == 1.
    start = _find_last_contiguous_shape(value_shape, batch_shape)
    if start is None:
        return values
    if start > 0:
        values = values.mean(dim=tuple(range(start)))

    # Identity objectives may still expose ``batch_shape + m``. Restore q only
    # when exactly one event/objective axis remains, avoiding interference with
    # transformed q-like axes such as InputPerturbation's q * n_w dimension.
    if q == 1 and values.ndim == len(batch_shape) + 1:
        values = values.unsqueeze(len(batch_shape))
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
    """Apply an objective and collapse only leading DeepGP sample axes."""

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

        if self.objective is None:
            values = samples
        elif eval_X is not None:
            try:
                values = self.objective(samples, X=eval_X)
            except TypeError:
                values = self.objective(samples)
        else:
            values = self.objective(samples)

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
]
