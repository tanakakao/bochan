from __future__ import annotations

import torch
from torch import Tensor

from bochan.models.classification.multiclass.base.multioutput import MultiOutputMulticlassProbsPosterior
from bochan.models.components.multiclass import MulticlassProbsPosterior


def _as_sample_shape(sample_shape: torch.Size | None = None) -> torch.Size:
    return torch.Size() if sample_shape is None else torch.Size(sample_shape)


def _probability_to_objective_extended_shape(
    mean: Tensor,
    sample_shape: torch.Size | None = None,
) -> torch.Size:
    """Return BoTorch MO-compatible extended shape for multiclass probabilities.

    Multiclass probability posteriors expose ``mean`` with a final class dimension:

    - single-output: ``batch_shape x q x C``
    - multi-output: ``batch_shape x q x m x C``

    EHVI / NEHVI objectives convert this to objective values by reducing the
    class dimension, so the effective objective shape is:

    - single-output: ``batch_shape x q``
    - multi-output: ``batch_shape x q x m``

    BoTorch's NoisyExpectedHypervolumeMixin uses ``posterior._extended_shape()[-2]``
    to infer the baseline q-size. Returning the raw probability shape would make
    it read ``m`` as q for multi-output multiclass posteriors. Therefore this
    helper drops only the final class dimension.
    """
    if mean.ndim < 1:
        return _as_sample_shape(sample_shape)
    return _as_sample_shape(sample_shape) + torch.Size(mean.shape[:-1])


def _multioutput_multiclass_extended_shape(
    self: MultiOutputMulticlassProbsPosterior,
    sample_shape: torch.Size = torch.Size(),
) -> torch.Size:
    return _probability_to_objective_extended_shape(self.mean, sample_shape=sample_shape)


def _single_multiclass_extended_shape(
    self: MulticlassProbsPosterior,
    sample_shape: torch.Size = torch.Size(),
) -> torch.Size:
    return _probability_to_objective_extended_shape(self.mean, sample_shape=sample_shape)


def _multioutput_multiclass_batch_shape(self: MultiOutputMulticlassProbsPosterior) -> torch.Size:
    """Effective t-batch shape for BoTorch samplers.

    The raw probability shape is ``batch_shape x q x m x C``. BoTorch's MO
    samplers reason over objective samples ``batch_shape x q x m``. Therefore the
    batch shape is all dimensions before ``q x m x C``.
    """
    mean = self.mean
    if mean.ndim <= 3:
        return torch.Size()
    return torch.Size(mean.shape[:-3])


def _single_multiclass_batch_shape(self: MulticlassProbsPosterior) -> torch.Size:
    """Effective t-batch shape for single-output multiclass posteriors.

    The raw probability shape is ``batch_shape x q x C``. Objective samples are
    ``batch_shape x q`` after reducing the class dimension, so the batch shape is
    all dimensions before ``q x C``.
    """
    mean = self.mean
    if mean.ndim <= 2:
        return torch.Size()
    return torch.Size(mean.shape[:-2])


def apply_multiclass_posterior_compat() -> None:
    """Patch multiclass posterior wrappers for BoTorch sampler / NEHVI compatibility."""
    MultiOutputMulticlassProbsPosterior._extended_shape = _multioutput_multiclass_extended_shape  # type: ignore[method-assign]
    MulticlassProbsPosterior._extended_shape = _single_multiclass_extended_shape  # type: ignore[method-assign]

    # Some BoTorch samplers access posterior.batch_shape directly when updating
    # cached base samples for qNEHVI. The custom probability posteriors are not
    # GPyTorchPosterior subclasses, so provide the property explicitly.
    MultiOutputMulticlassProbsPosterior.batch_shape = property(_multioutput_multiclass_batch_shape)  # type: ignore[attr-defined]
    MulticlassProbsPosterior.batch_shape = property(_single_multiclass_batch_shape)  # type: ignore[attr-defined]


apply_multiclass_posterior_compat()


__all__ = ["apply_multiclass_posterior_compat"]
