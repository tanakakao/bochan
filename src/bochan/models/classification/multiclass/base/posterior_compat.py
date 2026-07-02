from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
from torch import Tensor

from bochan.models.classification.multiclass.base.kronecker_multitask import (
    KroneckerMultiTaskMulticlassClassificationGPModel,
    KroneckerMultiTaskMulticlassProbsPosterior,
)
from bochan.models.classification.multiclass.base.multioutput import (
    MultiOutputMulticlassProbsPosterior,
)
from bochan.models.components.multiclass import MulticlassProbsPosterior

try:
    from botorch.sampling.get_sampler import GetSampler
    from botorch.sampling.normal import SobolQMCNormalSampler
except Exception:  # pragma: no cover - BoTorch import compatibility guard
    GetSampler = None  # type: ignore[assignment]
    SobolQMCNormalSampler = None  # type: ignore[assignment]


_KRONECKER_ORIGINAL_POSTERIOR_ATTR = "_bochan_original_posterior_before_q1_compat"
_KRONECKER_ORIGINAL_LOGITS_ATTR = "_bochan_original_probability_logits_before_q1_compat"


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
    return _probability_to_objective_extended_shape(
        self.mean,
        sample_shape=sample_shape,
    )


def _single_multiclass_extended_shape(
    self: MulticlassProbsPosterior,
    sample_shape: torch.Size = torch.Size(),
) -> torch.Size:
    return _probability_to_objective_extended_shape(
        self.mean,
        sample_shape=sample_shape,
    )


def _multioutput_multiclass_batch_shape(
    self: MultiOutputMulticlassProbsPosterior,
) -> torch.Size:
    """Effective t-batch shape for BoTorch samplers.

    The raw probability shape is ``batch_shape x q x m x C``. BoTorch's MO
    samplers reason over objective samples ``batch_shape x q x m``. Therefore the
    batch shape is all dimensions before ``q x m x C``.
    """
    mean = self.mean
    if mean.ndim <= 3:
        return torch.Size()
    return torch.Size(mean.shape[:-3])


def _single_multiclass_batch_shape(
    self: MulticlassProbsPosterior,
) -> torch.Size:
    """Effective t-batch shape for single-output multiclass posteriors.

    The raw probability shape is ``batch_shape x q x C``. Objective samples are
    ``batch_shape x q`` after reducing the class dimension, so the batch shape is
    all dimensions before ``q x C``.
    """
    mean = self.mean
    if mean.ndim <= 2:
        return torch.Size()
    return torch.Size(mean.shape[:-2])


def _input_batch_shape_and_q(X: Tensor) -> tuple[torch.Size, int]:
    """Resolve the public candidate layout represented by ``X``."""
    X = torch.as_tensor(X)
    if X.ndim == 1:
        return torch.Size(), 1
    if X.ndim == 2:
        return torch.Size(), int(X.shape[-2])
    return torch.Size(X.shape[:-2]), int(X.shape[-2])


def _shape_endswith(shape: Sequence[int], suffix: Sequence[int]) -> bool:
    if len(suffix) == 0:
        return True
    if len(shape) < len(suffix):
        return False
    return tuple(shape[-len(suffix) :]) == tuple(suffix)


def _kronecker_probability_logits_with_q1(
    self: KroneckerMultiTaskMulticlassProbsPosterior,
    latent: Tensor,
) -> Tensor:
    """Restore the candidate singleton omitted by GPyTorch when ``q == 1``.

    The correlated Kronecker latent distribution can expose ``batch x C x m``
    instead of ``batch x C x 1 x m`` for a one-point candidate batch. After the
    class axis is moved to the end, this becomes ``batch x m x C`` and a
    multi-output objective mistakes the t-batch axis for q. The model posterior
    records the public input layout, allowing this method to restore exactly the
    missing candidate axis without changing normal ``q > 1`` layouts.
    """
    original = getattr(
        type(self),
        _KRONECKER_ORIGINAL_LOGITS_ATTR,
    )
    logits = original(self, latent)

    expected_q = getattr(self, "_bochan_expected_q", None)
    batch_shape = torch.Size(
        getattr(self, "_bochan_expected_batch_shape", torch.Size())
    )
    if expected_q != 1:
        return logits

    num_outputs = (
        len(self.output_indices)
        if self.output_indices is not None
        else int(logits.shape[-2])
    )
    with_q_suffix = tuple(batch_shape) + (
        1,
        num_outputs,
        self.num_classes,
    )
    if _shape_endswith(logits.shape, with_q_suffix):
        return logits

    without_q_suffix = tuple(batch_shape) + (
        num_outputs,
        self.num_classes,
    )
    if _shape_endswith(logits.shape, without_q_suffix):
        return logits.unsqueeze(-3)
    return logits


def _kronecker_posterior_with_input_layout(
    self: KroneckerMultiTaskMulticlassClassificationGPModel,
    X: Tensor,
    output_indices: Optional[Sequence[int]] = None,
    observation_noise: bool | Tensor = False,
    posterior_transform: Any = None,
    **kwargs: Any,
):
    """Attach the raw t-batch and q layout to the Kronecker probability posterior."""
    original = getattr(
        type(self),
        _KRONECKER_ORIGINAL_POSTERIOR_ATTR,
    )
    posterior = original(
        self,
        X,
        output_indices=output_indices,
        observation_noise=observation_noise,
        posterior_transform=posterior_transform,
        **kwargs,
    )
    if isinstance(posterior, KroneckerMultiTaskMulticlassProbsPosterior):
        batch_shape, q = _input_batch_shape_and_q(torch.as_tensor(X))
        posterior._bochan_expected_batch_shape = batch_shape
        posterior._bochan_expected_q = q
    return posterior


def _patch_kronecker_q1_shape() -> None:
    """Patch correlated multiclass posteriors to preserve an explicit q axis."""
    model_cls = KroneckerMultiTaskMulticlassClassificationGPModel
    posterior_cls = KroneckerMultiTaskMulticlassProbsPosterior

    if not hasattr(model_cls, _KRONECKER_ORIGINAL_POSTERIOR_ATTR):
        setattr(
            model_cls,
            _KRONECKER_ORIGINAL_POSTERIOR_ATTR,
            model_cls.posterior,
        )
        model_cls.posterior = _kronecker_posterior_with_input_layout  # type: ignore[method-assign]

    if not hasattr(posterior_cls, _KRONECKER_ORIGINAL_LOGITS_ATTR):
        setattr(
            posterior_cls,
            _KRONECKER_ORIGINAL_LOGITS_ATTR,
            posterior_cls._probability_logits,
        )
        posterior_cls._probability_logits = _kronecker_probability_logits_with_q1  # type: ignore[method-assign]


def _make_sobol_sampler(sample_shape: torch.Size, seed: int | None = None):
    if SobolQMCNormalSampler is None:
        raise NotImplementedError("SobolQMCNormalSampler is unavailable.")
    try:
        return SobolQMCNormalSampler(
            sample_shape=sample_shape,
            seed=seed,
        )
    except TypeError:
        # Older BoTorch versions may not expose the seed keyword.
        return SobolQMCNormalSampler(sample_shape=sample_shape)


if GetSampler is not None:

    @GetSampler.register(MultiOutputMulticlassProbsPosterior)
    def _get_multioutput_multiclass_sampler(
        posterior: MultiOutputMulticlassProbsPosterior,
        sample_shape: torch.Size,
        seed: int | None = None,
    ):
        """Return a normal MC sampler for multi-output multiclass probability posterior.

        ``MultiOutputMulticlassProbsPosterior`` implements ``rsample`` and
        ``rsample_from_base_samples``, so BoTorch's ``SobolQMCNormalSampler`` can
        be used. Registering this function prevents ``get_sampler`` from falling
        through to the generic ``NotImplementedError`` path in qEHVI/qNEHVI.
        """
        return _make_sobol_sampler(
            sample_shape=sample_shape,
            seed=seed,
        )

    @GetSampler.register(MulticlassProbsPosterior)
    def _get_single_multiclass_sampler(
        posterior: MulticlassProbsPosterior,
        sample_shape: torch.Size,
        seed: int | None = None,
    ):
        """Return a normal MC sampler for single-output multiclass probability posterior."""
        return _make_sobol_sampler(
            sample_shape=sample_shape,
            seed=seed,
        )


def apply_multiclass_posterior_compat() -> None:
    """Patch multiclass posterior wrappers for BoTorch sampler / NEHVI compatibility."""
    _patch_kronecker_q1_shape()

    MultiOutputMulticlassProbsPosterior._extended_shape = _multioutput_multiclass_extended_shape  # type: ignore[method-assign]
    MulticlassProbsPosterior._extended_shape = _single_multiclass_extended_shape  # type: ignore[method-assign]

    # Some BoTorch samplers access posterior.batch_shape directly when updating
    # cached base samples for qNEHVI. The custom probability posteriors are not
    # GPyTorchPosterior subclasses, so provide the property explicitly.
    MultiOutputMulticlassProbsPosterior.batch_shape = property(_multioutput_multiclass_batch_shape)  # type: ignore[attr-defined]
    MulticlassProbsPosterior.batch_shape = property(_single_multiclass_batch_shape)  # type: ignore[attr-defined]


apply_multiclass_posterior_compat()


__all__ = ["apply_multiclass_posterior_compat"]
