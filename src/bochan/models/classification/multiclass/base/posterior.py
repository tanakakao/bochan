from __future__ import annotations

import torch

from bochan.models.classification.multiclass._components import MulticlassProbsPosterior
from bochan.models.multioutput.multiclass import (
    MultiOutputMulticlassProbsPosterior,
)

try:
    from botorch.sampling.get_sampler import GetSampler
    from botorch.sampling.normal import SobolQMCNormalSampler
except Exception:  # pragma: no cover - BoTorch version guard
    GetSampler = None  # type: ignore[assignment]
    SobolQMCNormalSampler = None  # type: ignore[assignment]


def _make_sobol_sampler(
    sample_shape: torch.Size,
    seed: int | None = None,
):
    if SobolQMCNormalSampler is None:
        raise NotImplementedError("SobolQMCNormalSampler is unavailable.")
    try:
        return SobolQMCNormalSampler(sample_shape=sample_shape, seed=seed)
    except TypeError:
        return SobolQMCNormalSampler(sample_shape=sample_shape)


if GetSampler is not None:

    @GetSampler.register(MultiOutputMulticlassProbsPosterior)
    def _get_multioutput_multiclass_sampler(
        posterior: MultiOutputMulticlassProbsPosterior,
        sample_shape: torch.Size,
        seed: int | None = None,
    ):
        return _make_sobol_sampler(sample_shape=sample_shape, seed=seed)

    @GetSampler.register(MulticlassProbsPosterior)
    def _get_single_multiclass_sampler(
        posterior: MulticlassProbsPosterior,
        sample_shape: torch.Size,
        seed: int | None = None,
    ):
        return _make_sobol_sampler(sample_shape=sample_shape, seed=seed)


__all__: list[str] = []
