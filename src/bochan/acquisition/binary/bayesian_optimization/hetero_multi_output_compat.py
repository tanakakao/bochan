"""Compatibility fixes for heteroscedastic binary multi-output acquisitions."""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import Tensor

from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
    MCMultiOutputObjective,
)
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import (
    FastNondominatedPartitioning,
)

from .hetero_multi_output import (
    qHeteroMultiOutputBinaryExpectedHypervolumeImprovement as _BaseHeteroBinaryEHVI,
)


class qHeteroMultiOutputBinaryExpectedHypervolumeImprovement(
    _BaseHeteroBinaryEHVI
):
    """Binary heteroscedastic qEHVI with stable defaults."""

    def __init__(
        self,
        model: Model,
        ref_point: Union[Tensor, list[float]],
        partitioning: FastNondominatedPartitioning,
        *,
        beta: float = 1.0,
        noise_penalty: float = 0.3,
        default_sigma: float = 0.0,
        noise_is_log_var: bool = True,
        samples_are_probs: bool = False,
        apply_sigmoid_if_needed: bool = True,
        eps: float = 1e-6,
        sampler: Optional[SobolQMCNormalSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        constraints: Optional[list] = None,
        X_pending: Optional[Tensor] = None,
        eta: Union[float, Tensor] = 1e-3,
        fat: bool = False,
    ) -> None:
        resolved_sampler = sampler or SobolQMCNormalSampler(
            sample_shape=torch.Size([128])
        )
        qExpectedHypervolumeImprovement.__init__(
            self,
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=resolved_sampler,
            objective=objective or IdentityMCMultiOutputObjective(),
            constraints=constraints or [],
            X_pending=X_pending,
            eta=eta,
            fat=fat,
        )
        self.beta = float(beta)
        self.noise_penalty = float(noise_penalty)
        self.default_sigma = float(default_sigma)
        self.noise_is_log_var = bool(noise_is_log_var)
        self.samples_are_probs = bool(samples_are_probs)
        self.apply_sigmoid_if_needed = bool(apply_sigmoid_if_needed)
        self.eps = float(eps)


__all__ = ["qHeteroMultiOutputBinaryExpectedHypervolumeImprovement"]
