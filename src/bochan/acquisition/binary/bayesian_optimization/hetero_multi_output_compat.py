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
        resolved_constraints = constraints if constraints else None
        qExpectedHypervolumeImprovement.__init__(
            self,
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=resolved_sampler,
            objective=objective or IdentityMCMultiOutputObjective(),
            constraints=resolved_constraints,
            X_pending=X_pending,
            eta=eta,
            fat=fat,
        )
        # Some supported BoTorch versions only create these attributes when
        # constraints are configured. The inherited custom forward path uses
        # BoTorch's `_compute_qehvi`, so preserve the attributes explicitly when
        # they are needed without replacing any buffers already registered by
        # BoTorch.
        if resolved_constraints is not None and not hasattr(self, "eta"):
            self.eta = eta
        if not hasattr(self, "fat"):
            self.fat = bool(fat)

        self.beta = float(beta)
        self.noise_penalty = float(noise_penalty)
        self.default_sigma = float(default_sigma)
        self.noise_is_log_var = bool(noise_is_log_var)
        self.samples_are_probs = bool(samples_are_probs)
        self.apply_sigmoid_if_needed = bool(apply_sigmoid_if_needed)
        self.eps = float(eps)


__all__ = ["qHeteroMultiOutputBinaryExpectedHypervolumeImprovement"]
