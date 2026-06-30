"""Compatibility fixes for heteroscedastic multi-output regression acquisitions."""

from __future__ import annotations

from typing import Optional, Union

from torch import Tensor

from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import (
    FastNondominatedPartitioning,
)

from .hetero_multi_output import (
    qHeteroMultiOutputRegressionExpectedHypervolumeImprovement as _BaseHeteroEHVI,
)


class qHeteroMultiOutputRegressionExpectedHypervolumeImprovement(_BaseHeteroEHVI):
    """Heteroscedastic qEHVI without overwriting BoTorch's ``eta`` buffer.

    BoTorch registers ``eta`` as a tensor buffer in
    ``qExpectedHypervolumeImprovement``. The previous implementation assigned the
    original float to ``self.eta`` after ``super().__init__()``, which raises a
    ``TypeError`` whenever outcome constraints are enabled.
    """

    def __init__(
        self,
        model: Model,
        ref_point: Union[Tensor, list[float]],
        partitioning: FastNondominatedPartitioning,
        *,
        beta: float = 2.0,
        noise_penalty: float = 2.0,
        default_sigma: float = 0.0,
        noise_is_log_var: bool = True,
        sampler: Optional[SobolQMCNormalSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        constraints: Optional[list] = None,
        X_pending: Optional[Tensor] = None,
        eta: Union[float, Tensor] = 1e-3,
        fat: bool = False,
    ) -> None:
        qExpectedHypervolumeImprovement.__init__(
            self,
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            objective=objective,
            constraints=constraints or [],
            X_pending=X_pending,
            eta=eta,
            fat=fat,
        )
        self.beta_ht = float(beta)
        self.noise_penalty_ht = float(noise_penalty)
        self.default_sigma_ht = float(default_sigma)
        self.noise_is_log_var = bool(noise_is_log_var)


__all__ = ["qHeteroMultiOutputRegressionExpectedHypervolumeImprovement"]
