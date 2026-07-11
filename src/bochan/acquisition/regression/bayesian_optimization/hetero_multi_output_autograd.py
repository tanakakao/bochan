"""Compatibility fixes for heteroscedastic multi-output regression acquisitions."""

from __future__ import annotations

import torch
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
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
from torch import Tensor

from . import hetero_multi_output as _hetero


class _AutogradSafeHeteroRegressionMCMultiOutputObjective(
    _hetero._HeteroRegressionMCMultiOutputObjective
):
    """Detach static baseline evaluations while preserving candidate gradients.

    qNEHVI evaluates its objective on ``X_baseline`` during construction and
    caches the resulting tensors. The heteroscedastic objective performs extra
    posterior calls, so those cached tensors can otherwise retain a graph through
    model parameters. Reusing that cache across optimizer steps then attempts a
    second backward pass through an already-freed graph.

    Baseline and no-grad evaluations do not need input gradients. Candidate
    evaluations receive an ``X`` tensor with ``requires_grad=True`` and retain
    the normal differentiable path used by acquisition optimization.
    """

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        if X is None:
            raise ValueError(
                "X must be provided for "
                "_AutogradSafeHeteroRegressionMCMultiOutputObjective."
            )

        track_candidate_grad = torch.is_grad_enabled() and bool(X.requires_grad)
        with torch.set_grad_enabled(track_candidate_grad):
            return super().forward(samples=samples, X=X)


class qHeteroMultiOutputRegressionExpectedHypervolumeImprovement(
    _hetero.qHeteroMultiOutputRegressionExpectedHypervolumeImprovement
):
    """Heteroscedastic qEHVI with BoTorch-version-safe constraint state.

    ``constraints=None`` must remain ``None``. Converting it to an empty list can
    leave ``self.constraints`` non-None while some BoTorch versions do not create
    the corresponding ``eta`` buffer for zero constraints. Then ``_compute_qehvi``
    enters its constrained path and raises ``AttributeError: ... has no attribute
    'eta'``.

    When actual outcome constraints are supplied, BoTorch owns the tensor
    conversion and registration of ``eta``. The support subclass therefore
    does not overwrite that buffer after initialization.
    """

    def __init__(
        self,
        model: Model,
        ref_point: Tensor | list[float],
        partitioning: FastNondominatedPartitioning,
        *,
        beta: float = 2.0,
        noise_penalty: float = 2.0,
        default_sigma: float = 0.0,
        noise_is_log_var: bool = True,
        sampler: SobolQMCNormalSampler | None = None,
        objective: MCMultiOutputObjective | None = None,
        constraints: list | None = None,
        X_pending: Tensor | None = None,
        eta: float | Tensor = 1e-3,
        fat: bool = False,
    ) -> None:
        qExpectedHypervolumeImprovement.__init__(
            self,
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            X_pending=X_pending,
            eta=eta,
            fat=fat,
        )
        self.beta_ht = float(beta)
        self.noise_penalty_ht = float(noise_penalty)
        self.default_sigma_ht = float(default_sigma)
        self.noise_is_log_var = bool(noise_is_log_var)


class qHeteroMultiOutputRegressionNoisyExpectedHypervolumeImprovement(
    _hetero.qHeteroMultiOutputRegressionNoisyExpectedHypervolumeImprovement
):
    """Heteroscedastic qNEHVI without reusable baseline autograd graphs."""

    def __init__(
        self,
        model: Model,
        ref_point: Tensor,
        X_baseline: Tensor,
        *,
        sampler: SobolQMCNormalSampler | None = None,
        objective: MCMultiOutputObjective | None = None,
        constraints: list | None = None,
        X_pending: Tensor | None = None,
        eta: float | Tensor = 1e-3,
        fat: bool = False,
        prune_baseline: bool = False,
        alpha: float = 0.0,
        cache_pending: bool = True,
        max_iep: int = 0,
        incremental_nehvi: bool = True,
        cache_root: bool = True,
        marginalize_dim: int | None = None,
        beta: float = 0.0,
        noise_penalty: float = 0.0,
        default_sigma: float = 0.0,
        noise_is_log_var: bool = True,
    ) -> None:
        base_objective = objective or IdentityMCMultiOutputObjective()
        hetero_objective = _AutogradSafeHeteroRegressionMCMultiOutputObjective(
            base_objective=base_objective,
            model=model,
            beta=beta,
            noise_penalty=noise_penalty,
            default_sigma=default_sigma,
            noise_is_log_var=noise_is_log_var,
        )
        qNoisyExpectedHypervolumeImprovement.__init__(
            self,
            model=model,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            objective=hetero_objective,
            constraints=constraints,
            X_pending=X_pending,
            eta=eta,
            fat=fat,
            prune_baseline=prune_baseline,
            alpha=alpha,
            cache_pending=cache_pending,
            max_iep=max_iep,
            incremental_nehvi=incremental_nehvi,
            cache_root=cache_root,
            marginalize_dim=marginalize_dim,
        )
        self.base_objective = base_objective
        self.hetero_objective = hetero_objective


__all__ = [
    "qHeteroMultiOutputRegressionExpectedHypervolumeImprovement",
    "qHeteroMultiOutputRegressionNoisyExpectedHypervolumeImprovement",
]
