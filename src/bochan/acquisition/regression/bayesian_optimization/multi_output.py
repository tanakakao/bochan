from __future__ import annotations

"""Multi-output regression Bayesian optimization acquisitions."""

from typing import Optional, Union

import torch
from torch import Tensor

from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement as qMultiOutputRegressionExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement as qMultiOutputRegressionNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
    MCMultiOutputObjective,
)
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.objective import compute_smoothed_feasibility_indicator
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform

try:
    from botorch.acquisition.multi_objective.monte_carlo import (
        qLogExpectedHypervolumeImprovement as qMultiOutputRegressionLogExpectedHypervolumeImprovement,
        qLogNoisyExpectedHypervolumeImprovement as qMultiOutputRegressionLogNoisyExpectedHypervolumeImprovement,
    )
except Exception:  # pragma: no cover
    qMultiOutputRegressionLogExpectedHypervolumeImprovement = None
    qMultiOutputRegressionLogNoisyExpectedHypervolumeImprovement = None


class qMultiOutputRegressionNParEGO(MCAcquisitionFunction):
    """Multi-output regression NParEGO acquisition.

    ``objective`` is treated as a multi-output preprocessing objective. For
    example, ``RegressionLinearMCObjective`` can select outputs, align maximize /
    minimize directions, apply output scales, and convert equality targets. The
    resulting objective values are then scalarized internally with an augmented
    Chebyshev scalarization and evaluated using qEI-style improvement.

    Args:
        model: BoTorch-compatible multi-output model.
        X_baseline: Existing evaluated inputs used to determine ``best_value``.
        ref_point: Reference point in the transformed objective space.
        weights: NParEGO scalarization weights. Random normalized weights are
            generated when omitted.
        sampler: Posterior sampler. Defaults to 128 Sobol QMC samples.
        objective: Optional multi-output preprocessing objective. Identity is
            used when omitted.
        constraints: BoTorch outcome constraints applied to raw posterior
            samples. A constraint is feasible when its value is ``<= 0``.
        X_pending: Pending candidates appended during acquisition evaluation.
        eta: Constraint smoothing temperature.
        fat: Whether to use fat-tailed smooth feasibility approximations.
        rho: Augmentation coefficient for Chebyshev scalarization.
    """

    def __init__(
        self,
        model: Model,
        X_baseline: Tensor,
        ref_point: Union[Tensor, list[float]],
        *,
        weights: Optional[Tensor] = None,
        sampler: Optional[SobolQMCNormalSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        constraints: Optional[list] = None,
        X_pending: Optional[Tensor] = None,
        eta: Union[float, Tensor] = 1e-3,
        fat: bool = False,
        rho: float = 0.05,
    ) -> None:
        sampler = sampler or SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        base_objective = objective or IdentityMCMultiOutputObjective()
        super().__init__(model=model, sampler=sampler, objective=base_objective)

        if not torch.is_tensor(X_baseline):
            raise TypeError(f"X_baseline must be a Tensor. Got {type(X_baseline)}.")
        if X_baseline.ndim < 2:
            raise ValueError(
                "X_baseline must have shape n x d or batch_shape x n x d. "
                f"Got {tuple(X_baseline.shape)}."
            )
        if rho < 0.0:
            raise ValueError("rho must be non-negative.")

        self.X_baseline = X_baseline.detach()
        self.base_objective = base_objective
        self.constraints = list(constraints or [])
        self.eta = eta
        self.fat = bool(fat)
        self.rho = float(rho)
        self.set_X_pending(X_pending)

        tkwargs = {
            "device": self.X_baseline.device,
            "dtype": self.X_baseline.dtype,
        }
        ref = torch.as_tensor(ref_point, **tkwargs).reshape(-1)
        if ref.numel() == 0:
            raise ValueError("ref_point must contain at least one value.")

        with torch.no_grad():
            posterior = model.posterior(self.X_baseline)
            baseline_values = base_objective(posterior.mean, X=self.X_baseline)
            if baseline_values.ndim < 1:
                raise RuntimeError(
                    "objective must return values with an objective dimension. "
                    f"Got shape {tuple(baseline_values.shape)}."
                )
            objective_dim = int(baseline_values.shape[-1])

        if objective_dim != ref.numel():
            raise ValueError(
                "ref_point length must match the transformed objective dimension. "
                f"Got ref_point length {ref.numel()} and objective dimension "
                f"{objective_dim}."
            )

        if weights is None:
            random_weights = torch.rand(objective_dim, **tkwargs)
            scalarization_weights = random_weights / random_weights.sum().clamp_min(1e-12)
        else:
            scalarization_weights = torch.as_tensor(weights, **tkwargs).reshape(-1)
            if scalarization_weights.numel() != objective_dim:
                raise ValueError(
                    "weights length must match the transformed objective dimension. "
                    f"Got weights length {scalarization_weights.numel()} and "
                    f"objective dimension {objective_dim}."
                )
            if bool((scalarization_weights < 0).any()):
                raise ValueError("weights must be non-negative.")
            if float(scalarization_weights.sum()) <= 0.0:
                raise ValueError("weights must contain at least one positive value.")
            scalarization_weights = (
                scalarization_weights / scalarization_weights.sum().clamp_min(1e-12)
            )

        self.register_buffer("weights", scalarization_weights)
        self.register_buffer("ref_point", ref)

        with torch.no_grad():
            baseline_score = self._scalarize(baseline_values)
            self.register_buffer("best_value", baseline_score.max())

    def _scalarize(self, values: Tensor) -> Tensor:
        """Apply augmented Chebyshev scalarization in objective space."""

        if values.ndim < 1 or values.shape[-1] != self.weights.numel():
            raise RuntimeError(
                "objective output dimension must match NParEGO weights. "
                f"Got values.shape={tuple(values.shape)} and "
                f"weights.shape={tuple(self.weights.shape)}."
            )

        weights = self.weights.to(device=values.device, dtype=values.dtype)
        ref_point = self.ref_point.to(device=values.device, dtype=values.dtype)
        shifted = values - ref_point
        weighted = shifted * weights
        chebyshev = weighted.min(dim=-1).values
        augmentation = self.rho * weighted.sum(dim=-1)
        return chebyshev + augmentation

    def _apply_constraints(self, improvement: Tensor, samples: Tensor) -> Tensor:
        if not self.constraints:
            return improvement
        feasibility = compute_smoothed_feasibility_indicator(
            constraints=self.constraints,
            samples=samples,
            eta=self.eta,
            fat=self.fat,
        )
        return improvement * feasibility

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        samples = self.get_posterior_samples(posterior)
        objective_values = self.base_objective(samples, X=X)
        scalarized = self._scalarize(objective_values)
        improvement = (
            scalarized - self.best_value.to(device=scalarized.device, dtype=scalarized.dtype)
        ).clamp_min(0.0)
        improvement = self._apply_constraints(improvement, samples)

        # qEI-style reduction: maximize within q, then average MC sample dims.
        value = improvement.max(dim=-1).values
        batch_ndim = len(X.shape[:-2])
        while value.ndim > batch_ndim:
            value = value.mean(dim=0)
        return value


__all__ = [
    "qMultiOutputRegressionExpectedHypervolumeImprovement",
    "qMultiOutputRegressionNoisyExpectedHypervolumeImprovement",
    "qMultiOutputRegressionLogExpectedHypervolumeImprovement",
    "qMultiOutputRegressionLogNoisyExpectedHypervolumeImprovement",
    "qMultiOutputRegressionNParEGO",
]
