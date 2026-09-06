"""Correlated multi-output Gaussian multi-fidelity surrogate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from botorch.models.kernels.linear_truncated_fidelity import LinearTruncatedFidelityKernel
from botorch.models.multitask import KroneckerMultiTaskGP
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from gpytorch.kernels import ProductKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from torch import Tensor

from .spec import FidelitySpec, ResolvedFidelitySpec


def _resolved_spec(
    spec: FidelitySpec | ResolvedFidelitySpec,
    *,
    d: int,
    bounds: Tensor | None,
) -> ResolvedFidelitySpec:
    return spec if isinstance(spec, ResolvedFidelitySpec) else spec.resolve(d=d, bounds=bounds)


def _multifidelity_data_kernel(
    *,
    d: int,
    fidelity_features: Sequence[int],
    nu: float,
) -> ScaleKernel:
    """Build the Wu-style linear-truncated data kernel used by BoTorch MF GPs."""

    kernels = [
        LinearTruncatedFidelityKernel(
            fidelity_dims=[int(feature)],
            dimension=d,
            nu=nu,
            power_prior=GammaPrior(3.0, 3.0),
        )
        for feature in fidelity_features
    ]
    return ScaleKernel(
        ProductKernel(*kernels),
        outputscale_prior=GammaPrior(2.0, 0.15),
    )


class GaussianCorrelatedMultiFidelityGP(KroneckerMultiTaskGP):
    """Kronecker ICM model coupling physical outputs and fidelity-aware inputs.

    The model uses the same ``n x d`` design matrix for every output and therefore
    requires a fully observed ``n x m`` outcome matrix. Output correlations are
    learned by the ICM task covariance while the data covariance uses BoTorch's
    linear-truncated multi-fidelity kernel over the configured fidelity features.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        fidelity_spec: FidelitySpec | ResolvedFidelitySpec,
        bounds: Tensor | None = None,
        rank: int | None = None,
        nu: float = 2.5,
        outcome_transform: OutcomeTransform | None = None,
        input_transform: Any | None = None,
        task_covar_prior: Any | None = None,
        **kwargs: Any,
    ) -> None:
        X = torch.as_tensor(train_X)
        Y = torch.as_tensor(train_Y, dtype=X.dtype, device=X.device)
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("Correlated multi-output MF requires train_X [n,d] and train_Y [n,m].")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("train_X and train_Y must contain the same number of rows.")
        if Y.shape[-1] < 2:
            raise ValueError("Correlated multi-output MF requires at least two outputs.")
        if not bool(torch.isfinite(X).all()) or not bool(torch.isfinite(Y).all()):
            raise ValueError("Correlated multi-output MF requires fully observed finite train_X/train_Y.")
        if train_Yvar is not None:
            raise NotImplementedError(
                "Phase 64 correlated multi-output MF does not support train_Yvar; "
                "use independent multi-output MF for known-noise outputs."
            )

        resolved = _resolved_spec(fidelity_spec, d=int(X.shape[-1]), bounds=bounds)
        if outcome_transform is None:
            outcome_transform = Standardize(m=int(Y.shape[-1]))
        data_kernel = _multifidelity_data_kernel(
            d=int(X.shape[-1]),
            fidelity_features=resolved.fidelity_features,
            nu=float(nu),
        )
        super().__init__(
            train_X=X,
            train_Y=Y,
            data_covar_module=data_kernel,
            rank=rank,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            task_covar_prior=task_covar_prior,
            **kwargs,
        )
        self.fidelity_mode = "feature"
        self.fidelity_features = tuple(resolved.fidelity_features)
        self.target_fidelities = dict(resolved.target_fidelities or {})
        self.input_mode = "continuous"
        self.cat_dims: tuple[int, ...] = ()
        self.multi_output_fidelity = "correlated"
        self.is_multifidelity_model = True
        self.num_fidelity_outputs = int(Y.shape[-1])

    def fidelity_metadata(self) -> dict[str, Any]:
        return {
            "fidelity_mode": self.fidelity_mode,
            "fidelity_features": self.fidelity_features,
            "target_fidelities": dict(self.target_fidelities),
            "input_mode": self.input_mode,
            "cat_dims": self.cat_dims,
            "multi_output_fidelity": self.multi_output_fidelity,
            "num_fidelity_outputs": self.num_fidelity_outputs,
        }

    def make_mll(self) -> ExactMarginalLogLikelihood:
        return ExactMarginalLogLikelihood(self.likelihood, self)


__all__ = ["GaussianCorrelatedMultiFidelityGP"]
