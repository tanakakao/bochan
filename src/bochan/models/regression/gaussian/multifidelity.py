"""Wide-format multi-fidelity Gaussian regression models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from botorch.utils.types import DEFAULT
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import Likelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from bochan.models.components.mixed_kronecker import (
    build_mixed_kronecker_kernel,
    get_continuous_dims,
    normalize_mixed_dims,
    validate_mixed_input_transform_for_training,
)
from bochan.models.multitask.task_feature import (
    PerturbationAwareWidePosterior,
    TaskFeatureInputTransform,
)


class FidelityFeatureInputTransform(TaskFeatureInputTransform):
    """Transform public design columns while preserving appended fidelity."""


def _fidelity_tensor(values: Sequence[float] | Tensor, X: Tensor, m: int) -> Tensor:
    result = torch.as_tensor(values, dtype=X.dtype, device=X.device).reshape(-1)
    if result.numel() != m:
        raise ValueError(f"Expected {m} fidelity_values, got {result.numel()}.")
    if not torch.isfinite(result).all() or torch.unique(result).numel() != m:
        raise ValueError("fidelity_values must be finite and unique.")
    if bool(((result < 0.0) | (result > 1.0)).any()):
        raise ValueError("fidelity_values must be within [0, 1].")
    return result


def wide_fidelity_to_long(
    train_X: Tensor,
    train_Y: Tensor,
    fidelity_values: Sequence[float] | Tensor,
    train_Yvar: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Convert ``X=[n,d], Y=[n,m]`` into observed long rows ``[x, fidelity]``."""

    X = torch.as_tensor(train_X)
    Y = torch.as_tensor(train_Y, dtype=X.dtype, device=X.device)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError("train_X and train_Y must have shapes [n,d] and [n,m].")
    if Y.shape[1] < 2:
        raise ValueError("Wide multi-fidelity models require at least two fidelity columns.")
    if torch.isinf(Y).any():
        raise ValueError("train_Y may contain NaN, but not inf.")

    values = _fidelity_tensor(fidelity_values, X, int(Y.shape[1]))
    observed = ~torch.isnan(Y)
    missing = torch.where(~observed.any(dim=0))[0]
    if missing.numel():
        raise ValueError(
            "Every fidelity must have at least one observation. Missing columns: "
            f"{missing.detach().cpu().tolist()}."
        )

    Yvar_long = None
    if train_Yvar is not None:
        Yvar = torch.as_tensor(train_Yvar, dtype=X.dtype, device=X.device)
        if Yvar.shape != Y.shape:
            raise ValueError("train_Yvar must have the same shape as train_Y.")
        observed_var = Yvar[observed]
        if not torch.isfinite(observed_var).all() or bool((observed_var < 0).any()):
            raise ValueError("Observed train_Yvar values must be finite and non-negative.")

    row_idx, fidelity_idx = observed.nonzero(as_tuple=True)
    X_long = torch.cat([X[row_idx], values[fidelity_idx, None]], dim=-1).contiguous()
    Y_long = Y[row_idx, fidelity_idx, None].contiguous()
    if train_Yvar is not None:
        Yvar_long = Yvar[row_idx, fidelity_idx, None].contiguous()
    return X_long, Y_long, Yvar_long


def _prepare_input_transform(transform: InputTransform | None, data_dim: int) -> InputTransform | None:
    if transform is None or isinstance(transform, TaskFeatureInputTransform):
        return transform
    return FidelityFeatureInputTransform(transform, data_dim=data_dim)


def _prepare_outcome_transform(transform: Any) -> Any:
    if (
        transform is not None
        and transform is not DEFAULT
        and transform.__class__.__name__ == "AutoStandardizeOutcomeTransform"
    ):
        return Standardize(m=1)
    return transform


class WideMultiFidelityGP(SingleTaskMultiFidelityGP):
    """Single-output multi-fidelity GP trained from wide fidelity columns."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        fidelity_values: Sequence[float] | Tensor,
        target_fidelity: float | Tensor | None = None,
        linear_truncated: bool = True,
        nu: float = 2.5,
        covar_module: Kernel | None = None,
        likelihood: Likelihood | None = None,
        outcome_transform: OutcomeTransform | Any | None = None,
        input_transform: InputTransform | None = None,
    ) -> None:
        raw_X = torch.as_tensor(train_X)
        raw_Y = torch.as_tensor(train_Y, dtype=raw_X.dtype, device=raw_X.device)
        values = _fidelity_tensor(fidelity_values, raw_X, int(raw_Y.shape[-1]))
        target = values.max() if target_fidelity is None else torch.as_tensor(
            target_fidelity, dtype=raw_X.dtype, device=raw_X.device
        ).reshape(())
        matches = torch.isclose(values, target)
        if not bool(matches.any()):
            raise ValueError("target_fidelity must be one of fidelity_values.")

        X_long, Y_long, Yvar_long = wide_fidelity_to_long(
            raw_X, raw_Y, values, train_Yvar=train_Yvar
        )
        data_dim = int(raw_X.shape[-1])
        super().__init__(
            train_X=X_long,
            train_Y=Y_long,
            train_Yvar=Yvar_long,
            data_fidelities=[data_dim],
            linear_truncated=linear_truncated,
            nu=nu,
            covar_module=covar_module,
            likelihood=likelihood,
            outcome_transform=_prepare_outcome_transform(outcome_transform),
            input_transform=_prepare_input_transform(input_transform, data_dim),
        )
        self.data_dim = data_dim
        self.num_fidelities = int(values.numel())
        self.target_fidelity = float(target.detach().cpu())
        self.target_fidelity_index = int(torch.where(matches)[0][0])
        self.train_X_raw = raw_X.detach().clone()
        self.train_Y_raw = raw_Y.detach().clone()
        self.train_Yvar_raw = None if train_Yvar is None else torch.as_tensor(train_Yvar).detach().clone()
        self.register_buffer("fidelity_values", values.detach().clone())

    def _reference_input(self) -> Tensor:
        train_inputs = self.train_inputs
        return train_inputs[0] if isinstance(train_inputs, tuple) else train_inputs

    def _public_X(self, X: Tensor) -> Tensor:
        ref = self._reference_input()
        X = torch.as_tensor(X, dtype=ref.dtype, device=ref.device)
        if X.shape[-1] != self.data_dim:
            raise ValueError(f"Expected public input dimension {self.data_dim}, got {X.shape[-1]}.")
        return X

    def _append_fidelity(self, X: Tensor, fidelity: float | Tensor) -> Tensor:
        X = self._public_X(X)
        value = torch.as_tensor(fidelity, dtype=X.dtype, device=X.device).reshape(())
        column = value.expand(*X.shape[:-1], 1)
        return torch.cat([X, column], dim=-1)

    def expand_fidelities(
        self, X: Tensor, fidelity_values: Sequence[float] | Tensor | None = None
    ) -> Tensor:
        X = self._public_X(X)
        values = self.fidelity_values if fidelity_values is None else torch.as_tensor(
            fidelity_values, dtype=X.dtype, device=X.device
        ).reshape(-1)
        q = int(X.shape[-2])
        expanded = X.unsqueeze(-2).expand(*X.shape[:-2], q, values.numel(), self.data_dim)
        fidelity = values.view(*([1] * (X.ndim - 2)), 1, -1, 1).expand(
            *X.shape[:-2], q, values.numel(), 1
        )
        return torch.cat([expanded, fidelity], dim=-1).reshape(
            *X.shape[:-2], q * values.numel(), self.data_dim + 1
        )

    def posterior_at_fidelity(self, X: Tensor, fidelity: float | Tensor, **kwargs: Any):
        return super().posterior(self._append_fidelity(X, fidelity), **kwargs)

    def posterior_at_target_fidelity(self, X: Tensor, **kwargs: Any):
        return self.posterior_at_fidelity(X, self.target_fidelity, **kwargs)

    def posterior_all_fidelities(
        self,
        X: Tensor,
        fidelity_indices: Sequence[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ):
        if not isinstance(observation_noise, bool):
            raise TypeError("Tensor observation_noise is not supported for wide posterior.")
        selected = list(range(self.num_fidelities)) if fidelity_indices is None else [
            int(i) for i in fidelity_indices
        ]
        if not selected or min(selected) < 0 or max(selected) >= self.num_fidelities:
            raise ValueError("Invalid fidelity_indices.")
        X = self._public_X(X)
        base = super().posterior(
            self.expand_fidelities(X, self.fidelity_values[selected]),
            observation_noise=observation_noise,
            posterior_transform=None,
            **kwargs,
        )
        posterior = PerturbationAwareWidePosterior(
            base,
            public_q=int(X.shape[-2]),
            num_tasks=len(selected),
            output_indices=list(range(len(selected))),
            input_ndim=X.ndim,
        )
        return posterior_transform(posterior) if posterior_transform is not None else posterior

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ):
        ref = self._reference_input()
        X = torch.as_tensor(X, dtype=ref.dtype, device=ref.device)
        if X.shape[-1] == self.data_dim:
            return self.posterior_at_target_fidelity(
                X,
                output_indices=output_indices,
                observation_noise=observation_noise,
                posterior_transform=posterior_transform,
                **kwargs,
            )
        if X.shape[-1] == self.data_dim + 1:
            return super().posterior(
                X,
                output_indices=output_indices,
                observation_noise=observation_noise,
                posterior_transform=posterior_transform,
                **kwargs,
            )
        raise ValueError(f"Expected input dimension {self.data_dim} or {self.data_dim + 1}.")

    def forward(self, X: Tensor):
        X = torch.as_tensor(X)
        if X.shape[-1] == self.data_dim:
            X = self._append_fidelity(X, self.target_fidelity)
        return super().forward(X)

    def condition_on_observations(
        self, X: Tensor, Y: Tensor, noise: Tensor | None = None, **kwargs: Any
    ):
        ref = self._reference_input()
        X = torch.as_tensor(X, dtype=ref.dtype, device=ref.device)
        if X.shape[-1] == self.data_dim:
            X = self._append_fidelity(X, self.target_fidelity)
        elif X.shape[-1] != self.data_dim + 1:
            raise ValueError(f"Expected input dimension {self.data_dim} or {self.data_dim + 1}.")
        return super().condition_on_observations(X=X, Y=Y, noise=noise, **kwargs)

    def make_mll(self) -> ExactMarginalLogLikelihood:
        """Return the exact marginal log likelihood for this model."""
        return ExactMarginalLogLikelihood(self.likelihood, self)


class WideMixedMultiFidelityGP(WideMultiFidelityGP):
    """Wide multi-fidelity GP for mixed continuous/categorical design inputs."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        cat_dims: Sequence[int],
        fidelity_values: Sequence[float] | Tensor,
        target_fidelity: float | Tensor | None = None,
        covar_module: Kernel | None = None,
        likelihood: Likelihood | None = None,
        outcome_transform: OutcomeTransform | Any | None = None,
        input_transform: InputTransform | None = None,
    ) -> None:
        raw_X = torch.as_tensor(train_X)
        cat_dims = normalize_mixed_dims(cat_dims, int(raw_X.shape[-1]))
        validation_transform = (
            input_transform.base_transform
            if isinstance(input_transform, TaskFeatureInputTransform)
            else input_transform
        )
        validate_mixed_input_transform_for_training(
            raw_X, validation_transform, cat_dims=cat_dims
        )
        if covar_module is None:
            covar_module = build_mixed_kronecker_kernel(
                d=int(raw_X.shape[-1]), cat_dims=cat_dims
            )
        super().__init__(
            train_X=raw_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            fidelity_values=fidelity_values,
            target_fidelity=target_fidelity,
            linear_truncated=False,
            covar_module=covar_module,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self.cat_dims = list(cat_dims)
        self.cont_dims = get_continuous_dims(self.data_dim, cat_dims)

    def make_mll(self) -> ExactMarginalLogLikelihood:
        """Return the exact marginal log likelihood for this mixed model."""
        return ExactMarginalLogLikelihood(self.likelihood, self)


WideMultiFidelityMixedGP = WideMixedMultiFidelityGP

__all__ = [
    "FidelityFeatureInputTransform",
    "WideMixedMultiFidelityGP",
    "WideMultiFidelityGP",
    "WideMultiFidelityMixedGP",
    "wide_fidelity_to_long",
]
