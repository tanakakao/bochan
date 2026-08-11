"""Core wide multi-fidelity binary SVGP implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional, Union

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.transforms.input import InputTransform
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.means import Mean
from torch import Tensor

from ._multifidelity_utils import (
    make_default_data_kernel,
    make_multifidelity_kernel,
    normalize_fidelity_values,
    prepare_fidelity_input_transform,
    validate_binary_targets,
    wide_fidelity_to_long,
    wide_probability_tensors,
)
from .models import BinaryClassificationGPModel as _BinaryClassificationGPModel
from bochan.models.multioutput.binary import MultiOutputBernoulliPosterior


class _WideMultiFidelityBinaryCore(_BinaryClassificationGPModel):
    """Binary SVGP trained from ``X=[n,d]`` and wide fidelity labels."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        fidelity_values: Sequence[float] | Tensor,
        target_fidelity: float | Tensor | None = None,
        likelihood: BernoulliLikelihood | None = None,
        input_transform: InputTransform | None = None,
        mean_module: Mean | None = None,
        covar_module: Kernel | None = None,
        fidelity_covar_module: Kernel | None = None,
        num_inducing: int = 128,
        inducing_points: Tensor | None = None,
        learn_inducing_locations: bool = True,
        _full_covar_module: Kernel | None = None,
    ) -> None:
        raw_X = torch.as_tensor(train_X)
        raw_Y = torch.as_tensor(train_Y, dtype=raw_X.dtype, device=raw_X.device)
        if raw_Y.ndim != 2:
            raise ValueError("train_Y must have shape [n, n_fidelities].")
        validate_binary_targets(raw_Y)
        values = normalize_fidelity_values(
            fidelity_values, raw_X, int(raw_Y.shape[-1])
        )
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
        full_kernel = _full_covar_module
        if full_kernel is None:
            data_kernel = covar_module or make_default_data_kernel(data_dim, raw_X)
            full_kernel = make_multifidelity_kernel(
                data_dim=data_dim,
                ref_X=raw_X,
                data_covar_module=data_kernel,
                fidelity_covar_module=fidelity_covar_module,
            )
        super().__init__(
            train_X=X_long,
            train_Y=Y_long.squeeze(-1),
            train_Yvar=None if Yvar_long is None else Yvar_long.squeeze(-1),
            likelihood=likelihood,
            input_transform=prepare_fidelity_input_transform(input_transform, data_dim),
            mean_module=mean_module,
            covar_module=full_kernel,
            num_inducing=num_inducing,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
        )
        self.data_dim = data_dim
        self.num_fidelities = int(values.numel())
        self.target_fidelity = float(target.detach().cpu())
        self.target_fidelity_index = int(torch.where(matches)[0][0])
        self.train_X_wide = raw_X.detach().clone()
        self.train_Y_wide = raw_Y.detach().clone()
        self.train_Yvar_wide = self._prepare_wide_noise(train_Yvar, raw_X, raw_Y)
        self.register_buffer("fidelity_values", values.detach().clone())

    @staticmethod
    def _prepare_wide_noise(train_Yvar, raw_X: Tensor, raw_Y: Tensor):
        if train_Yvar is None:
            return None
        noise = torch.as_tensor(
            train_Yvar, dtype=raw_X.dtype, device=raw_X.device
        ).detach().clone()
        return torch.where(
            torch.isnan(raw_Y), torch.full_like(noise, float("nan")), noise
        )

    def _reference_input(self) -> Tensor:
        inputs = self.train_inputs_raw
        return inputs[0] if isinstance(inputs, tuple) else inputs

    def _coerce_X(self, X: Tensor) -> Tensor:
        ref = self._reference_input()
        return torch.as_tensor(X, dtype=ref.dtype, device=ref.device)

    def _public_X(self, X: Tensor) -> Tensor:
        X = self._coerce_X(X)
        if X.shape[-1] != self.data_dim:
            raise ValueError(
                f"Expected public input dimension {self.data_dim}, got {X.shape[-1]}."
            )
        return X

    def _append_fidelity(self, X: Tensor, fidelity: float | Tensor) -> Tensor:
        X = self._public_X(X)
        value = torch.as_tensor(fidelity, dtype=X.dtype, device=X.device).reshape(())
        if not bool(torch.isfinite(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError("fidelity must be finite and within [0, 1].")
        return torch.cat([X, value.expand(*X.shape[:-1], 1)], dim=-1)

    def _internal_X(self, X: Tensor) -> Tensor:
        X = self._coerce_X(X)
        if X.shape[-1] == self.data_dim:
            return self._append_fidelity(X, self.target_fidelity)
        if X.shape[-1] == self.data_dim + 1:
            fidelity = X[..., -1]
            if not torch.isfinite(fidelity).all() or bool(
                ((fidelity < 0.0) | (fidelity > 1.0)).any()
            ):
                raise ValueError("Fidelity values must be finite and within [0, 1].")
            return X
        raise ValueError(
            f"Expected input dimension {self.data_dim} or {self.data_dim + 1}, "
            f"got {X.shape[-1]}."
        )

    def expand_fidelities(self, X: Tensor, fidelity_values=None) -> Tensor:
        X = self._public_X(X)
        values = self.fidelity_values if fidelity_values is None else torch.as_tensor(
            fidelity_values, dtype=X.dtype, device=X.device
        ).reshape(-1)
        if values.numel() == 0 or not torch.isfinite(values).all():
            raise ValueError("At least one finite fidelity value is required.")
        if bool(((values < 0.0) | (values > 1.0)).any()):
            raise ValueError("Fidelity values must be within [0, 1].")
        q = int(X.shape[-2])
        expanded = X.unsqueeze(-2).expand(
            *X.shape[:-2], q, values.numel(), self.data_dim
        )
        fidelity = values.view(*([1] * (X.ndim - 2)), 1, -1, 1).expand(
            *X.shape[:-2], q, values.numel(), 1
        )
        return torch.cat([expanded, fidelity], dim=-1).reshape(
            *X.shape[:-2], q * values.numel(), self.data_dim + 1
        )

    def posterior_at_fidelity(self, X: Tensor, fidelity, **kwargs: Any):
        return super().posterior(self._append_fidelity(X, fidelity), **kwargs)

    def posterior_at_target_fidelity(self, X: Tensor, **kwargs: Any):
        return self.posterior_at_fidelity(X, self.target_fidelity, **kwargs)

    def posterior_all_fidelities(
        self,
        X: Tensor,
        fidelity_indices: Sequence[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ) -> MultiOutputBernoulliPosterior:
        if not isinstance(observation_noise, bool):
            raise TypeError(
                "Tensor observation_noise is not supported for the all-fidelity posterior."
            )
        selected = self._selected_fidelities(fidelity_indices)
        X = self._public_X(X)
        base = super().posterior(
            self.expand_fidelities(X, self.fidelity_values[selected]),
            observation_noise=observation_noise,
            posterior_transform=None,
            **kwargs,
        )
        mean, variance = wide_probability_tensors(
            base, public_q=int(X.shape[-2]), num_fidelities=len(selected)
        )
        posterior = MultiOutputBernoulliPosterior(mean=mean, variance=variance)
        return posterior_transform(posterior) if posterior_transform else posterior

    def _selected_fidelities(self, indices: Sequence[int] | None) -> list[int]:
        selected = list(range(self.num_fidelities)) if indices is None else [
            int(index) for index in indices
        ]
        if not selected or min(selected) < 0 or max(selected) >= self.num_fidelities:
            raise ValueError("Invalid fidelity_indices.")
        return selected

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[list[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ):
        X = self._coerce_X(X)
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
        raise ValueError("Unexpected input dimension for multi-fidelity posterior.")

    def probability_posterior(self, X: Tensor, **kwargs: Any):
        return self.posterior(X, **kwargs)

    def latent_posterior_at_fidelity(self, X: Tensor, fidelity) -> GPyTorchPosterior:
        return super().latent_posterior(self._append_fidelity(X, fidelity))

    def latent_posterior_at_target_fidelity(self, X: Tensor) -> GPyTorchPosterior:
        return self.latent_posterior_at_fidelity(X, self.target_fidelity)

    def latent_posterior_all_fidelities(
        self, X: Tensor, fidelity_indices: Sequence[int] | None = None
    ) -> GPyTorchPosterior:
        selected = self._selected_fidelities(fidelity_indices)
        X = self._public_X(X)
        return super().latent_posterior(
            self.expand_fidelities(X, self.fidelity_values[selected])
        )

    def latent_posterior(self, X: Tensor) -> GPyTorchPosterior:
        return super().latent_posterior(self._internal_X(X))

    def class_probs(self, X: Tensor, **kwargs: Any) -> Tensor:
        p_one = self.posterior(X, **kwargs).mean
        return torch.cat([1.0 - p_one, p_one], dim=-1)

    def class_probs_all_fidelities(
        self, X: Tensor, fidelity_indices: Sequence[int] | None = None, **kwargs: Any
    ) -> Tensor:
        p_one = self.posterior_all_fidelities(
            X, fidelity_indices=fidelity_indices, **kwargs
        ).mean
        return torch.stack([1.0 - p_one, p_one], dim=-1)

    def predict_class(self, X: Tensor, threshold: float = 0.5, **kwargs: Any):
        return (self.posterior(X, **kwargs).mean >= float(threshold)).long()

    def forward(self, X: Tensor):
        return super().forward(self._internal_X(X))

    def _condition_constructor_kwargs(self) -> dict[str, Any]:
        return {}
