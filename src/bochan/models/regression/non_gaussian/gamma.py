from __future__ import annotations

import copy
from typing import Any

import torch
from torch import Tensor

from bochan.models.regression.gamma.non_gaussian.gamma import (
    GammaLogLikelihood,
    GammaPosterior,
    GammaGPModel as _GammaGPModel,
    GammaMixedGPModel as _GammaMixedGPModel,
    build_mixed_gamma_kernel,
)
from bochan.models.components.gamma import prepare_positive_targets, clone_input_transform


class _AlignedGammaMixin:
    """Alignment layer for the historical non_gaussian import path."""

    @property
    def train_inputs(self) -> tuple[Tensor, ...]:
        return self.model.train_inputs

    @train_inputs.setter
    def train_inputs(self, value) -> None:
        self._train_inputs_outer = value

    def posterior(
        self,
        X: Tensor,
        output_indices=None,
        observation_noise: bool | Tensor = False,
        posterior_transform=None,
        **kwargs: Any,
    ) -> GammaPosterior:
        return super().posterior(
            X=X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )


class GammaGPModel(_AlignedGammaMixin, _GammaGPModel):
    """Gamma GP model aligned with regression / ordinal wrapper conventions."""


class GammaMixedGPModel(_AlignedGammaMixin, _GammaMixedGPModel):
    """Mixed-input Gamma GP model aligned with wrapper conventions."""

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> "GammaMixedGPModel":
        if kwargs.get("noise") is not None:
            raise NotImplementedError("GammaMixedGPModel does not support noise in condition_on_observations.")
        if isinstance(X, tuple):
            X = X[0]
        X = torch.as_tensor(X, device=self.train_inputs_raw[0].device, dtype=self.train_inputs_raw[0].dtype)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        Y = prepare_positive_targets(Y, X, min_value=self.min_mean)
        new_X = torch.cat([self.train_inputs_raw[0], X], dim=-2)
        new_Y = torch.cat([self.train_targets, Y], dim=0)
        return self.__class__(
            train_X=new_X,
            train_Y=new_Y,
            cat_dims=list(self.cat_dims),
            likelihood=copy.deepcopy(self.likelihood),
            input_transform=clone_input_transform(self.input_transform),
            mean_module=copy.deepcopy(self.model.mean_module),
            covar_module=copy.deepcopy(self.model.covar_module),
            num_inducing_points=self.num_inducing_points,
            inducing_points=self.model.variational_strategy.inducing_points.detach().clone(),
            learn_inducing_locations=self.learn_inducing_locations,
            link=self.link,
            init_concentration=float(self.likelihood.concentration.detach().cpu()),
            learn_concentration=self.learn_concentration,
            exp_clip=self.exp_clip,
            min_mean=self.min_mean,
            min_concentration=self.min_concentration,
        )


__all__ = [
    "GammaLogLikelihood",
    "GammaPosterior",
    "GammaGPModel",
    "GammaMixedGPModel",
    "build_mixed_gamma_kernel",
]
