from __future__ import annotations

import copy
from typing import Any

import torch
from torch import Tensor

from bochan.models.regression.poisson.non_gaussian.poisson import (
    PoissonLogLikelihood,
    PoissonPosterior,
    PoissonGPModel as _PoissonGPModel,
    PoissonMixedGPModel as _PoissonMixedGPModel,
    build_mixed_poisson_kernel,
)
from bochan.models.components.poisson import prepare_count_targets, clone_input_transform


class _AlignedPoissonMixin:
    """Alignment layer for the historical non_gaussian import path.

    The concrete implementation stores raw inputs on the outer wrapper's
    ``train_inputs``.  Existing regression / ordinal wrappers use
    ``train_inputs`` for the latent-model inputs and ``train_inputs_raw`` for raw
    search-space inputs.  This mixin exposes that convention without changing the
    distribution-specific implementation directly.
    """

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
    ) -> PoissonPosterior:
        return super().posterior(
            X=X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )


class PoissonGPModel(_AlignedPoissonMixin, _PoissonGPModel):
    """Poisson GP model aligned with regression / ordinal wrapper conventions."""


class PoissonMixedGPModel(_AlignedPoissonMixin, _PoissonMixedGPModel):
    """Mixed-input Poisson GP model aligned with wrapper conventions."""

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> "PoissonMixedGPModel":
        if kwargs.get("noise") is not None:
            raise NotImplementedError("PoissonMixedGPModel does not support noise in condition_on_observations.")
        if isinstance(X, tuple):
            X = X[0]
        X = torch.as_tensor(X, device=self.train_inputs_raw[0].device, dtype=self.train_inputs_raw[0].dtype)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        Y = prepare_count_targets(Y, X)
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
            exp_clip=self.exp_clip,
            min_rate=self.min_rate,
        )


__all__ = [
    "PoissonLogLikelihood",
    "PoissonPosterior",
    "PoissonGPModel",
    "PoissonMixedGPModel",
    "build_mixed_poisson_kernel",
]
