"""Conditioning support for wide multi-fidelity binary classifiers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import Tensor

from ._multifidelity_utils import validate_binary_targets
from .models import _prepare_binary_conditioning_data


class _WideMultiFidelityConditioningMixin:
    """Rebuild a variational wide model after appending labelled observations."""

    def _wide_noise_for_conditioning(
        self,
        new_Y: Tensor,
        new_Yvar: Tensor | None,
        fidelity_indices: Tensor,
    ) -> Tensor | None:
        if self.train_Yvar_wide is None and new_Yvar is None:
            return None
        old_noise = self.train_Yvar_wide
        if old_noise is None:
            old_noise = torch.where(
                torch.isnan(self.train_Y_wide),
                torch.full_like(self.train_Y_wide, float("nan")),
                torch.zeros_like(self.train_Y_wide),
            )
        else:
            old_noise = old_noise.clone()
        new_noise = torch.full(
            (new_Y.shape[0], self.num_fidelities),
            float("nan"),
            dtype=new_Y.dtype,
            device=new_Y.device,
        )
        values = torch.zeros_like(new_Y) if new_Yvar is None else new_Yvar
        new_noise[
            torch.arange(new_Y.shape[0], device=new_Y.device), fidelity_indices
        ] = values
        return torch.cat([old_noise, new_noise], dim=0)

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        noise: Tensor | None = None,
        **kwargs: Any,
    ):
        """Return a reconstructed model containing the new observations."""
        del kwargs
        X_flat, Y_flat, Yvar_flat = _prepare_binary_conditioning_data(
            self._internal_X(X), Y, noise
        )
        validate_binary_targets(Y_flat.reshape(-1, 1))
        matches = torch.isclose(
            X_flat[:, -1, None], self.fidelity_values[None, :]
        )
        if not bool(matches.any(dim=-1).all()):
            invalid = X_flat[~matches.any(dim=-1), -1]
            raise ValueError(
                "Conditioning observations must use configured fidelity_values. "
                f"Invalid values: {invalid.detach().cpu().tolist()}."
            )
        fidelity_indices = matches.long().argmax(dim=-1)
        X_public = X_flat[:, : self.data_dim]
        new_Y_wide = torch.full(
            (Y_flat.shape[0], self.num_fidelities),
            float("nan"),
            dtype=Y_flat.dtype,
            device=Y_flat.device,
        )
        new_Y_wide[
            torch.arange(Y_flat.shape[0], device=Y_flat.device), fidelity_indices
        ] = Y_flat
        train_X = torch.cat(
            [self.train_X_wide, X_public.to(self.train_X_wide)], dim=0
        )
        train_Y = torch.cat(
            [self.train_Y_wide, new_Y_wide.to(self.train_Y_wide)], dim=0
        )
        train_Yvar = self._wide_noise_for_conditioning(
            Y_flat, Yvar_flat, fidelity_indices
        )
        inducing_points = (
            self.model.variational_strategy.inducing_points.detach().clone()
        )
        new_model = self.__class__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            fidelity_values=self.fidelity_values.detach().clone(),
            target_fidelity=self.target_fidelity,
            likelihood=deepcopy(self.likelihood),
            input_transform=deepcopy(self.input_transform),
            mean_module=deepcopy(self.model.mean_module),
            num_inducing_points=int(inducing_points.shape[-2]),
            inducing_points=inducing_points,
            learn_inducing_locations=getattr(
                self.model.variational_strategy,
                "learn_inducing_locations",
                True,
            ),
            _full_covar_module=deepcopy(self.model.covar_module),
            **self._condition_constructor_kwargs(),
        )
        new_model.load_state_dict(self.state_dict(), strict=False)
        new_model.eval()
        return new_model
