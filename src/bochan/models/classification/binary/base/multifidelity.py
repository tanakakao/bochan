"""Wide-format multi-fidelity binary classification models."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from botorch.models.transforms.input import InputTransform
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.means import Mean
from torch import Tensor

from bochan.models.components.mixed_kronecker import (
    build_mixed_kronecker_kernel,
    get_continuous_dims,
    normalize_mixed_dims,
    validate_mixed_input_transform_for_training,
)

from ._multifidelity_conditioning import _WideMultiFidelityConditioningMixin
from ._multifidelity_core import _WideMultiFidelityBinaryCore


class WideMultiFidelityBinaryClassificationGPModel(
    _WideMultiFidelityConditioningMixin,
    _WideMultiFidelityBinaryCore,
):
    """Normal-input binary SVGP trained from wide ordered-fidelity labels."""


class WideMixedMultiFidelityBinaryClassificationGPModel(
    WideMultiFidelityBinaryClassificationGPModel
):
    """Mixed-input binary SVGP trained from wide ordered-fidelity labels."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        train_Yvar: Tensor | None = None,
        *,
        fidelity_values,
        target_fidelity=None,
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
        normalized_cat_dims = normalize_mixed_dims(
            cat_dims, int(raw_X.shape[-1])
        )
        validation_transform = getattr(
            input_transform,
            "base_transform",
            input_transform,
        )
        validate_mixed_input_transform_for_training(
            raw_X, validation_transform, cat_dims=normalized_cat_dims
        )
        data_kernel = covar_module
        if _full_covar_module is None and data_kernel is None:
            data_kernel = build_mixed_kronecker_kernel(
                d=int(raw_X.shape[-1]), cat_dims=normalized_cat_dims
            ).base_kernel
        super().__init__(
            train_X=raw_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            fidelity_values=fidelity_values,
            target_fidelity=target_fidelity,
            likelihood=likelihood,
            input_transform=input_transform,
            mean_module=mean_module,
            covar_module=data_kernel,
            fidelity_covar_module=fidelity_covar_module,
            num_inducing=num_inducing,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            _full_covar_module=_full_covar_module,
        )
        self.cat_dims = list(normalized_cat_dims)
        self.cont_dims = get_continuous_dims(self.data_dim, normalized_cat_dims)
        self._ignore_X_dims_scaling_check = [*self.cat_dims, self.data_dim]

    def _condition_constructor_kwargs(self):
        return {"cat_dims": list(self.cat_dims)}


WideMultiFidelityBinaryClassificationMixedGPModel = (
    WideMixedMultiFidelityBinaryClassificationGPModel
)

__all__ = [
    "WideMixedMultiFidelityBinaryClassificationGPModel",
    "WideMultiFidelityBinaryClassificationGPModel",
    "WideMultiFidelityBinaryClassificationMixedGPModel",
]
