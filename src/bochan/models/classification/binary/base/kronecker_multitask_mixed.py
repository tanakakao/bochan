from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any, Optional

import torch
from botorch.models.transforms.input import InputTransform
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.means import Mean
from torch import Tensor

from bochan.models.components.kronecker_multitask import (
    canonicalize_block_design_targets,
)
from bochan.models.components.mixed_kronecker import (
    build_mixed_kronecker_kernel,
    get_continuous_dims,
    normalize_mixed_dims,
    transform_mixed_inputs,
    validate_mixed_input_transform_for_training,
)

from .kronecker_multitask import KroneckerMultiTaskBinaryClassificationGPModel


class KroneckerMultiTaskBinaryClassificationMixedGPModel(
    KroneckerMultiTaskBinaryClassificationGPModel
):
    r"""Mixed-input block-design binary classifier with ICM task covariance.

    ``train_X`` has shape ``[n, d]`` and contains continuous and categorical
    columns. ``train_Y`` has shape ``[n, m]`` and all tasks are observed at the
    same input locations. The data kernel is

    ``continuous + categorical + continuous * categorical``

    and is multiplied by the learned task covariance through the parent ICM
    model.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        train_Yvar: Optional[Tensor] = None,
        *,
        rank: Optional[int] = None,
        likelihood: Optional[BernoulliLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        mean_module: Optional[Mean] = None,
        data_covar_module: Optional[Kernel] = None,
        num_inducing: int = 128,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
    ) -> None:
        raw_train_X = torch.as_tensor(train_X).contiguous()
        normalized_cat_dims = normalize_mixed_dims(
            cat_dims,
            raw_train_X.shape[-1],
        )
        validate_mixed_input_transform_for_training(
            raw_train_X,
            input_transform,
            cat_dims=normalized_cat_dims,
        )
        if data_covar_module is None:
            data_covar_module = build_mixed_kronecker_kernel(
                d=raw_train_X.shape[-1],
                cat_dims=normalized_cat_dims,
            )

        self.cat_dims = list(normalized_cat_dims)
        self.cont_dims = get_continuous_dims(
            raw_train_X.shape[-1],
            normalized_cat_dims,
        )
        super().__init__(
            train_X=raw_train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            rank=rank,
            likelihood=likelihood,
            input_transform=input_transform,
            mean_module=mean_module,
            data_covar_module=data_covar_module,
            num_inducing=num_inducing,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
        )
        # Parent construction does not know about mixed metadata.
        self.cat_dims = list(normalized_cat_dims)
        self.cont_dims = get_continuous_dims(
            raw_train_X.shape[-1],
            normalized_cat_dims,
        )

    def transform_inputs(self, X: Tensor) -> Tensor:
        return transform_mixed_inputs(
            X,
            self.input_transform,
            cat_dims=self.cat_dims,
        )

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        noise: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> "KroneckerMultiTaskBinaryClassificationMixedGPModel":
        """Append complete mixed-input block-design observations."""
        X = torch.as_tensor(
            X,
            device=self.train_inputs_raw[0].device,
            dtype=self.train_inputs_raw[0].dtype,
        )
        if X.ndim != 2 or X.shape[-1] != self.train_inputs_raw[0].shape[-1]:
            raise ValueError(
                "X must have shape [n_new, d] with the same feature dimension as train_X."
            )
        Y = canonicalize_block_design_targets(X, Y, target_dtype=X.dtype)
        if Y.shape[-1] != self.num_tasks:
            raise ValueError(
                f"Y must contain {self.num_tasks} tasks, got {Y.shape[-1]}."
            )
        self._validate_binary_targets(Y)
        Yvar = self._canonicalize_train_Yvar(noise, train_X=X, train_Y=Y)

        train_X_full = torch.cat([self.train_inputs_raw[0], X], dim=-2)
        train_Y_full = torch.cat([self.train_targets, Y], dim=-2)

        train_Yvar_full = None
        if self.train_Yvar is not None or Yvar is not None:
            old_noise = self.train_Yvar
            if old_noise is None:
                old_noise = torch.zeros_like(self.train_targets)
            if Yvar is None:
                Yvar = torch.zeros_like(Y)
            train_Yvar_full = torch.cat([old_noise, Yvar], dim=-2)

        new_model = self.__class__(
            train_X=train_X_full,
            train_Y=train_Y_full,
            cat_dims=self.cat_dims,
            train_Yvar=train_Yvar_full,
            rank=self.rank,
            likelihood=copy.deepcopy(self.likelihood),
            input_transform=copy.deepcopy(self.input_transform),
            mean_module=copy.deepcopy(self.model.mean_module),
            data_covar_module=copy.deepcopy(self.model.data_covar_module),
            num_inducing=self.num_inducing,
            inducing_points=self.inducing_points_raw.detach().clone(),
            learn_inducing_locations=self.learn_inducing_locations,
        )
        new_model.load_state_dict(copy.deepcopy(self.state_dict()), strict=False)
        new_model.eval()
        new_model.likelihood.eval()
        return new_model


__all__ = ["KroneckerMultiTaskBinaryClassificationMixedGPModel"]
