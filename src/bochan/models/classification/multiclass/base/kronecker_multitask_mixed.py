from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any, Optional

import torch
from botorch.models.transforms.input import InputTransform
from gpytorch.kernels import Kernel
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

from .kronecker_multitask import (
    BlockDesignMulticlassLikelihood,
    KroneckerMultiTaskMulticlassClassificationGPModel,
)


class KroneckerMultiTaskMulticlassClassificationMixedGPModel(
    KroneckerMultiTaskMulticlassClassificationGPModel
):
    r"""Mixed-input block-design multiclass classifier with correlated tasks.

    Every task uses the same class vocabulary. Each class logit has its own
    mixed-input data kernel and ICM task covariance. The kernel parameter batch
    shape is ``[num_classes, 1]`` so class-specific parameters broadcast over
    the latent-rank axis.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        *,
        num_classes: Optional[int] = None,
        rank: Optional[int] = None,
        likelihood: Optional[BlockDesignMulticlassLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        mean_module: Optional[Mean] = None,
        data_covar_module: Optional[Kernel] = None,
        num_inducing_points: int = 128,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        temperature: float = 1.0,
    ) -> None:
        raw_train_X = torch.as_tensor(train_X).contiguous()
        raw_train_Y = torch.as_tensor(train_Y)
        resolved_num_classes = (
            int(raw_train_Y.max().item()) + 1
            if num_classes is None
            else int(num_classes)
        )
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
                batch_shape=torch.Size([resolved_num_classes, 1]),
            )

        self.cat_dims = list(normalized_cat_dims)
        self.cont_dims = get_continuous_dims(
            raw_train_X.shape[-1],
            normalized_cat_dims,
        )
        super().__init__(
            train_X=raw_train_X,
            train_Y=raw_train_Y,
            num_classes=resolved_num_classes,
            rank=rank,
            likelihood=likelihood,
            input_transform=input_transform,
            mean_module=mean_module,
            data_covar_module=data_covar_module,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            temperature=temperature,
        )
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
        **kwargs: Any,
    ) -> "KroneckerMultiTaskMulticlassClassificationMixedGPModel":
        if kwargs.get("noise") is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support noise in condition_on_observations."
            )
        if isinstance(X, tuple):
            X = X[0]
        X = torch.as_tensor(
            X,
            device=self.train_inputs_raw[0].device,
            dtype=self.train_inputs_raw[0].dtype,
        )
        if X.ndim == 1:
            X = X.unsqueeze(0)
        if X.ndim != 2 or X.shape[-1] != self.train_inputs_raw[0].shape[-1]:
            raise ValueError(
                "X must have shape [n_new, d] with the same feature dimension as train_X."
            )
        Y = canonicalize_block_design_targets(X, Y, target_dtype=torch.long)
        if Y.shape[-1] != self.num_tasks:
            raise ValueError(
                f"Y must contain {self.num_tasks} tasks, got {Y.shape[-1]}."
            )
        self._validate_multiclass_targets(Y, num_classes=self.num_classes)

        new_X = torch.cat([self.train_inputs_raw[0], X], dim=-2)
        new_Y = torch.cat([self.train_targets, Y], dim=-2)
        new_model = self.__class__(
            train_X=new_X,
            train_Y=new_Y,
            cat_dims=self.cat_dims,
            num_classes=self.num_classes,
            rank=self.rank,
            likelihood=copy.deepcopy(self.likelihood),
            input_transform=copy.deepcopy(self.input_transform),
            mean_module=copy.deepcopy(self.model.mean_module),
            data_covar_module=copy.deepcopy(self.model.data_covar_module),
            num_inducing_points=self.num_inducing_points,
            inducing_points=self.inducing_points_raw.detach().clone(),
            learn_inducing_locations=self.learn_inducing_locations,
            temperature=self.temperature,
        )
        new_model.load_state_dict(copy.deepcopy(self.state_dict()), strict=False)
        new_model.eval()
        new_model.likelihood.eval()
        return new_model


__all__ = ["KroneckerMultiTaskMulticlassClassificationMixedGPModel"]
