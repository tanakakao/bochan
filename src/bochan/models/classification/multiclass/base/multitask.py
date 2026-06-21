from __future__ import annotations

import copy
from typing import Any, Optional, Sequence

import torch
from botorch.models.transforms.input import InputTransform
from gpytorch.kernels import IndexKernel, Kernel
from gpytorch.likelihoods import SoftmaxLikelihood
from gpytorch.means import Mean
from torch import Tensor

from bochan.models.classification.binary.base.multitask import _TaskProductKernel
from bochan.models.components.mixed_multitask import (
    build_mixed_task_data_kernel,
    normalize_mixed_task_dims,
    normalize_task_feature,
    transform_mixed_task_inputs,
    validate_mixed_task_input_transform,
)
from bochan.models.components.multiclass import (
    apply_input_transform_for_eval,
    apply_input_transform_for_training,
    build_default_multiclass_covar_module,
    clone_input_transform,
    infer_num_classes,
    prepare_class_targets,
    to_device_dtype_transform,
)

from .models import _BaseMulticlassClassificationModel, _LatentMulticlassSVGP


class _BaseMultiTaskMulticlassClassificationGPModel(
    _BaseMulticlassClassificationModel
):
    """Shared long-format multiclass multi-task implementation."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_tasks: int,
        task_feature: int = -1,
        rank: int = 1,
        cat_dims: Optional[Sequence[int]] = None,
        num_classes: Optional[int] = None,
        likelihood: Optional[SoftmaxLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        mean_module: Optional[Mean] = None,
        data_covar_module: Optional[Kernel] = None,
        task_covar_module: Optional[IndexKernel] = None,
        num_inducing_points: int = 128,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        temperature: float = 1.0,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        if train_X.ndim != 2 or train_X.shape[-1] < 2:
            raise ValueError(
                "train_X must have shape [N, d] with at least one data feature "
                "and one task feature."
            )
        task_feature = normalize_task_feature(task_feature, train_X.shape[-1])
        self._validate_task_feature(
            train_X,
            num_tasks=num_tasks,
            task_feature=task_feature,
        )

        normalized_cat_dims: Optional[list[int]] = None
        if cat_dims is not None:
            normalized_cat_dims, task_feature = normalize_mixed_task_dims(
                cat_dims,
                task_feature=task_feature,
                d=train_X.shape[-1],
            )
            validate_mixed_task_input_transform(
                train_X,
                input_transform,
                cat_dims=normalized_cat_dims,
                task_feature=task_feature,
            )

        num_classes = infer_num_classes(train_Y, num_classes)
        train_Y = prepare_class_targets(train_Y, train_X, num_classes=num_classes)
        input_transform = to_device_dtype_transform(
            clone_input_transform(input_transform),
            train_X,
        )
        train_X_tf = apply_input_transform_for_training(
            train_X,
            input_transform,
            name=f"{self.__class__.__name__}.input_transform",
        )
        self._validate_task_feature(
            train_X_tf,
            num_tasks=num_tasks,
            task_feature=task_feature,
            name="transformed train_X",
        )

        data_dims = [i for i in range(train_X.shape[-1]) if i != task_feature]
        if data_covar_module is None:
            if normalized_cat_dims is None:
                data_covar_module = build_default_multiclass_covar_module(
                    train_X_tf[..., data_dims],
                    num_classes=num_classes,
                )
            else:
                data_covar_module = build_mixed_task_data_kernel(
                    d=train_X.shape[-1],
                    cat_dims=normalized_cat_dims,
                    task_feature=task_feature,
                    batch_shape=torch.Size([num_classes]),
                )
        if task_covar_module is None:
            task_covar_module = IndexKernel(
                num_tasks=int(num_tasks),
                rank=int(rank),
                batch_shape=torch.Size([num_classes]),
            )

        covar_module = _TaskProductKernel(
            data_kernel=data_covar_module,
            task_kernel=task_covar_module,
            task_feature=task_feature,
            input_dim=train_X.shape[-1],
        )
        latent_model = _LatentMulticlassSVGP(
            train_X=train_X_tf,
            train_Y=train_Y,
            num_classes=num_classes,
            inducing_points=inducing_points,
            num_inducing_points=num_inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            mean_module=mean_module,
            covar_module=covar_module,
        )
        latent_model.data_covar_module = data_covar_module
        latent_model.task_covar_module = task_covar_module
        latent_model.task_feature = int(task_feature)
        latent_model.num_tasks = int(num_tasks)
        latent_model.rank = int(rank)

        likelihood = likelihood or SoftmaxLikelihood(
            num_features=num_classes,
            num_classes=num_classes,
            mixing_weights=False,
        )
        super().__init__(
            latent_model=latent_model,
            likelihood=likelihood,
            train_X=train_X,
            train_Y=train_Y,
            num_classes=num_classes,
            input_transform=input_transform,
            cat_dims=normalized_cat_dims,
            num_inducing_points=num_inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            temperature=temperature,
        )
        self.num_tasks = int(num_tasks)
        self.task_feature = int(task_feature)
        self.rank = int(rank)
        self.cat_dims = normalized_cat_dims

    @staticmethod
    def _validate_task_feature(
        X: Tensor,
        *,
        num_tasks: int,
        task_feature: int,
        name: str = "train_X",
    ) -> None:
        values = X[..., task_feature]
        if not torch.allclose(values, values.round()):
            raise ValueError(f"{name} task feature must be integer-coded.")
        invalid = (values < 0) | (values >= int(num_tasks))
        if bool(invalid.any()):
            bad = torch.unique(values[invalid]).detach().cpu().tolist()
            raise ValueError(
                f"{name} contains invalid task ids {bad}; expected ids in "
                f"[0, {int(num_tasks) - 1}]."
            )

    @property
    def task_covar_matrix(self) -> Tensor:
        return self.model.task_covar_module.covar_matrix.to_dense()

    def transform_inputs(self, X: Tensor) -> Tensor:
        if self.cat_dims is not None:
            return transform_mixed_task_inputs(
                X,
                self.input_transform,
                cat_dims=self.cat_dims,
                task_feature=self.task_feature,
            )
        X_tf = apply_input_transform_for_eval(X, self.input_transform, cat_dims=None)
        self._validate_task_feature(
            X_tf,
            num_tasks=self.num_tasks,
            task_feature=self.task_feature,
            name="transformed X",
        )
        return X_tf

    def _extra_constructor_kwargs(self) -> dict[str, Any]:
        return {}

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        **kwargs: Any,
    ) -> "_BaseMultiTaskMulticlassClassificationGPModel":
        if kwargs.get("noise") is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support noise."
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
        self._validate_task_feature(
            X,
            num_tasks=self.num_tasks,
            task_feature=self.task_feature,
            name="new X",
        )
        Y = prepare_class_targets(Y, X, num_classes=self.num_classes)
        new_X = torch.cat([self.train_inputs_raw[0], X], dim=-2)
        new_Y = torch.cat([self.train_targets, Y], dim=0)

        new_model = self.__class__(
            train_X=new_X,
            train_Y=new_Y,
            num_tasks=self.num_tasks,
            task_feature=self.task_feature,
            rank=self.rank,
            num_classes=self.num_classes,
            likelihood=copy.deepcopy(self.likelihood),
            input_transform=clone_input_transform(self.input_transform),
            mean_module=copy.deepcopy(self.model.mean_module),
            data_covar_module=copy.deepcopy(self.model.data_covar_module),
            task_covar_module=copy.deepcopy(self.model.task_covar_module),
            num_inducing_points=self.num_inducing_points,
            inducing_points=self.model.variational_strategy.inducing_points.detach().clone(),
            learn_inducing_locations=self.learn_inducing_locations,
            temperature=self.temperature,
            **self._extra_constructor_kwargs(),
        )
        new_model.load_state_dict(copy.deepcopy(self.state_dict()), strict=False)
        new_model.eval()
        new_model.likelihood.eval()
        return new_model


class MultiTaskMulticlassClassificationGPModel(
    _BaseMultiTaskMulticlassClassificationGPModel
):
    """Continuous-input long-format multiclass multi-task GP."""


class MultiTaskMulticlassClassificationMixedGPModel(
    _BaseMultiTaskMulticlassClassificationGPModel
):
    """Mixed-input long-format multiclass multi-task GP."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=cat_dims,
            **kwargs,
        )

    def _extra_constructor_kwargs(self) -> dict[str, Any]:
        return {"cat_dims": self.cat_dims}


__all__ = [
    "MultiTaskMulticlassClassificationGPModel",
    "MultiTaskMulticlassClassificationMixedGPModel",
]
