from __future__ import annotations

import copy
from typing import Any, Optional, Sequence

import torch
from botorch.models.transforms.input import InputTransform
from gpytorch.kernels import IndexKernel, Kernel
from gpytorch.means import Mean
from torch import Tensor

from bochan.fit.ordinal import fit_ordinal_gp
from bochan.models.components.mixed_multitask import (
    MixedTaskProductKernel,
    build_mixed_task_data_kernel,
    normalize_mixed_task_dims,
    transform_mixed_task_inputs,
    validate_mixed_task_input_transform,
)

from .multitask import MultiTaskOrdinalGPModel


class MultiTaskOrdinalMixedGPModel(MultiTaskOrdinalGPModel):
    """Long-format mixed-input ordinal multi-task GP.

    The model supports different observation locations per task. Categorical data
    features are handled by a mixed kernel, while the explicit task-id column is
    handled by an ``IndexKernel``. All tasks share one ordinal class definition
    and one set of cutpoints.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        num_classes: Optional[int] = None,
        *,
        num_tasks: int,
        task_feature: int = -1,
        rank: int = 1,
        inducing_points_num: int = 128,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        mean_module: Optional[Mean] = None,
        data_covar_module: Optional[Kernel] = None,
        task_covar_module: Optional[IndexKernel] = None,
        input_transform: Optional[InputTransform] = None,
        eps: float = 1e-8,
        init_gap: float = 1.0,
        fix_first_cutpoint: bool = True,
        conditioning_steps: int = 50,
        conditioning_lr: Optional[float] = None,
        conditioning_batch_size: Optional[int] = None,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        cat_dims, task_feature = normalize_mixed_task_dims(
            cat_dims,
            task_feature=task_feature,
            d=train_X.shape[-1],
        )
        validate_mixed_task_input_transform(
            train_X,
            input_transform,
            cat_dims=cat_dims,
            task_feature=task_feature,
        )
        if data_covar_module is None:
            data_covar_module = build_mixed_task_data_kernel(
                d=train_X.shape[-1],
                cat_dims=cat_dims,
                task_feature=task_feature,
            )

        self.cat_dims = list(cat_dims)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            num_classes=num_classes,
            num_tasks=num_tasks,
            task_feature=task_feature,
            rank=rank,
            inducing_points_num=inducing_points_num,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            mean_module=mean_module,
            data_covar_module=data_covar_module,
            task_covar_module=task_covar_module,
            input_transform=input_transform,
            eps=eps,
            init_gap=init_gap,
            fix_first_cutpoint=fix_first_cutpoint,
            conditioning_steps=conditioning_steps,
            conditioning_lr=conditioning_lr,
            conditioning_batch_size=conditioning_batch_size,
        )
        self.cat_dims = list(cat_dims)
        self.model.covar_module = MixedTaskProductKernel(
            self.model.data_covar_module,
            self.model.task_covar_module,
            task_feature=self.task_feature,
            input_dim=train_X.shape[-1],
        ).to(device=train_X.device, dtype=train_X.dtype)

    def transform_inputs(self, X: Tensor) -> Tensor:
        return transform_mixed_task_inputs(
            X,
            self.input_transform,
            cat_dims=self.cat_dims,
            task_feature=self.task_feature,
        )

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        refit: bool = True,
        num_steps: Optional[int] = None,
        lr: Optional[float] = None,
        batch_size: Optional[int] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "MultiTaskOrdinalMixedGPModel":
        if kwargs.get("noise") is not None:
            raise NotImplementedError(
                f"noise is not supported for {self.__class__.__name__}."
            )

        X = self._canonicalize_observation_X(X)
        Y = self._canonicalize_new_Y(Y, n=X.shape[-2])

        new_train_X = torch.cat([self.train_inputs_raw[0], X], dim=-2)
        new_train_Y = torch.cat([self.train_targets, Y], dim=0)

        new_model = self.__class__(
            train_X=new_train_X,
            train_Y=new_train_Y,
            cat_dims=self.cat_dims,
            num_classes=self.num_classes,
            num_tasks=self.num_tasks,
            task_feature=self.task_feature,
            rank=self.rank,
            inducing_points_num=self.model.variational_strategy.inducing_points.shape[-2],
            inducing_points=self.inducing_points_raw.detach().clone(),
            learn_inducing_locations=self.learn_inducing_locations,
            mean_module=copy.deepcopy(self.model.mean_module),
            data_covar_module=copy.deepcopy(self.model.data_covar_module),
            task_covar_module=copy.deepcopy(self.model.task_covar_module),
            input_transform=copy.deepcopy(self.input_transform),
            eps=self.eps,
            init_gap=self.init_gap,
            fix_first_cutpoint=self.fix_first_cutpoint,
            conditioning_steps=self.conditioning_steps,
            conditioning_lr=self.conditioning_lr,
            conditioning_batch_size=self.conditioning_batch_size,
        )
        new_model.load_state_dict(copy.deepcopy(self.state_dict()), strict=True)

        if refit:
            steps = self.conditioning_steps if num_steps is None else int(num_steps)
            refit_lr = self.conditioning_lr if lr is None else float(lr)
            refit_bs = self.conditioning_batch_size if batch_size is None else batch_size
            fit_ordinal_gp(
                new_model,
                num_epochs=steps,
                lr=refit_lr,
                batch_size=refit_bs,
                verbose=verbose,
            )
        else:
            new_model.eval()
            new_model.likelihood.eval()
        return new_model


__all__ = ["MultiTaskOrdinalMixedGPModel"]
