from __future__ import annotations

import copy
from typing import Optional

import torch
from torch import Tensor

from botorch.models.approximate_gp import ApproximateGPyTorchModel
from botorch.models.transforms.input import InputTransform

from gpytorch.kernels import IndexKernel, Kernel, RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean, Mean

from bochan.fit.ordinal import fit_ordinal_gp
from bochan.likelihoods.ordinal import OrdinalLogitLikelihood

from .models import (
    OrdinalGPModel,
    _OrdinalLatentGP,
    _canonicalize_inducing_points,
    _infer_num_classes_from_train_Y,
    _prepare_input_transform,
    _transform_tensor_for_training,
)
from bochan.models.classification.binary.base.multitask import _TaskProductKernel


class _MultiTaskOrdinalLatentGP(_OrdinalLatentGP):
    """Latent variational GP for ordinal multi-task regression/classification."""

    def __init__(
        self,
        train_X: Tensor,
        num_tasks: int,
        task_feature: int = -1,
        rank: int = 1,
        inducing_points_num: int = 128,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        mean_module: Optional[Mean] = None,
        data_covar_module: Optional[Kernel] = None,
        task_covar_module: Optional[IndexKernel] = None,
    ) -> None:
        if train_X.ndim != 2:
            raise ValueError(f"train_X must be [N, d], got shape={tuple(train_X.shape)}.")
        if train_X.shape[-1] < 2:
            raise ValueError("train_X must include at least one data feature and one task feature.")

        n, d = train_X.shape
        task_feature_pos = int(task_feature)
        if task_feature_pos < 0:
            task_feature_pos = d + task_feature_pos
        if not 0 <= task_feature_pos < d:
            raise ValueError(f"task_feature={task_feature} is out of bounds for input dim {d}.")

        if inducing_points is None:
            m = min(int(inducing_points_num), n)
            perm = torch.randperm(n, device=train_X.device)[:m]
            inducing_points = train_X[perm].clone()
        else:
            inducing_points = _canonicalize_inducing_points(
                inducing_points,
                d=d,
                device=train_X.device,
                dtype=train_X.dtype,
            )

        super().__init__(
            train_X=train_X,
            inducing_points_num=inducing_points_num,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            mean_module=mean_module,
            covar_module=ScaleKernel(RBFKernel(ard_num_dims=d)),
        )

        if mean_module is None:
            mean_module = ConstantMean()
        if data_covar_module is None:
            data_covar_module = ScaleKernel(RBFKernel(ard_num_dims=d - 1))
        if task_covar_module is None:
            task_covar_module = IndexKernel(num_tasks=int(num_tasks), rank=int(rank))

        self.mean_module = mean_module.to(device=train_X.device, dtype=train_X.dtype)
        self.data_covar_module = data_covar_module.to(device=train_X.device, dtype=train_X.dtype)
        self.task_covar_module = task_covar_module.to(device=train_X.device, dtype=train_X.dtype)
        self.covar_module = _TaskProductKernel(
            data_kernel=self.data_covar_module,
            task_kernel=self.task_covar_module,
            task_feature=task_feature_pos,
            input_dim=d,
        ).to(device=train_X.device, dtype=train_X.dtype)

        self.num_tasks = int(num_tasks)
        self.task_feature = int(task_feature_pos)
        self.rank = int(rank)
        self.train_inputs = (train_X,)
        self.to(device=train_X.device, dtype=train_X.dtype)


class MultiTaskOrdinalGPModel(OrdinalGPModel):
    """BoTorch-compatible ordinal GP with an explicit task feature.

    The input tensor must be in long format and contain a task-id column.
    For example, with ``task_feature=-1``, ``train_X[..., -1]`` contains integer
    task ids in ``0, ..., num_tasks - 1``.  The latent covariance is an ICM-style
    product of a data kernel and an ``IndexKernel`` over task ids.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
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
        raw_train_X = self._canonicalize_train_X(train_X)
        if raw_train_X.shape[-1] < 2:
            raise ValueError("train_X must include at least one data feature and one task feature.")

        task_feature_pos = int(task_feature)
        if task_feature_pos < 0:
            task_feature_pos = raw_train_X.shape[-1] + task_feature_pos
        if not 0 <= task_feature_pos < raw_train_X.shape[-1]:
            raise ValueError(
                f"task_feature={task_feature} is out of bounds for input dim {raw_train_X.shape[-1]}."
            )
        self._validate_task_feature(raw_train_X, num_tasks=num_tasks, task_feature=task_feature_pos)

        train_Y = self._canonicalize_train_Y(
            train_Y,
            raw_train_X.shape[-2],
            raw_train_X.device,
        )

        if num_classes is None:
            num_classes = _infer_num_classes_from_train_Y(train_Y)
        else:
            num_classes = int(num_classes)

        input_transform = _prepare_input_transform(input_transform, raw_train_X)

        if inducing_points is None:
            m = min(int(inducing_points_num), raw_train_X.shape[-2])
            perm = torch.randperm(raw_train_X.shape[-2], device=raw_train_X.device)[:m]
            raw_inducing_points = raw_train_X[perm].clone()
        else:
            raw_inducing_points = _canonicalize_inducing_points(
                inducing_points,
                d=raw_train_X.shape[-1],
                device=raw_train_X.device,
                dtype=raw_train_X.dtype,
            )
        self._validate_task_feature(
            raw_inducing_points,
            num_tasks=num_tasks,
            task_feature=task_feature_pos,
            name="inducing_points",
        )

        train_X_tf = _transform_tensor_for_training(
            raw_train_X,
            input_transform=input_transform,
            name="MultiTaskOrdinalGPModel.input_transform",
        )
        inducing_points_tf = _transform_tensor_for_training(
            raw_inducing_points,
            input_transform=input_transform,
            name="MultiTaskOrdinalGPModel.input_transform",
        )
        self._validate_task_feature(
            train_X_tf,
            num_tasks=num_tasks,
            task_feature=task_feature_pos,
            name="transformed train_X",
        )
        self._validate_task_feature(
            inducing_points_tf,
            num_tasks=num_tasks,
            task_feature=task_feature_pos,
            name="transformed inducing_points",
        )

        latent_model = _MultiTaskOrdinalLatentGP(
            train_X=train_X_tf,
            num_tasks=num_tasks,
            task_feature=task_feature_pos,
            rank=rank,
            inducing_points_num=inducing_points_num,
            inducing_points=inducing_points_tf,
            learn_inducing_locations=learn_inducing_locations,
            mean_module=mean_module,
            data_covar_module=data_covar_module,
            task_covar_module=task_covar_module,
        )

        likelihood = OrdinalLogitLikelihood(
            num_classes=num_classes,
            eps=eps,
            init_gap=init_gap,
            fix_first_cutpoint=fix_first_cutpoint,
        )

        ApproximateGPyTorchModel.__init__(
            self,
            model=latent_model,
            likelihood=likelihood,
            num_outputs=1,
        )

        self.input_transform = input_transform
        self.train_inputs = (train_X_tf,)
        self.train_inputs_raw = (raw_train_X,)
        self.train_targets = train_Y
        self.model.train_inputs = self.train_inputs
        self.model.train_targets = self.train_targets

        self.inducing_points_raw = raw_inducing_points
        self.inducing_points = inducing_points_tf

        self.num_classes = int(num_classes)
        self.num_tasks = int(num_tasks)
        self.task_feature = int(task_feature_pos)
        self.rank = int(rank)
        self.inducing_points_num = int(inducing_points_num)
        self.learn_inducing_locations = bool(learn_inducing_locations)

        self.eps = float(eps)
        self.init_gap = float(init_gap)
        self.fix_first_cutpoint = bool(fix_first_cutpoint)
        self.conditioning_steps = int(conditioning_steps)
        self.conditioning_lr = conditioning_lr
        self.conditioning_batch_size = conditioning_batch_size

        self.to(device=raw_train_X.device, dtype=raw_train_X.dtype)

    @staticmethod
    def _validate_task_feature(
        X: Tensor,
        *,
        num_tasks: int,
        task_feature: int,
        name: str = "train_X",
    ) -> None:
        task_values = X[..., task_feature]
        if not torch.allclose(task_values, task_values.round()):
            raise ValueError(f"{name} task feature must be integer-coded.")
        invalid = (task_values.round() < 0) | (task_values.round() >= int(num_tasks))
        if bool(invalid.any()):
            bad = torch.unique(task_values[invalid]).detach().cpu().tolist()
            raise ValueError(
                f"{name} contains invalid task ids {bad}; "
                f"expected ids in [0, {int(num_tasks) - 1}]."
            )

    def _canonicalize_observation_X(self, X: Tensor) -> Tensor:
        X = super()._canonicalize_observation_X(X)
        self._validate_task_feature(
            X,
            num_tasks=self.num_tasks,
            task_feature=self.task_feature,
            name="observation X",
        )
        return X

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        refit: bool = True,
        num_steps: Optional[int] = None,
        lr: Optional[float] = None,
        batch_size: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ) -> "MultiTaskOrdinalGPModel":
        if kwargs.get("noise") is not None:
            raise NotImplementedError("noise is not supported for MultiTaskOrdinalGPModel.")

        X = self._canonicalize_observation_X(X)
        Y = self._canonicalize_new_Y(Y, n=X.shape[-2])

        new_train_X = torch.cat([self.train_input_raw, X], dim=-2)
        new_train_Y = torch.cat([self.train_targets, Y], dim=0)

        new_model = self.__class__(
            train_X=new_train_X,
            train_Y=new_train_Y,
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
