from __future__ import annotations

from copy import deepcopy
from typing import Optional, Sequence

import torch
from torch import Tensor

from botorch.models.approximate_gp import ApproximateGPyTorchModel
from botorch.models.model import FantasizeMixin
from botorch.models.transforms.input import InputTransform

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import IndexKernel, Kernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.means import ConstantMean, Mean
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

from .models import (
    BinaryClassificationGPModel,
    _concat_optional_noise,
    _prepare_binary_conditioning_data,
    _prepare_train_Yvar,
    _select_inducing_points,
    _to_device_dtype_transform,
)


class _TaskProductKernel(Kernel):
    """Product kernel ``K_x(x, x') * K_task(t, t')`` for task-feature inputs.

    ``IndexKernel`` expects integer task indices, while BoTorch-style inputs often
    store the task feature in the same floating-point tensor as the continuous
    features.  This wrapper splits the task column, rounds/casts it to ``long``,
    and delegates the remaining columns to the data kernel.
    """

    has_lengthscale = False

    def __init__(
        self,
        data_kernel: Kernel,
        task_kernel: IndexKernel,
        task_feature: int,
        input_dim: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if input_dim < 2:
            raise ValueError("Multi-task models require at least one data feature and one task feature.")
        task_feature = int(task_feature)
        if task_feature < 0:
            task_feature = input_dim + task_feature
        if not 0 <= task_feature < input_dim:
            raise ValueError(
                f"task_feature={task_feature} is out of bounds for input_dim={input_dim}."
            )
        self.data_kernel = data_kernel
        self.task_kernel = task_kernel
        self.task_feature = task_feature
        self.input_dim = int(input_dim)
        self.data_dims = [i for i in range(input_dim) if i != task_feature]

    def _split(self, X: Tensor) -> tuple[Tensor, Tensor]:
        if X.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input feature dim {self.input_dim}, got {X.shape[-1]}."
            )
        X_data = X[..., self.data_dims]
        task = X[..., self.task_feature].round().long().unsqueeze(-1)
        return X_data, task

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ):
        x1_data, t1 = self._split(x1)
        x2_data, t2 = self._split(x2)
        data_covar = self.data_kernel(
            x1_data,
            x2_data,
            diag=diag,
            last_dim_is_batch=last_dim_is_batch,
            **params,
        )
        task_covar = self.task_kernel(t1, t2, diag=diag, **params)
        return data_covar * task_covar


class _LatentMultiTaskBinarySVGP(ApproximateGP):
    """Latent variational GP for binary multi-task classification."""

    def __init__(
        self,
        inducing_points: Tensor,
        train_inputs: Tensor,
        train_targets: Tensor,
        num_tasks: int,
        task_feature: int = -1,
        train_Yvar: Optional[Tensor] = None,
        mean_module: Optional[Mean] = None,
        data_covar_module: Optional[Kernel] = None,
        task_covar_module: Optional[IndexKernel] = None,
        rank: int = 1,
        learn_inducing_locations: bool = True,
    ) -> None:
        ref_x = train_inputs
        ref_dtype = ref_x.dtype
        ref_device = ref_x.device
        input_dim = ref_x.shape[-1]

        inducing_points = inducing_points.to(device=ref_device, dtype=ref_dtype)
        train_targets = train_targets.to(device=ref_device, dtype=ref_dtype)
        if train_Yvar is not None:
            train_Yvar = train_Yvar.to(device=ref_device, dtype=ref_dtype)

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.shape[-2]
        ).to(device=ref_device, dtype=ref_dtype)
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)

        task_feature_pos = task_feature if task_feature >= 0 else input_dim + task_feature
        data_dim = input_dim - 1
        if mean_module is None:
            mean_module = ConstantMean()
        if data_covar_module is None:
            data_covar_module = ScaleKernel(RBFKernel(ard_num_dims=data_dim))
        if task_covar_module is None:
            task_covar_module = IndexKernel(num_tasks=int(num_tasks), rank=int(rank))

        self.mean_module = mean_module.to(device=ref_device, dtype=ref_dtype)
        self.data_covar_module = data_covar_module.to(device=ref_device, dtype=ref_dtype)
        self.task_covar_module = task_covar_module.to(device=ref_device, dtype=ref_dtype)
        self.covar_module = _TaskProductKernel(
            data_kernel=self.data_covar_module,
            task_kernel=self.task_covar_module,
            task_feature=task_feature_pos,
            input_dim=input_dim,
        ).to(device=ref_device, dtype=ref_dtype)

        self.num_tasks = int(num_tasks)
        self.task_feature = int(task_feature_pos)
        self.rank = int(rank)
        self.train_inputs = (ref_x,)
        self.train_targets = train_targets
        self.train_Yvar = train_Yvar
        self.to(device=ref_device, dtype=ref_dtype)

    def forward(self, X: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultivariateNormal(mean_x, covar_x)


class MultiTaskBinaryClassificationGPModel(BinaryClassificationGPModel, FantasizeMixin):
    """BoTorch-style binary classification GP with an explicit task feature.

    The input tensor must be in long format and contain a task-id column.
    For example, with ``task_feature=-1``, ``train_X[..., -1]`` contains integer
    task ids in ``0, ..., num_tasks - 1``.  The latent covariance is an ICM-style
    product of a data kernel and an ``IndexKernel`` over task ids.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        *,
        num_tasks: int,
        task_feature: int = -1,
        rank: int = 1,
        likelihood: Optional[BernoulliLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        mean_module: Optional[Mean] = None,
        data_covar_module: Optional[Kernel] = None,
        task_covar_module: Optional[IndexKernel] = None,
        num_inducing_points: int = 20,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
    ) -> None:
        if train_X.ndim != 2:
            raise ValueError(f"train_X must be [N, d], got shape={tuple(train_X.shape)}.")
        if train_X.shape[-1] < 2:
            raise ValueError("train_X must include at least one data feature and one task feature.")

        task_feature_pos = int(task_feature)
        if task_feature_pos < 0:
            task_feature_pos = train_X.shape[-1] + task_feature_pos
        if not 0 <= task_feature_pos < train_X.shape[-1]:
            raise ValueError(
                f"task_feature={task_feature} is out of bounds for input dim {train_X.shape[-1]}."
            )
        self._validate_task_feature(train_X, num_tasks=num_tasks, task_feature=task_feature_pos)

        if train_Y.ndim > 1 and train_Y.shape[-1] == 1:
            train_Y = train_Y.squeeze(-1)
        train_Y = train_Y.to(dtype=train_X.dtype)

        train_Yvar = _prepare_train_Yvar(train_Yvar, train_X=train_X, train_Y=train_Y)

        self.num_tasks = int(num_tasks)
        self.task_feature = int(task_feature_pos)
        self.rank = int(rank)
        self.train_inputs = (train_X,)
        self.train_targets = train_Y
        self.train_Yvar = train_Yvar
        self.train_inputs_raw = (train_X,)
        self._train_targets = train_Y

        input_transform = _to_device_dtype_transform(input_transform, train_X)
        if input_transform is not None:
            input_transform.train()
            with torch.no_grad():
                transformed_train_X = input_transform(train_X).detach().clone()
            input_transform.eval()
            self._validate_task_feature(
                transformed_train_X,
                num_tasks=num_tasks,
                task_feature=task_feature_pos,
                name="transformed train_X",
            )
        else:
            transformed_train_X = train_X.detach().clone()

        inducing_points = _select_inducing_points(
            transformed_train_X,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
        )
        self._validate_task_feature(
            inducing_points,
            num_tasks=num_tasks,
            task_feature=task_feature_pos,
            name="inducing_points",
        )

        latent_model = _LatentMultiTaskBinarySVGP(
            inducing_points=inducing_points,
            train_inputs=transformed_train_X,
            train_targets=train_Y,
            train_Yvar=train_Yvar,
            num_tasks=num_tasks,
            task_feature=task_feature_pos,
            mean_module=mean_module,
            data_covar_module=data_covar_module,
            task_covar_module=task_covar_module,
            rank=rank,
            learn_inducing_locations=learn_inducing_locations,
        )

        likelihood = likelihood or BernoulliLikelihood()
        ApproximateGPyTorchModel.__init__(
            self,
            model=latent_model,
            likelihood=likelihood,
            num_outputs=1,
        )
        self.input_transform = input_transform
        self.to(train_X)

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
        if bool((task_values.round() < 0).any()) or bool((task_values.round() >= int(num_tasks)).any()):
            bad = torch.unique(task_values[(task_values.round() < 0) | (task_values.round() >= int(num_tasks))])
            raise ValueError(
                f"{name} contains invalid task ids {bad.detach().cpu().tolist()}; "
                f"expected ids in [0, {int(num_tasks) - 1}]."
            )

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        noise: Optional[Tensor] = None,
        **kwargs,
    ) -> "MultiTaskBinaryClassificationGPModel":
        X_new, Y_new, Yvar_new = _prepare_binary_conditioning_data(X, Y, noise)
        self._validate_task_feature(
            X_new,
            num_tasks=self.num_tasks,
            task_feature=self.task_feature,
            name="new X",
        )

        train_X_old = self.train_inputs_raw[0]
        train_Y_old = self.train_targets
        if train_Y_old.ndim > 1 and train_Y_old.shape[-1] == 1:
            train_Y_old = train_Y_old.squeeze(-1)

        train_X_full = torch.cat(
            [train_X_old, X_new.to(dtype=train_X_old.dtype, device=train_X_old.device)],
            dim=0,
        )
        train_Y_full = torch.cat(
            [train_Y_old, Y_new.to(dtype=train_Y_old.dtype, device=train_Y_old.device)],
            dim=0,
        )
        train_Yvar_full = _concat_optional_noise(
            old_Y=train_Y_old,
            old_Yvar=self.train_Yvar,
            new_Y=Y_new,
            new_Yvar=Yvar_new,
            dtype=train_X_old.dtype,
            device=train_X_old.device,
        )

        inducing_points = self.model.variational_strategy.inducing_points.detach().clone()
        new_model = self.__class__(
            train_X=train_X_full,
            train_Y=train_Y_full,
            train_Yvar=train_Yvar_full,
            num_tasks=self.num_tasks,
            task_feature=self.task_feature,
            rank=self.rank,
            likelihood=deepcopy(self.likelihood),
            input_transform=deepcopy(self.input_transform),
            mean_module=deepcopy(self.model.mean_module),
            data_covar_module=deepcopy(self.model.data_covar_module),
            task_covar_module=deepcopy(self.model.task_covar_module),
            num_inducing_points=inducing_points.shape[-2],
            inducing_points=inducing_points,
            learn_inducing_locations=getattr(
                self.model.variational_strategy,
                "learn_inducing_locations",
                True,
            ),
        )
        new_model.load_state_dict(self.state_dict(), strict=False)
        new_model.eval()
        return new_model
