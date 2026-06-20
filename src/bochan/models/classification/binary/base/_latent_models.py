from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
from torch import Tensor
from gpytorch.models import ApproximateGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean, Mean
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)

from botorch.models.model import Model
from botorch.models.kernels.categorical import CategoricalKernel
from gpytorch.constraints import GreaterThan


def _as_train_tensor(train_inputs: Tensor | tuple[Tensor, ...]) -> Tensor:
    """train_inputs から基準 Tensor を取り出す。"""
    return train_inputs[0] if isinstance(train_inputs, tuple) else train_inputs


def _build_default_cont_kernel(
    *,
    cont_dims: Sequence[int],
    ref_x: Tensor,
    cont_kernel_factory: Optional[
        Callable[[torch.Size, int, Optional[list[int]]], Kernel]
    ] = None,
) -> Optional[Kernel]:
    """連続変数用 kernel を作る。

    multiclass base model に合わせ、既定値は ARD Matérn 2.5 とする。
    """
    if len(cont_dims) == 0:
        return None

    if cont_kernel_factory is not None:
        kernel = cont_kernel_factory(
            torch.Size([]),
            len(cont_dims),
            list(cont_dims),
        )
        return kernel.to(device=ref_x.device, dtype=ref_x.dtype)

    kernel = MaternKernel(
        nu=2.5,
        ard_num_dims=len(cont_dims),
        active_dims=tuple(cont_dims),
        batch_shape=torch.Size([]),
    )
    return kernel.to(device=ref_x.device, dtype=ref_x.dtype)


def _build_default_cat_kernel(
    *,
    cat_dims: Sequence[int],
    ref_x: Tensor,
) -> Optional[Kernel]:
    """カテゴリ変数用 kernel を作る。"""
    if len(cat_dims) == 0:
        return None

    batch_shape = ref_x.shape[:-2]
    kernel = CategoricalKernel(
        batch_shape=batch_shape,
        ard_num_dims=len(cat_dims),
        active_dims=tuple(cat_dims),
        lengthscale_constraint=GreaterThan(1e-6),
    )
    return kernel.to(device=ref_x.device, dtype=ref_x.dtype)


def _build_default_mixed_covar_module(
    *,
    ref_x: Tensor,
    cont_dims: Sequence[int],
    cat_dims: Sequence[int],
    cont_kernel_factory: Optional[
        Callable[[torch.Size, int, Optional[list[int]]], Kernel]
    ] = None,
) -> Kernel:
    """mixed input 用のデフォルト covar_module を作る。

    multiclass mixed model と同様に、加法項と積項で別々の kernel
    instance を用いる。

    - continuous only: ``cont_1``
    - categorical only: ``cat_1``
    - mixed: ``cont_1 + cat_1 + cont_2 * cat_2``
    """
    cont_1 = _build_default_cont_kernel(
        cont_dims=cont_dims,
        ref_x=ref_x,
        cont_kernel_factory=cont_kernel_factory,
    )
    cat_1 = _build_default_cat_kernel(
        cat_dims=cat_dims,
        ref_x=ref_x,
    )

    if cont_1 is None and cat_1 is None:
        raise ValueError("At least one kernel component must be available.")

    if cont_1 is None:
        base_kernel = cat_1
    elif cat_1 is None:
        base_kernel = cont_1
    else:
        cont_2 = _build_default_cont_kernel(
            cont_dims=cont_dims,
            ref_x=ref_x,
            cont_kernel_factory=cont_kernel_factory,
        )
        cat_2 = _build_default_cat_kernel(
            cat_dims=cat_dims,
            ref_x=ref_x,
        )
        if cont_2 is None or cat_2 is None:
            raise RuntimeError("Failed to build mixed interaction kernels.")
        base_kernel = cont_1 + cat_1 + cont_2 * cat_2

    covar_module = ScaleKernel(base_kernel, batch_shape=torch.Size([]))
    return covar_module.to(device=ref_x.device, dtype=ref_x.dtype)


class _LatentBinarySVGP(ApproximateGP):
    """2値分類用の latent SVGP。

    ``train_inputs`` を基準に variational distribution、mean、kernel、
    parameter、buffer の dtype/device を統一する。
    """

    def __init__(
        self,
        inducing_points: Tensor,
        train_inputs: Tensor | tuple[Tensor, ...],
        train_targets: Tensor,
        train_Yvar: Optional[Tensor] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        learn_inducing_locations: bool = True,
    ) -> None:
        ref_x = _as_train_tensor(train_inputs)
        ref_dtype = ref_x.dtype
        ref_device = ref_x.device

        inducing_points = inducing_points.to(device=ref_device, dtype=ref_dtype)
        train_targets = train_targets.to(device=ref_device, dtype=ref_dtype)
        if train_Yvar is not None:
            train_Yvar = train_Yvar.to(device=ref_device, dtype=ref_dtype)

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(-2),
        ).to(device=ref_device, dtype=ref_dtype)

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )

        super().__init__(variational_strategy)

        if mean_module is None:
            mean_module = ConstantMean()

        if covar_module is None:
            covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=inducing_points.shape[-1],
                    batch_shape=torch.Size([]),
                ),
                batch_shape=torch.Size([]),
            )

        self.mean_module = mean_module.to(device=ref_device, dtype=ref_dtype)
        self.covar_module = covar_module.to(device=ref_device, dtype=ref_dtype)

        self.train_inputs = (ref_x,)
        self.train_targets = train_targets
        self.train_Yvar = train_Yvar

        self.to(device=ref_device, dtype=ref_dtype)

    def transform_inputs(self, X: Tensor) -> Tensor:
        """外側wrapperで変換済みの内部入力をそのまま返す。"""
        return X

    def forward(self, X: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultivariateNormal(mean_x, covar_x)


class _LatentMixedBinarySVGP(ApproximateGP):
    """mixed input（連続 + カテゴリ）対応の latent SVGP。

    デフォルト kernel は multiclass mixed model と同様に
    ``cont_1 + cat_1 + cont_2 * cat_2`` とする。
    """

    def __init__(
        self,
        inducing_points: Tensor,
        cat_dims: Sequence[int],
        train_inputs: Tensor | tuple[Tensor, ...],
        train_targets: Tensor,
        train_Yvar: Optional[Tensor] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        cont_kernel_factory: Optional[
            Callable[[torch.Size, int, Optional[list[int]]], Kernel]
        ] = None,
        learn_inducing_locations: bool = True,
    ) -> None:
        ref_x = _as_train_tensor(train_inputs)
        ref_dtype = ref_x.dtype
        ref_device = ref_x.device
        d = ref_x.shape[-1]

        cat_dims = sorted(int(i) for i in cat_dims)
        cont_dims = [i for i in range(d) if i not in cat_dims]

        inducing_points = inducing_points.to(device=ref_device, dtype=ref_dtype)
        train_targets = train_targets.to(device=ref_device, dtype=ref_dtype)
        if train_Yvar is not None:
            train_Yvar = train_Yvar.to(device=ref_device, dtype=ref_dtype)

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(-2),
        ).to(device=ref_device, dtype=ref_dtype)

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )

        super().__init__(variational_strategy)

        if mean_module is None:
            mean_module = ConstantMean()

        if covar_module is None:
            covar_module = _build_default_mixed_covar_module(
                ref_x=ref_x,
                cont_dims=cont_dims,
                cat_dims=cat_dims,
                cont_kernel_factory=cont_kernel_factory,
            )

        self.mean_module = mean_module.to(device=ref_device, dtype=ref_dtype)
        self.covar_module = covar_module.to(device=ref_device, dtype=ref_dtype)

        self.cat_dims = list(cat_dims)
        self.cont_dims = list(cont_dims)

        self.train_inputs = (ref_x,)
        self.train_targets = train_targets
        self.train_Yvar = train_Yvar

        self.to(device=ref_device, dtype=ref_dtype)

    def transform_inputs(self, X: Tensor) -> Tensor:
        """外側wrapperで変換済みの内部入力をそのまま返す。"""
        return X

    def forward(self, X: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultivariateNormal(mean_x, covar_x)
