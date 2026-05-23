from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
from torch import Tensor
from gpytorch.models import ApproximateGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
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
    """
    連続変数用 kernel を作る。
    """
    if len(cont_dims) == 0:
        return None

    if cont_kernel_factory is not None:
        k = cont_kernel_factory(torch.Size([]), len(cont_dims), list(cont_dims))
        return k.to(device=ref_x.device, dtype=ref_x.dtype)

    k = RBFKernel(
        ard_num_dims=len(cont_dims),
        active_dims=tuple(cont_dims),
        batch_shape=torch.Size([]),
    )
    return k.to(device=ref_x.device, dtype=ref_x.dtype)

def _build_default_cat_kernel(
    *,
    cat_dims: Sequence[int],
    ref_x: Tensor,
) -> Optional[Kernel]:
    """
    カテゴリ変数用 kernel を作る。
    """
    if len(cat_dims) == 0:
        return None

    batch_shape = ref_x.shape[:-2]  # batched input にも対応
    k = CategoricalKernel(
        batch_shape=batch_shape,
        ard_num_dims=len(cat_dims),
        active_dims=tuple(cat_dims),
        lengthscale_constraint=GreaterThan(1e-6),
    )
    return k.to(device=ref_x.device, dtype=ref_x.dtype)

def _build_default_mixed_covar_module(
    *,
    ref_x: Tensor,
    cont_dims: Sequence[int],
    cat_dims: Sequence[int],
    cont_kernel_factory: Optional[
        Callable[[torch.Size, int, Optional[list[int]]], Kernel]
    ] = None,
) -> Kernel:
    """
    mixed input 用のデフォルト covar_module を作る。

    方針:
        - continuous only: cont
        - categorical only: cat
        - mixed: cont + cat + cont * cat
    """
    cont_kernel = _build_default_cont_kernel(
        cont_dims=cont_dims,
        ref_x=ref_x,
        cont_kernel_factory=cont_kernel_factory,
    )
    cat_kernel = _build_default_cat_kernel(
        cat_dims=cat_dims,
        ref_x=ref_x,
    )

    if cont_kernel is None and cat_kernel is None:
        raise ValueError("At least one kernel component must be available.")

    if cont_kernel is None:
        base_kernel = cat_kernel
    elif cat_kernel is None:
        base_kernel = cont_kernel
    else:
        base_kernel = cont_kernel + cat_kernel + cont_kernel * cat_kernel

    covar_module = ScaleKernel(base_kernel, batch_shape=torch.Size([]))
    return covar_module.to(device=ref_x.device, dtype=ref_x.dtype)


class _LatentBinarySVGP(ApproximateGP):
    """
    2値分類用の latent SVGP。

    重要:
        train_inputs を基準に、variational distribution / mean / kernel /
        すべての parameter / buffer の dtype / device を統一する。
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
                RBFKernel(
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

        # 念のため全体を再統一
        self.to(device=ref_device, dtype=ref_dtype)

    def forward(self, X: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultivariateNormal(mean_x, covar_x)


class _LatentMixedBinarySVGP(ApproximateGP):
    """
    mixed input（連続 + カテゴリ）対応の latent SVGP。

    デフォルト kernel:
        cont + cat + cont * cat
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
        ord_dims = sorted(set(range(d)) - set(cat_dims))

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

        # 念のため全体を再統一
        self.to(device=ref_device, dtype=ref_dtype)

    def forward(self, X: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultivariateNormal(mean_x, covar_x)