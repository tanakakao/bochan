from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
from torch import Tensor

from botorch.models.map_saas import add_saas_prior
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel
from gpytorch.means import Mean

from bochan.models.components.poisson import PoissonLink
from bochan.models.regression.non_gaussian.poisson import (
    PoissonGPModel,
    PoissonLogLikelihood,
    PoissonMixedGPModel,
)


def build_map_saas_poisson_covar_module(
    train_X: Tensor,
    *,
    ard_num_dims: Optional[int] = None,
    tau: float | None = None,
    log_scale: bool = True,
    nu: float = 2.5,
) -> ScaleKernel:
    """Poisson latent GP 用の MAP-SAAS Matern kernel を作る。"""
    if ard_num_dims is None:
        ard_num_dims = train_X.shape[-1]
    base_kernel = MaternKernel(
        nu=float(nu),
        ard_num_dims=int(ard_num_dims),
        batch_shape=torch.Size([]),
    ).to(train_X)
    add_saas_prior(base_kernel=base_kernel, tau=tau, log_scale=bool(log_scale))
    return ScaleKernel(base_kernel=base_kernel, batch_shape=torch.Size([])).to(train_X)


class SaasPoissonGPModel(PoissonGPModel):
    """MAP-SAAS kernel を使う Poisson GP 回帰モデル。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        likelihood: Optional[PoissonLogLikelihood] = None,
        input_transform: Any | None = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        num_inducing_points: int = 128,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        tau: float | None = None,
        saas_log_scale: bool = True,
        saas_nu: float = 2.5,
        link: PoissonLink = "softplus",
        exp_clip: float = 20.0,
        min_rate: float = 1e-8,
    ) -> None:
        self.tau = tau
        self.saas_log_scale = bool(saas_log_scale)
        self.saas_nu = float(saas_nu)
        if covar_module is None:
            covar_module = build_map_saas_poisson_covar_module(
                train_X=train_X,
                tau=tau,
                log_scale=saas_log_scale,
                nu=saas_nu,
            )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            input_transform=input_transform,
            mean_module=mean_module,
            covar_module=covar_module,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            link=link,
            exp_clip=exp_clip,
            min_rate=min_rate,
        )


class SaasPoissonMixedGPModel(PoissonMixedGPModel):
    """MAP-SAAS kernel を使う mixed Poisson GP 回帰モデル。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: Sequence[int],
        likelihood: Optional[PoissonLogLikelihood] = None,
        input_transform: Any | None = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        num_inducing_points: int = 128,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        tau: float | None = None,
        saas_log_scale: bool = True,
        saas_nu: float = 2.5,
        link: PoissonLink = "softplus",
        exp_clip: float = 20.0,
        min_rate: float = 1e-8,
    ) -> None:
        self.tau = tau
        self.saas_log_scale = bool(saas_log_scale)
        self.saas_nu = float(saas_nu)
        if covar_module is None:
            covar_module = build_map_saas_poisson_covar_module(
                train_X=train_X,
                ard_num_dims=train_X.shape[-1],
                tau=tau,
                log_scale=saas_log_scale,
                nu=saas_nu,
            )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=cat_dims,
            likelihood=likelihood,
            input_transform=input_transform,
            mean_module=mean_module,
            covar_module=covar_module,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            link=link,
            exp_clip=exp_clip,
            min_rate=min_rate,
        )


__all__ = [
    "build_map_saas_poisson_covar_module",
    "SaasPoissonGPModel",
    "SaasPoissonMixedGPModel",
]
