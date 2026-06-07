from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
from torch import Tensor

from botorch.models.map_saas import add_saas_prior
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel
from gpytorch.means import Mean

from bochan.models.classification.multiclass import (
    MulticlassClassificationGPModel,
    MulticlassClassificationMixedGPModel,
)


def build_map_saas_multiclass_covar_module(
    train_X: Tensor,
    *,
    num_classes: int,
    ard_num_dims: Optional[int] = None,
    tau: float | None = None,
    log_scale: bool = True,
    nu: float = 2.5,
) -> ScaleKernel:
    """多クラス latent GP 用 MAP-SAAS Matern kernel を作る。"""
    if ard_num_dims is None:
        ard_num_dims = train_X.shape[-1]
    batch_shape = torch.Size([int(num_classes)])
    base_kernel = MaternKernel(
        nu=float(nu),
        ard_num_dims=int(ard_num_dims),
        batch_shape=batch_shape,
    ).to(train_X)
    add_saas_prior(base_kernel=base_kernel, tau=tau, log_scale=bool(log_scale))
    return ScaleKernel(base_kernel=base_kernel, batch_shape=batch_shape).to(train_X)


class SaasMulticlassClassificationGPModel(MulticlassClassificationGPModel):
    """MAP-SAAS kernel を使う多クラス分類 GP。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: Optional[int] = None,
        covar_module: Optional[Kernel] = None,
        tau: float | None = None,
        saas_log_scale: bool = True,
        saas_nu: float = 2.5,
        **kwargs: Any,
    ) -> None:
        if num_classes is None:
            y = train_Y.squeeze(-1) if train_Y.ndim > 1 and train_Y.shape[-1] == 1 else train_Y
            num_classes = int(y.max().item()) + 1
        if covar_module is None:
            covar_module = build_map_saas_multiclass_covar_module(
                train_X=train_X,
                num_classes=num_classes,
                tau=tau,
                log_scale=saas_log_scale,
                nu=saas_nu,
            )
        self.tau = tau
        self.saas_log_scale = bool(saas_log_scale)
        self.saas_nu = float(saas_nu)
        super().__init__(train_X=train_X, train_Y=train_Y, num_classes=num_classes, covar_module=covar_module, **kwargs)


class SaasMulticlassClassificationMixedGPModel(MulticlassClassificationMixedGPModel):
    """MAP-SAAS kernel を使う mixed 多クラス分類 GP。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: Sequence[int],
        num_classes: Optional[int] = None,
        covar_module: Optional[Kernel] = None,
        tau: float | None = None,
        saas_log_scale: bool = True,
        saas_nu: float = 2.5,
        **kwargs: Any,
    ) -> None:
        if num_classes is None:
            y = train_Y.squeeze(-1) if train_Y.ndim > 1 and train_Y.shape[-1] == 1 else train_Y
            num_classes = int(y.max().item()) + 1
        if covar_module is None:
            covar_module = build_map_saas_multiclass_covar_module(
                train_X=train_X,
                num_classes=num_classes,
                ard_num_dims=train_X.shape[-1],
                tau=tau,
                log_scale=saas_log_scale,
                nu=saas_nu,
            )
        self.tau = tau
        self.saas_log_scale = bool(saas_log_scale)
        self.saas_nu = float(saas_nu)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=cat_dims,
            num_classes=num_classes,
            covar_module=covar_module,
            **kwargs,
        )


__all__ = [
    'SaasMulticlassClassificationGPModel',
    'SaasMulticlassClassificationMixedGPModel',
    'build_map_saas_multiclass_covar_module',
]
