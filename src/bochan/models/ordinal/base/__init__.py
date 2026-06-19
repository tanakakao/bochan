from __future__ import annotations

from typing import Optional

from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel
from torch import Tensor

from .models import OrdinalGPModel as _OrdinalGPModel
from .models import OrdinalMixedGPModel
from .multioutput import MultiOutputOrdinalModel
from .multitask import MultiTaskOrdinalGPModel


class OrdinalGPModel(_OrdinalGPModel):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        num_classes: Optional[int] = None,
        inducing_points_num: int = 128,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        mean_module=None,
        covar_module: Optional[Kernel] = None,
        input_transform=None,
        eps: float = 1e-8,
        init_gap: float = 1.0,
        fix_first_cutpoint: bool = True,
        conditioning_steps: int = 50,
        conditioning_lr: Optional[float] = None,
        conditioning_batch_size: Optional[int] = None,
    ) -> None:
        if covar_module is None:
            covar_module = ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=train_X.shape[-1])
            ).to(device=train_X.device, dtype=train_X.dtype)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            num_classes=num_classes,
            inducing_points_num=inducing_points_num,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            mean_module=mean_module,
            covar_module=covar_module,
            input_transform=input_transform,
            eps=eps,
            init_gap=init_gap,
            fix_first_cutpoint=fix_first_cutpoint,
            conditioning_steps=conditioning_steps,
            conditioning_lr=conditioning_lr,
            conditioning_batch_size=conditioning_batch_size,
        )


__all__ = [
    "OrdinalGPModel",
    "OrdinalMixedGPModel",
    "MultiOutputOrdinalModel",
    "MultiTaskOrdinalGPModel",
]
