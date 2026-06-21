from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Optional

import torch
from botorch.models.transforms.input import InputTransform
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.means import Mean
from torch import Tensor

from .kronecker_multitask import KroneckerMultiTaskBinaryClassificationGPModel
from .kronecker_multitask_mixed import (
    KroneckerMultiTaskBinaryClassificationMixedGPModel,
)
from .models import (
    BinaryClassificationGPModel as _BinaryClassificationGPModel,
    BinaryClassificationMixedGPModel as _BinaryClassificationMixedGPModel,
)
from .multioutput import MultiOutputBinaryClassificationModel
from .multitask import MultiTaskBinaryClassificationGPModel
from .multitask_mixed import MultiTaskBinaryClassificationMixedGPModel


class BinaryClassificationGPModel(_BinaryClassificationGPModel):
    """Binary SVGP model with multiclass-aligned defaults."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        likelihood: Optional[BernoulliLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        num_inducing_points: int = 128,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            input_transform=input_transform,
            mean_module=mean_module,
            covar_module=covar_module,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
        )


class BinaryClassificationMixedGPModel(_BinaryClassificationMixedGPModel):
    """Mixed-input binary SVGP with multiclass-aligned defaults."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        train_Yvar: Optional[Tensor] = None,
        likelihood: Optional[BernoulliLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        cont_kernel_factory: Optional[
            Callable[[torch.Size, int, Optional[list[int]]], Kernel]
        ] = None,
        num_inducing_points: int = 128,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=cat_dims,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            input_transform=input_transform,
            mean_module=mean_module,
            covar_module=covar_module,
            cont_kernel_factory=cont_kernel_factory,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
        )


__all__ = [
    "BinaryClassificationGPModel",
    "BinaryClassificationMixedGPModel",
    "KroneckerMultiTaskBinaryClassificationGPModel",
    "KroneckerMultiTaskBinaryClassificationMixedGPModel",
    "MultiOutputBinaryClassificationModel",
    "MultiTaskBinaryClassificationGPModel",
    "MultiTaskBinaryClassificationMixedGPModel",
]
