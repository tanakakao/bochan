from __future__ import annotations

import copy
from typing import Any, Optional, Sequence

import torch
from torch import Tensor
from torch.nn import Parameter

from botorch.models.transforms.input import InputTransform
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import SoftmaxLikelihood
from gpytorch.means import Mean

from bochan.models.components.multiclass import (
    move_class_dim_to_last,
    prepare_class_targets,
)
from bochan.models.classification.multiclass import (
    MulticlassClassificationGPModel,
    MulticlassClassificationMixedGPModel,
)


class SparseOutlierSoftmaxLikelihood(SoftmaxLikelihood):
    """
    学習点ごとの sparse softmax logit offset を持つ likelihood。

    Notes:
        `outlier_indices` で指定した学習点だけに `delta_i[k]` を足します。
        予測時には学習点に対応しないので offset は使われません。
    """

    def __init__(
        self,
        *,
        num_features: int,
        num_classes: int,
        dim: int,
        outlier_indices: Optional[list[int]] = None,
        delta_init: float = 0.0,
        mixing_weights: bool = False,
    ) -> None:
        super().__init__(
            num_features=num_features,
            num_classes=num_classes,
            mixing_weights=mixing_weights,
        )
        self.dim = int(dim)
        self.num_classes = int(num_classes)
        self.support = list(outlier_indices or [])
        init = torch.full(
            (len(self.support), self.num_classes),
            float(delta_init),
            dtype=torch.get_default_dtype(),
        )
        self.register_parameter("raw_delta", Parameter(init))
        self.delta_init = float(delta_init)

    @property
    def dense_delta(self) -> Tensor:
        """shape [n, C] の dense offset を返す。"""
        dense = torch.zeros(
            self.dim,
            self.num_classes,
            dtype=self.raw_delta.dtype,
            device=self.raw_delta.device,
        )
        if len(self.support) > 0:
            idx = torch.tensor(self.support, dtype=torch.long, device=dense.device)
            dense[idx] = self.raw_delta
        return dense

    def _shift_function_dist(self, function_dist: MultivariateNormal) -> MultivariateNormal:
        mean = function_dist.mean
        dense = self.dense_delta.to(device=mean.device, dtype=mean.dtype)

        # 学習点数と合わない場合は予測時とみなし、offset しない。
        if mean.shape[-1] == self.dim and mean.shape[0] == self.num_classes:
            # mean: [C, n]
            delta = dense.transpose(-1, -2)
        elif mean.shape[-2:] == dense.shape:
            # mean: [n, C]
            delta = dense
        else:
            return function_dist

        shifted_mean = mean + delta
        return function_dist.__class__(shifted_mean, function_dist.lazy_covariance_matrix)

    def expected_log_prob(
        self,
        observations: Tensor,
        function_dist: MultivariateNormal,
        *params: Any,
        **kwargs: Any,
    ) -> Tensor:
        function_dist = self._shift_function_dist(function_dist)
        return super().expected_log_prob(observations, function_dist, *params, **kwargs)

    def log_marginal(
        self,
        observations: Tensor,
        function_dist: MultivariateNormal,
        *params: Any,
        **kwargs: Any,
    ) -> Tensor:
        function_dist = self._shift_function_dist(function_dist)
        return super().log_marginal(observations, function_dist, *params, **kwargs)


class OutlierRelevancePursuitMulticlassClassificationGPModel(MulticlassClassificationGPModel):
    """学習点 outlier RRP を持つ多クラス分類 GP。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: Optional[int] = None,
        input_transform: Optional[InputTransform] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        num_inducing_points: int = 128,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        outlier_indices: Optional[list[int]] = None,
        delta_init: float = 0.0,
        temperature: float = 1.0,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        if num_classes is None:
            y_tmp = train_Y.squeeze(-1) if train_Y.ndim > 1 and train_Y.shape[-1] == 1 else train_Y
            num_classes = int(torch.as_tensor(y_tmp).max().item()) + 1
        train_Y = prepare_class_targets(train_Y, train_X, num_classes=num_classes)
        likelihood = SparseOutlierSoftmaxLikelihood(
            num_features=num_classes,
            num_classes=num_classes,
            dim=train_X.shape[-2],
            outlier_indices=outlier_indices,
            delta_init=delta_init,
            mixing_weights=False,
        )
        self.outlier_indices = outlier_indices
        self.delta_init = float(delta_init)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            num_classes=num_classes,
            likelihood=likelihood,
            input_transform=input_transform,
            mean_module=mean_module,
            covar_module=covar_module,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            temperature=temperature,
        )


class OutlierRelevancePursuitMulticlassClassificationMixedGPModel(MulticlassClassificationMixedGPModel):
    """mixed 入力版の多クラス outlier RRP 分類 GP。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: Sequence[int],
        num_classes: Optional[int] = None,
        input_transform: Optional[InputTransform] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        num_inducing_points: int = 128,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        outlier_indices: Optional[list[int]] = None,
        delta_init: float = 0.0,
        temperature: float = 1.0,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        if num_classes is None:
            y_tmp = train_Y.squeeze(-1) if train_Y.ndim > 1 and train_Y.shape[-1] == 1 else train_Y
            num_classes = int(torch.as_tensor(y_tmp).max().item()) + 1
        train_Y = prepare_class_targets(train_Y, train_X, num_classes=num_classes)
        likelihood = SparseOutlierSoftmaxLikelihood(
            num_features=num_classes,
            num_classes=num_classes,
            dim=train_X.shape[-2],
            outlier_indices=outlier_indices,
            delta_init=delta_init,
            mixing_weights=False,
        )
        self.outlier_indices = outlier_indices
        self.delta_init = float(delta_init)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=cat_dims,
            num_classes=num_classes,
            likelihood=likelihood,
            input_transform=input_transform,
            mean_module=mean_module,
            covar_module=covar_module,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            temperature=temperature,
        )


__all__ = [
    "SparseOutlierSoftmaxLikelihood",
    "OutlierRelevancePursuitMulticlassClassificationGPModel",
    "OutlierRelevancePursuitMulticlassClassificationMixedGPModel",
]
