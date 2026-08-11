from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
from torch import Tensor
from torch.nn import Parameter

from botorch.models.relevance_pursuit import RelevancePursuitMixin
from botorch.models.transforms.input import InputTransform
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import SoftmaxLikelihood
from gpytorch.means import Mean

from bochan.models.components.multiclass import prepare_class_targets
from bochan.models.classification.multiclass import (
    MulticlassClassificationGPModel,
    MulticlassClassificationMixedGPModel,
)


def _multiclass_delta_norm(value: Tensor) -> Tensor:
    """RRP の support 選択用に class-wise offset を学習点ごとの大きさへ集約する。"""
    if value.ndim <= 1:
        return value.abs()
    return torch.linalg.vector_norm(value, ord=2, dim=-1)


class SparseOutlierSoftmaxLikelihood(SoftmaxLikelihood, RelevancePursuitMixin):
    """
    学習点ごとの sparse softmax logit offset を持つ likelihood。

    Notes:
        `outlier_indices` で指定した学習点だけに `delta_i[k]` を足します。
        予測時には学習点に対応しないので offset は使われません。

        BoTorch の RelevancePursuitMixin は 1D sparse parameter を前提にしている。
        multiclass では sparse parameter が shape ``[num_support, num_classes]`` に
        なるため、support 操作をこの likelihood 内で 2D 用に override している。
    """

    def __init__(
        self,
        *,
        num_features: int,
        num_classes: int,
        dim: int,
        outlier_indices: Optional[list[int]] = None,
        delta_init: float = 0.0,
        expanded_base_indices: Optional[Tensor] = None,
        mixing_weights: bool = False,
    ) -> None:
        SoftmaxLikelihood.__init__(
            self,
            num_features=num_features,
            num_classes=num_classes,
            mixing_weights=mixing_weights,
        )
        RelevancePursuitMixin.__init__(self, dim=int(dim), support=outlier_indices)
        self.dim = int(dim)
        self.num_classes = int(num_classes)

        init = torch.full(
            (len(self.support), self.num_classes),
            float(delta_init),
            dtype=torch.get_default_dtype(),
        )
        self.register_parameter("raw_delta", Parameter(init))
        if expanded_base_indices is None:
            expanded_base_indices = torch.empty(0, dtype=torch.long)
        self.register_buffer("expanded_base_indices", expanded_base_indices.to(dtype=torch.long))
        self.delta_init = float(delta_init)
        self._expansion_modifier = _multiclass_delta_norm
        self._contraction_modifier = _multiclass_delta_norm

    @property
    def sparse_parameter(self) -> Parameter:
        """RelevancePursuitMixin が扱う sparse softmax offset。"""
        return self.raw_delta

    def set_sparse_parameter(self, value: Parameter) -> None:
        """sparse parameter を更新する。"""
        value = value.to(self.raw_delta)
        if value.ndim == 1:
            if value.numel() == 0:
                value = value.reshape(0, self.num_classes)
            elif self.num_classes > 1 and value.numel() == self.num_classes:
                value = value.unsqueeze(0)
            else:
                raise ValueError(
                    "SparseOutlierSoftmaxLikelihood requires a 2D sparse parameter "
                    f"with shape [support, num_classes]. Got shape={tuple(value.shape)}."
                )
        if value.ndim != 2 or value.shape[-1] != self.num_classes:
            raise ValueError(
                "SparseOutlierSoftmaxLikelihood sparse parameter shape mismatch. "
                f"Expected [..., {self.num_classes}], got {tuple(value.shape)}."
            )
        self.raw_delta = Parameter(value)

    def set_expanded_base_indices(self, expanded_base_indices: Optional[Tensor]) -> None:
        """InputPerturbation 展開用の base index を更新する。"""
        if expanded_base_indices is None:
            expanded_base_indices = torch.empty(0, dtype=torch.long, device=self.raw_delta.device)
        self.expanded_base_indices = expanded_base_indices.to(dtype=torch.long, device=self.raw_delta.device)

    def to_sparse(self):
        """2D sparse parameter を sparse 表現 ``[len(support), C]`` に変換する。"""
        if not self.is_sparse:
            self.set_sparse_parameter(Parameter(self.sparse_parameter[self.support]))
            self._is_sparse = True
        return self

    def to_dense(self):
        """2D sparse parameter を dense 表現 ``[dim, C]`` に変換する。"""
        if self.is_sparse:
            dense_parameter = torch.zeros(
                self.dim,
                self.num_classes,
                dtype=self.sparse_parameter.dtype,
                device=self.sparse_parameter.device,
            )
            if len(self.support) > 0:
                idx = torch.tensor(self.support, dtype=torch.long, device=dense_parameter.device)
                dense_parameter[idx] = self.sparse_parameter
            self.set_sparse_parameter(Parameter(dense_parameter))
            self._is_sparse = False
        return self

    def expand_support(self, indices: list[int]):
        """support を拡張し、追加点に class-wise zero offset を追加する。"""
        indices = [int(i) for i in indices]
        for i in indices:
            if i in self.support:
                raise ValueError(f"Feature {i} already in the support.")

        self.support.extend(indices)
        if self.is_sparse and len(indices) > 0:
            zeros = torch.zeros(
                len(indices),
                self.num_classes,
                dtype=self.sparse_parameter.dtype,
                device=self.sparse_parameter.device,
            )
            self.set_sparse_parameter(Parameter(torch.cat((self.sparse_parameter, zeros), dim=0)))
        return self

    def contract_support(self, indices: list[int]):
        """support を縮小し、2D offset の対応行を削除または 0 化する。"""
        indices = [int(i) for i in indices]
        sparse_indices = list(range(len(self.support)))
        original_support = list(self.support)
        for i in indices:
            if i not in self.support:
                raise ValueError(f"Feature {i} is not in support.")
            sparse_indices.remove(original_support.index(i))
            self.support.remove(i)

        if self.is_sparse:
            self.set_sparse_parameter(Parameter(self.sparse_parameter[sparse_indices]))
        else:
            requires_grad = self.sparse_parameter.requires_grad
            self.sparse_parameter.requires_grad_(False)
            if len(indices) > 0:
                self.sparse_parameter[indices, :] = 0.0
            self.sparse_parameter.requires_grad_(requires_grad)
        return self

    def remove_support(self):
        """support を空にする。sparse 表現では ``[0, C]`` の parameter を保持する。"""
        self._support = []
        requires_grad = self.sparse_parameter.requires_grad
        if self.is_sparse:
            empty = torch.empty(
                0,
                self.num_classes,
                dtype=self.sparse_parameter.dtype,
                device=self.sparse_parameter.device,
            )
            self.set_sparse_parameter(Parameter(empty))
        else:
            self.sparse_parameter.requires_grad_(False)
            self.sparse_parameter[:] = 0.0
        self.sparse_parameter.requires_grad_(requires_grad)
        return self

    def support_contraction(self, mll, n: int = 1, modifier=None) -> bool:
        """2D sparse parameter 用に contraction の index 変換だけ安全化する。"""
        if len(self.support) == 0 or n <= 0:
            return False

        is_sparse = self.is_sparse
        self.to_sparse()
        x = self.sparse_parameter

        modifier = modifier if modifier is not None else self._contraction_modifier
        if modifier is not None:
            x = modifier(x)

        sparse_indices = [int(i) for i in x.argsort(descending=False)[:n].tolist()]
        indices = [self.support[i] for i in sparse_indices]
        self.contract_support(indices)
        if not is_sparse:
            self.to_dense()
        return True

    @property
    def dense_delta(self) -> Tensor:
        """shape [n, C] の dense offset を返す。"""
        if not self.is_sparse:
            return self.raw_delta

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

    def _dense_delta_for_mean(self, mean: Tensor) -> Optional[Tensor]:
        """function_dist.mean に alignment 可能な dense offset を返す。"""
        n = mean.shape[-1]
        if n == self.dim:
            return self.dense_delta.to(device=mean.device, dtype=mean.dtype)
        if self.expanded_base_indices.numel() > 0 and n == self.expanded_base_indices.numel():
            base_idx = self.expanded_base_indices.to(device=mean.device)
            dense = self.dense_delta.to(device=mean.device, dtype=mean.dtype)
            return dense[base_idx]
        return None

    def _shift_function_dist(self, function_dist: MultivariateNormal) -> MultivariateNormal:
        mean = function_dist.mean
        dense = self._dense_delta_for_mean(mean)
        if dense is None:
            return function_dist

        if mean.shape[-1] == dense.shape[-2] and mean.shape[-2] == self.num_classes:
            # mean: [..., C, n]
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


class RobustRelevancePursuitMulticlassClassificationGPModel(MulticlassClassificationGPModel):
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
        num_inducing: int = 128,
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
            num_inducing=num_inducing,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            temperature=temperature,
        )


class RobustRelevancePursuitMulticlassClassificationMixedGPModel(MulticlassClassificationMixedGPModel):
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
        num_inducing: int = 128,
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
            num_inducing=num_inducing,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            temperature=temperature,
        )


__all__ = [
    "SparseOutlierSoftmaxLikelihood",
    "RobustRelevancePursuitMulticlassClassificationGPModel",
    "RobustRelevancePursuitMulticlassClassificationMixedGPModel",
]
