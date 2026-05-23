from __future__ import annotations

import copy
from typing import Any, Optional, Sequence

import torch
from torch import Tensor
from torch.nn import Parameter
from botorch.models.relevance_pursuit import RelevancePursuitMixin
from botorch.models.transforms.input import InputTransform
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.means import Mean

from bochan.models.components.poisson import (
    PoissonLink,
    PoissonLogLikelihood,
    align_like,
    prepare_count_targets,
)
from bochan.models.regression.non_gaussian.poisson import (
    PoissonGPModel,
    PoissonMixedGPModel,
)


class SparseOutlierPoissonLikelihood(PoissonLogLikelihood, RelevancePursuitMixin):
    """学習点ごとの sparse log-rate offset を持つ Poisson likelihood。

    学習時だけ latent value に offset `delta_i` を加えます。

        y_i ~ Poisson(rate(f_i + delta_i))

    予測時は新規点に対応する offset がないため、通常の Poisson likelihood として動作します。

    Args:
        dim: 元の学習点数。
        outlier_indices: 初期 support として扱う学習点 index。
        delta_init: sparse offset の初期値。
        expanded_base_indices: InputPerturbation 等で学習点が展開された場合の base index。
        link: Poisson rate link。`"softplus"` または `"exp"`。
        exp_clip: `link="exp"` のときの latent clipping 上限。
        min_rate: rate の下限。
    """

    def __init__(
        self,
        *,
        dim: int,
        outlier_indices: Optional[list[int]] = None,
        delta_init: float = 0.0,
        expanded_base_indices: Optional[Tensor] = None,
        link: PoissonLink = "softplus",
        exp_clip: float = 20.0,
        min_rate: float = 1e-8,
    ) -> None:
        PoissonLogLikelihood.__init__(self, link=link, exp_clip=exp_clip, min_rate=min_rate)
        RelevancePursuitMixin.__init__(self, dim=int(dim), support=outlier_indices)

        init = torch.full((len(self.support),), float(delta_init), dtype=torch.get_default_dtype())
        self.register_parameter("raw_delta", Parameter(init))
        if expanded_base_indices is None:
            expanded_base_indices = torch.empty(0, dtype=torch.long)
        self.register_buffer("expanded_base_indices", expanded_base_indices.to(dtype=torch.long))
        self.delta_init = float(delta_init)
        self._expansion_modifier = torch.abs
        self._contraction_modifier = torch.abs

    @property
    def sparse_parameter(self) -> Parameter:
        """RelevancePursuitMixin が扱う sparse parameter。"""
        return self.raw_delta

    def set_sparse_parameter(self, value: Parameter) -> None:
        """sparse parameter を更新する。"""
        self.raw_delta = Parameter(value.to(self.raw_delta))

    def set_expanded_base_indices(self, expanded_base_indices: Optional[Tensor]) -> None:
        """InputPerturbation 展開用の base index を更新する。"""
        if expanded_base_indices is None:
            expanded_base_indices = torch.empty(0, dtype=torch.long, device=self.raw_delta.device)
        self.expanded_base_indices = expanded_base_indices.to(dtype=torch.long, device=self.raw_delta.device)

    @property
    def dense_delta(self) -> Tensor:
        """sparse offset を元の学習点数 `dim` の dense vector に戻す。"""
        dense = torch.zeros(self.dim, dtype=self.raw_delta.dtype, device=self.raw_delta.device)
        if len(self.support) > 0:
            idx = torch.tensor(self.support, dtype=torch.long, device=dense.device)
            dense[idx] = self.raw_delta
        return dense

    def _delta_for_function_dist(self, function_dist: MultivariateNormal) -> Optional[Tensor]:
        """function_dist.mean の event length に合う offset を返す。"""
        mean = function_dist.mean
        n = mean.shape[-1]
        if n == self.dim:
            return self.dense_delta.to(device=mean.device, dtype=mean.dtype)
        if self.expanded_base_indices.numel() > 0 and n == self.expanded_base_indices.numel():
            base_idx = self.expanded_base_indices.to(device=mean.device)
            return self.dense_delta.to(device=mean.device, dtype=mean.dtype)[base_idx]
        return None

    def _shift_train_function_dist(self, function_dist: MultivariateNormal) -> MultivariateNormal:
        """学習点に対応する distribution だけ latent mean を shift する。"""
        delta = self._delta_for_function_dist(function_dist)
        if delta is None:
            return function_dist
        delta = align_like(delta, function_dist.mean)
        return function_dist.__class__(function_dist.mean + delta, function_dist.lazy_covariance_matrix)

    def expected_log_prob(self, observations: Tensor, function_dist: MultivariateNormal, *params: Any, **kwargs: Any) -> Tensor:
        function_dist = self._shift_train_function_dist(function_dist)
        return super().expected_log_prob(observations, function_dist, *params, **kwargs)

    def log_marginal(self, observations: Tensor, function_dist: MultivariateNormal, *params: Any, **kwargs: Any) -> Tensor:
        function_dist = self._shift_train_function_dist(function_dist)
        return super().log_marginal(observations, function_dist, *params, **kwargs)


class OutlierRelevancePursuitPoissonGPModel(PoissonGPModel):
    """学習点 outlier RRP を持つ Poisson GP 回帰モデル。

    Notes:
        Gaussian regression の RRP は feature relevance を扱いますが、
        この Poisson RRP は count 観測の outlier / 過分散的な学習点を sparse に表します。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        input_transform: Optional[InputTransform] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        num_inducing_points: int = 128,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        outlier_indices: Optional[list[int]] = None,
        delta_init: float = 0.0,
        link: PoissonLink = "softplus",
        exp_clip: float = 20.0,
        min_rate: float = 1e-8,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        train_Y_count = prepare_count_targets(train_Y, train_X)
        likelihood = SparseOutlierPoissonLikelihood(
            dim=train_X.shape[-2],
            outlier_indices=outlier_indices,
            delta_init=delta_init,
            link=link,
            exp_clip=exp_clip,
            min_rate=min_rate,
        )
        self.delta_init = float(delta_init)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y_count,
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

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> "OutlierRelevancePursuitPoissonGPModel":
        if kwargs.get("noise") is not None:
            raise NotImplementedError("noise is not supported for OutlierRelevancePursuitPoissonGPModel.")
        if isinstance(X, tuple):
            X = X[0]
        X = torch.as_tensor(X, device=self.train_inputs_raw[0].device, dtype=self.train_inputs_raw[0].dtype)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        Y = prepare_count_targets(Y, X)
        new_X = torch.cat([self.train_inputs_raw[0], X], dim=-2)
        new_Y = torch.cat([self.train_targets, Y], dim=0)
        new_model = self.__class__(
            train_X=new_X,
            train_Y=new_Y,
            input_transform=copy.deepcopy(self.input_transform),
            mean_module=copy.deepcopy(self.model.mean_module),
            covar_module=copy.deepcopy(self.model.covar_module),
            num_inducing_points=self.num_inducing_points,
            inducing_points=self.model.variational_strategy.inducing_points.detach().clone(),
            learn_inducing_locations=self.learn_inducing_locations,
            outlier_indices=list(self.likelihood.support),
            delta_init=self.delta_init,
            link=self.link,
            exp_clip=self.exp_clip,
            min_rate=self.min_rate,
        )
        new_model.load_state_dict(copy.deepcopy(self.state_dict()), strict=False)
        new_model.eval()
        return new_model


class OutlierRelevancePursuitPoissonMixedGPModel(PoissonMixedGPModel):
    """mixed 入力版の Poisson outlier RRP モデル。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: Sequence[int],
        input_transform: Optional[InputTransform] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        num_inducing_points: int = 128,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        outlier_indices: Optional[list[int]] = None,
        delta_init: float = 0.0,
        link: PoissonLink = "softplus",
        exp_clip: float = 20.0,
        min_rate: float = 1e-8,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        train_Y_count = prepare_count_targets(train_Y, train_X)
        likelihood = SparseOutlierPoissonLikelihood(
            dim=train_X.shape[-2],
            outlier_indices=outlier_indices,
            delta_init=delta_init,
            link=link,
            exp_clip=exp_clip,
            min_rate=min_rate,
        )
        self.delta_init = float(delta_init)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y_count,
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

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> "OutlierRelevancePursuitPoissonMixedGPModel":
        if kwargs.get("noise") is not None:
            raise NotImplementedError("noise is not supported for OutlierRelevancePursuitPoissonMixedGPModel.")
        if isinstance(X, tuple):
            X = X[0]
        X = torch.as_tensor(X, device=self.train_inputs_raw[0].device, dtype=self.train_inputs_raw[0].dtype)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        Y = prepare_count_targets(Y, X)
        new_X = torch.cat([self.train_inputs_raw[0], X], dim=-2)
        new_Y = torch.cat([self.train_targets, Y], dim=0)
        new_model = self.__class__(
            train_X=new_X,
            train_Y=new_Y,
            cat_dims=list(self.cat_dims),
            input_transform=copy.deepcopy(self.input_transform),
            mean_module=copy.deepcopy(self.model.mean_module),
            covar_module=copy.deepcopy(self.model.covar_module),
            num_inducing_points=self.num_inducing_points,
            inducing_points=self.model.variational_strategy.inducing_points.detach().clone(),
            learn_inducing_locations=self.learn_inducing_locations,
            outlier_indices=list(self.likelihood.support),
            delta_init=self.delta_init,
            link=self.link,
            exp_clip=self.exp_clip,
            min_rate=self.min_rate,
        )
        new_model.load_state_dict(copy.deepcopy(self.state_dict()), strict=False)
        new_model.eval()
        return new_model


__all__ = [
    "SparseOutlierPoissonLikelihood",
    "OutlierRelevancePursuitPoissonGPModel",
    "OutlierRelevancePursuitPoissonMixedGPModel",
]
