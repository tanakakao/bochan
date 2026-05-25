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

from bochan.models.components.beta import BetaMeanLink, BetaLogLikelihood, align_like, prepare_beta_targets
from bochan.models.regression.non_gaussian.beta import BetaGPModel, BetaMixedGPModel


class SparseOutlierBetaLikelihood(BetaLogLikelihood, RelevancePursuitMixin):
    """
    学習点ごとの sparse mean-link offset を持つ Beta likelihood。

    学習時には latent value f_i に sparse offset δ_i を加えます。

        y_i ~ Beta(mean=mu(f_i + δ_i), concentration=φ)

    予測時には δ_i は使いません。
    """

    def __init__(
        self,
        *,
        dim: int,
        outlier_indices: Optional[list[int]] = None,
        delta_init: float = 0.0,
        expanded_base_indices: Optional[Tensor] = None,
        link: BetaMeanLink = "sigmoid",
        init_concentration: float = 20.0,
        learn_concentration: bool = True,
        eps: float = 1e-6,
        min_concentration: float = 1e-6,
    ) -> None:
        BetaLogLikelihood.__init__(self, link=link, init_concentration=init_concentration, learn_concentration=learn_concentration, eps=eps, min_concentration=min_concentration)
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
        return self.raw_delta

    def set_sparse_parameter(self, value: Parameter) -> None:
        self.raw_delta = Parameter(value.to(self.raw_delta))

    def set_expanded_base_indices(self, expanded_base_indices: Optional[Tensor]) -> None:
        if expanded_base_indices is None:
            expanded_base_indices = torch.empty(0, dtype=torch.long, device=self.raw_delta.device)
        self.expanded_base_indices = expanded_base_indices.to(dtype=torch.long, device=self.raw_delta.device)

    @property
    def dense_delta(self) -> Tensor:
        dense = torch.zeros(self.dim, dtype=self.raw_delta.dtype, device=self.raw_delta.device)
        if len(self.support) > 0:
            idx = torch.tensor(self.support, dtype=torch.long, device=dense.device)
            dense[idx] = self.raw_delta
        return dense

    def _delta_for_function_dist(self, function_dist: MultivariateNormal) -> Optional[Tensor]:
        mean = function_dist.mean
        n = mean.shape[-1]
        if n == self.dim:
            return self.dense_delta.to(device=mean.device, dtype=mean.dtype)
        if self.expanded_base_indices.numel() > 0 and n == self.expanded_base_indices.numel():
            base_idx = self.expanded_base_indices.to(device=mean.device)
            dense = self.dense_delta.to(device=mean.device, dtype=mean.dtype)
            return dense[base_idx]
        return None

    def _shift_train_function_dist(self, function_dist: MultivariateNormal) -> MultivariateNormal:
        delta = self._delta_for_function_dist(function_dist)
        if delta is None:
            return function_dist
        delta = align_like(delta, function_dist.mean)
        return function_dist.__class__(function_dist.mean + delta, function_dist.lazy_covariance_matrix)

    def expected_log_prob(self, observations: Tensor, function_dist: MultivariateNormal, *params: Any, **kwargs: Any) -> Tensor:
        return super().expected_log_prob(observations, self._shift_train_function_dist(function_dist), *params, **kwargs)

    def log_marginal(self, observations: Tensor, function_dist: MultivariateNormal, *params: Any, **kwargs: Any) -> Tensor:
        return super().log_marginal(observations, self._shift_train_function_dist(function_dist), *params, **kwargs)


class OutlierRelevancePursuitBetaGPModel(BetaGPModel):
    """学習点 outlier RRP を持つ Beta GP 回帰モデル。"""

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
        link: BetaMeanLink = "sigmoid",
        init_concentration: float = 20.0,
        learn_concentration: bool = True,
        eps: float = 1e-6,
        min_concentration: float = 1e-6,
        clip_targets: bool = True,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        train_Y_beta = prepare_beta_targets(train_Y, train_X, eps=eps, clip=clip_targets)
        likelihood = SparseOutlierBetaLikelihood(
            dim=train_X.shape[-2],
            outlier_indices=outlier_indices,
            delta_init=delta_init,
            expanded_base_indices=None,
            link=link,
            init_concentration=init_concentration,
            learn_concentration=learn_concentration,
            eps=eps,
            min_concentration=min_concentration,
        )
        self.outlier_indices = outlier_indices
        self.delta_init = float(delta_init)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y_beta,
            likelihood=likelihood,
            input_transform=input_transform,
            mean_module=mean_module,
            covar_module=covar_module,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            link=link,
            init_concentration=init_concentration,
            learn_concentration=learn_concentration,
            eps=eps,
            min_concentration=min_concentration,
            clip_targets=clip_targets,
        )

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> "OutlierRelevancePursuitBetaGPModel":
        if kwargs.get("noise") is not None:
            raise NotImplementedError("noise is not supported for OutlierRelevancePursuitBetaGPModel.")
        if isinstance(X, tuple):
            X = X[0]
        X = torch.as_tensor(X, device=self.train_inputs_raw[0].device, dtype=self.train_inputs_raw[0].dtype)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        Y = prepare_beta_targets(Y, X, eps=self.eps, clip=self.clip_targets)
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
            init_concentration=float(self.likelihood.concentration.detach().cpu()),
            learn_concentration=self.learn_concentration,
            eps=self.eps,
            min_concentration=self.min_concentration,
            clip_targets=self.clip_targets,
        )
        new_model.load_state_dict(copy.deepcopy(self.state_dict()), strict=False)
        new_model.eval()
        return new_model


class OutlierRelevancePursuitBetaMixedGPModel(BetaMixedGPModel):
    """mixed 入力版の Beta outlier RRP モデル。"""

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
        link: BetaMeanLink = "sigmoid",
        init_concentration: float = 20.0,
        learn_concentration: bool = True,
        eps: float = 1e-6,
        min_concentration: float = 1e-6,
        clip_targets: bool = True,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        train_Y_beta = prepare_beta_targets(train_Y, train_X, eps=eps, clip=clip_targets)
        likelihood = SparseOutlierBetaLikelihood(
            dim=train_X.shape[-2],
            outlier_indices=outlier_indices,
            delta_init=delta_init,
            expanded_base_indices=None,
            link=link,
            init_concentration=init_concentration,
            learn_concentration=learn_concentration,
            eps=eps,
            min_concentration=min_concentration,
        )
        self.outlier_indices = outlier_indices
        self.delta_init = float(delta_init)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y_beta,
            cat_dims=cat_dims,
            likelihood=likelihood,
            input_transform=input_transform,
            mean_module=mean_module,
            covar_module=covar_module,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            link=link,
            init_concentration=init_concentration,
            learn_concentration=learn_concentration,
            eps=eps,
            min_concentration=min_concentration,
            clip_targets=clip_targets,
        )


__all__ = ["SparseOutlierBetaLikelihood", "OutlierRelevancePursuitBetaGPModel", "OutlierRelevancePursuitBetaMixedGPModel"]
