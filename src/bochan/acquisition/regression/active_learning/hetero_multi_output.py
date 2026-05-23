
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from botorch.utils.transforms import t_batch_mode_transform

from .multi_output import (
    _MultiOutputRegressionActiveLearningBase,
    qMultiOutputRegressionIntegratedPosteriorVarianceProxy,
)




class _HeteroMultiOutputRegressionBase(_MultiOutputRegressionActiveLearningBase):
    """Noise-aware multi-output regression active-learning base.

    A generic heteroscedastic noise penalty is estimated from
    ``posterior(..., observation_noise=True).variance - posterior(..., observation_noise=False).variance``.
    This keeps the hetero regression family aligned with classification / ordinal hetero variants.
    """

    def __init__(
        self,
        model,
        *,
        noise_penalty_lambda: float = 1.0,
        noise_mode: str = "inverse_linear",
        noise_combine: str = "multiply",
        noise_min_weight: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.noise_penalty_lambda = float(noise_penalty_lambda)
        self.noise_mode = noise_mode
        self.noise_combine = noise_combine
        self.noise_min_weight = float(noise_min_weight)

    def _noise_variance(self, X: Tensor) -> Tensor:
        _, latent_var = self._posterior_mean_var(X, observation_noise=False)
        _, total_var = self._posterior_mean_var(X, observation_noise=True)
        return (total_var - latent_var).clamp_min(self.eps)

    def _noise_weight(self, X: Tensor) -> Tensor:
        noise = self._noise_variance(X)
        lam = self.noise_penalty_lambda
        if self.noise_mode == "none":
            w = torch.ones_like(noise)
        elif self.noise_mode == "inverse_linear":
            w = 1.0 / (1.0 + lam * noise)
        elif self.noise_mode == "inverse_sqrt":
            w = 1.0 / torch.sqrt(1.0 + lam * noise)
        elif self.noise_mode == "exp":
            w = torch.exp(-lam * noise)
        else:
            raise ValueError(f"Unknown noise_mode={self.noise_mode!r}.")
        if self.noise_min_weight > 0:
            w = self.noise_min_weight + (1.0 - self.noise_min_weight) * w
        return w

    def _apply_noise_weight(self, score_per_point: Tensor, X: Tensor) -> Tensor:
        weight = self._noise_weight(X)
        if weight.shape != score_per_point.shape:
            if weight.numel() == score_per_point.numel():
                weight = weight.reshape_as(score_per_point)
            else:
                # if output aggregation produced q score, aggregate weight the same way
                weight = self._aggregate_outputs(weight)
        if self.noise_combine == "multiply":
            return score_per_point * weight
        if self.noise_combine == "add":
            return score_per_point - (1.0 - weight)
        raise ValueError(f"Unknown noise_combine={self.noise_combine!r}.")


class qHeteroMultiOutputRegressionPredictiveEntropy(_HeteroMultiOutputRegressionBase):
    """heteroscedastic multi-output regression 用 predictive entropy acquisition。予測分布の曖昧さが大きい点を選びます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        予測が曖昧な点を探索したい場合の基本的な active learning acquisition です。
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        X = self._ensure_q_batch(X)
        _, var = self._posterior_mean_var(X, observation_noise=True)
        score = 0.5 * torch.log(2.0 * torch.pi * torch.e * var.clamp_min(self.eps))
        score = self._apply_noise_weight(score, X)
        score = self._aggregate_point_scores(score, X, name="qHeteroMultiOutputRegressionPredictiveEntropy")
        return score - self._total_penalty(X)


class qHeteroMultiOutputRegressionBALDProxy(_HeteroMultiOutputRegressionBase):
    """heteroscedastic multi-output regression 用 BALD / mutual-information acquisition。モデル不確実性を減らす情報量の大きい点を選びます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        BALD は predictive entropy から条件付き entropy を引いた情報利得として解釈できます。
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        X = self._ensure_q_batch(X)
        _, latent_var = self._posterior_mean_var(X, observation_noise=False)
        _, total_var = self._posterior_mean_var(X, observation_noise=True)
        noise_var = (total_var - latent_var).clamp_min(self.eps)
        score = 0.5 * torch.log(total_var.clamp_min(self.eps) / noise_var)
        score = self._apply_noise_weight(score, X)
        score = self._aggregate_point_scores(score, X, name="qHeteroMultiOutputRegressionBALDProxy")
        return score - self._total_penalty(X)


class qHeteroMultiOutputRegressionPosteriorVariance(_HeteroMultiOutputRegressionBase):
    """heteroscedastic multi-output regression 用 variance-based acquisition。posterior / probability / utility の分散が大きい点を選びます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        X = self._ensure_q_batch(X)
        _, var = self._posterior_mean_var(X, observation_noise=False)
        score = self._apply_noise_weight(var, X)
        score = self._aggregate_point_scores(score, X, name="qHeteroMultiOutputRegressionPosteriorVariance")
        return score - self._total_penalty(X)


class qHeteroMultiOutputRegressionMarginUncertainty(_HeteroMultiOutputRegressionBase):
    """heteroscedastic multi-output regression 用 margin uncertainty acquisition。決定境界または class 境界に近い点を選びます。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        target: straddle / margin 系で近づけたい目標値。
        beta: 不確実性または sample deviation をどれだけ重視するかを決める係数。
        **kwargs: 親クラスまたは BoTorch acquisition に渡す追加 keyword arguments。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """
    def __init__(self, model, target: float | Tensor = 0.0, beta: float = 1.96, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        if torch.is_tensor(target):
            self.register_buffer("target", target.detach().clone())
        else:
            self.target = torch.tensor(float(target))
        self.beta = float(beta)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        X = self._ensure_q_batch(X)
        post = self.model.posterior(X, observation_noise=False)
        mean = post.mean
        std = post.variance.clamp_min(self.eps).sqrt()
        if mean.ndim == X.ndim and mean.shape[-1] == 1:
            raw = self.beta * std.squeeze(-1) - (mean.squeeze(-1) - self.target.to(mean)).abs()
        else:
            target = self.target.to(mean)
            if target.ndim == 0:
                raw = self.beta * std - (mean - target).abs()
            else:
                raw = self.beta * std - (mean - target.view(*([1] * (mean.ndim - 1)), -1)).abs()
            raw = self._aggregate_outputs(raw)
        score = self._apply_noise_weight(raw, X)
        score = self._aggregate_point_scores(score, X, name="qHeteroMultiOutputRegressionMarginUncertainty")
        return score - self._total_penalty(X)


class qHeteroMultiOutputRegressionIntegratedPosteriorVarianceProxy(
    qMultiOutputRegressionIntegratedPosteriorVarianceProxy,
    _HeteroMultiOutputRegressionBase,
):
    """heteroscedastic multi-output regression 用 variance-based acquisition。posterior / probability / utility の分散が大きい点を選びます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """

    pass

__all__ = [
    "qHeteroMultiOutputRegressionPredictiveEntropy",
    "qHeteroMultiOutputRegressionBALDProxy",
    "qHeteroMultiOutputRegressionPosteriorVariance",
    "qHeteroMultiOutputRegressionMarginUncertainty",
    "qHeteroMultiOutputRegressionIntegratedPosteriorVarianceProxy",
]
