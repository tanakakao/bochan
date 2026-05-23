from __future__ import annotations

import torch
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from .single_output import _DeepPosteriorAcquisitionBase


class _HeteroRegressionActiveLearningBase(_DeepPosteriorAcquisitionBase):
    """Noise-aware regression active-learning base.

    The class uses model.posterior(X, observation_noise=True) when available and
    falls back to the latent posterior otherwise. This keeps the API parallel to
    the classification / ordinal hetero files while avoiding assumptions about a
    particular heteroscedastic wrapper implementation.
    """

    def _posterior_mean_std_total_latent_noise(
        self,
        X: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        try:
            post_latent = self.model.posterior(X, observation_noise=False)
            post_total = self.model.posterior(X, observation_noise=True)
        except Exception:
            post_latent = self.model.posterior(X)
            post_total = post_latent

        mean = post_latent.mean
        latent_var = post_latent.variance.clamp_min(self.eps)
        total_var = post_total.variance.clamp_min(self.eps)

        mean = self._reduce_outputs_if_needed(mean)
        latent_var = self._reduce_outputs_if_needed(latent_var)
        total_var = self._reduce_outputs_if_needed(total_var)

        if mean.ndim == X.ndim:
            mean = mean.squeeze(-1)
        if latent_var.ndim == X.ndim:
            latent_var = latent_var.squeeze(-1)
        if total_var.ndim == X.ndim:
            total_var = total_var.squeeze(-1)

        noise_var = (total_var - latent_var).clamp_min(self.eps)
        return mean, total_var.sqrt(), latent_var.sqrt(), noise_var.sqrt()


class qHeteroRegressionPredictiveEntropy(_HeteroRegressionActiveLearningBase):
    """heteroscedastic regression 用 predictive entropy acquisition。予測分布の曖昧さが大きい点を選びます。
    
    Args:
        noise_penalty: heteroscedastic noise を避けるための penalty 係数。大きいほど noise の大きい点を避けます。
        *args: 追加 positional arguments。通常は明示的に指定しません。
        **kwargs: 親クラスまたは BoTorch acquisition に渡す追加 keyword arguments。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        予測が曖昧な点を探索したい場合の基本的な active learning acquisition です。
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """

    def __init__(self, *args, noise_penalty: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.noise_penalty = float(noise_penalty)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        _, total_std, _, noise_std = self._posterior_mean_std_total_latent_noise(X)
        entropy = 0.5 * torch.log(
            2.0
            * torch.pi
            * torch.exp(torch.ones((), device=X.device, dtype=X.dtype))
            * total_std.pow(2).clamp_min(self.eps)
        )
        score_per_point = entropy - self.noise_penalty * noise_std
        score = self._aggregate_point_scores(
            score_per_point=score_per_point,
            X=X,
            context="qHeteroRegressionPredictiveEntropy",
        )
        return score - self._total_penalty(X)


class qHeteroRegressionBALDProxy(_HeteroRegressionActiveLearningBase):
    """heteroscedastic regression 用 BALD / mutual-information acquisition。モデル不確実性を減らす情報量の大きい点を選びます。
    
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
        _, total_std, _, noise_std = self._posterior_mean_std_total_latent_noise(X)
        score_per_point = 0.5 * torch.log(
            total_std.pow(2).clamp_min(self.eps) / noise_std.pow(2).clamp_min(self.eps)
        )
        score = self._aggregate_point_scores(
            score_per_point=score_per_point,
            X=X,
            context="qHeteroRegressionBALDProxy",
        )
        return score - self._total_penalty(X)


class qHeteroRegressionPosteriorVariance(_HeteroRegressionActiveLearningBase):
    """heteroscedastic regression 用 variance-based acquisition。posterior / probability / utility の分散が大きい点を選びます。
    
    Args:
        noise_penalty: heteroscedastic noise を避けるための penalty 係数。大きいほど noise の大きい点を避けます。
        *args: 追加 positional arguments。通常は明示的に指定しません。
        **kwargs: 親クラスまたは BoTorch acquisition に渡す追加 keyword arguments。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """

    def __init__(self, *args, noise_penalty: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.noise_penalty = float(noise_penalty)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        _, total_std, _, noise_std = self._posterior_mean_std_total_latent_noise(X)
        score_per_point = total_std.pow(2) - self.noise_penalty * noise_std.pow(2)
        score = self._aggregate_point_scores(
            score_per_point=score_per_point,
            X=X,
            context="qHeteroRegressionPosteriorVariance",
        )
        return score - self._total_penalty(X)


class qHeteroRegressionMarginUncertainty(_HeteroRegressionActiveLearningBase):
    """heteroscedastic regression 用 margin uncertainty acquisition。決定境界または class 境界に近い点を選びます。
    
    Args:
        target: straddle / margin 系で近づけたい目標値。
        beta: 不確実性または sample deviation をどれだけ重視するかを決める係数。
        noise_penalty: heteroscedastic noise を避けるための penalty 係数。大きいほど noise の大きい点を避けます。
        *args: 追加 positional arguments。通常は明示的に指定しません。
        **kwargs: 親クラスまたは BoTorch acquisition に渡す追加 keyword arguments。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """

    def __init__(
        self,
        *args,
        target: float = 0.0,
        beta: float = 1.96,
        noise_penalty: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.target = float(target)
        self.beta = float(beta)
        self.noise_penalty = float(noise_penalty)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        mean, total_std, _, noise_std = self._posterior_mean_std_total_latent_noise(X)
        score_per_point = (
            self.beta * total_std
            - (mean - self.target).abs()
            - self.noise_penalty * noise_std
        )
        score = self._aggregate_point_scores(
            score_per_point=score_per_point,
            X=X,
            context="qHeteroRegressionMarginUncertainty",
        )
        return score - self._total_penalty(X)


class qHeteroRegressionIntegratedPosteriorVarianceProxy(qHeteroRegressionPosteriorVariance):
    """heteroscedastic regression 用 variance-based acquisition。posterior / probability / utility の分散が大きい点を選びます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """

__all__ = [
    "qHeteroRegressionPredictiveEntropy",
    "qHeteroRegressionBALDProxy",
    "qHeteroRegressionPosteriorVariance",
    "qHeteroRegressionMarginUncertainty",
    "qHeteroRegressionIntegratedPosteriorVarianceProxy",
]
