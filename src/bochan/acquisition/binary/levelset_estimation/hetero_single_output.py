from __future__ import annotations

from typing import Callable, Optional

import torch
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition.binary.base import (
    NoiseCombineType,
    NoiseWeightMode,
    ROICombineType,
    ROIWeightMode,
    ReductionType,
    _BinaryClassificationAcqBase,
)
from ._utils import (
    align_pointwise_score_to_X,
    apply_classification_objective_to_score,
    bernoulli_entropy,
    boundary_kernel_weight,
    normalize_pointwise_tensor_to_orig,
)


class _BaseHeteroBinaryLevelSetAcquisition(_BinaryClassificationAcqBase):
    """heteroscedastic binary classification level-set acquisition の共通基底。"""

    def __init__(
        self,
        model,
        reduction: ReductionType = "mean",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        eps: float = 1e-6,
        # ROI
        roi_mode: ROIWeightMode = "none",
        roi_combine: ROICombineType = "multiply",
        roi_threshold: float = 0.5,
        roi_target_prob: float = 0.8,
        roi_interval: Optional[tuple[float, float]] = None,
        roi_beta: float = 20.0,
        roi_bandwidth: float = 0.15,
        roi_min_weight: float = 0.0,
        roi_weight_scale: float = 1.0,
        roi_weight_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        # noise
        noise_mode: NoiseWeightMode = "inverse_linear",
        noise_combine: NoiseCombineType = "multiply",
        noise_penalty_lambda: float = 1.0,
        noise_min_weight: float = 0.0,
        noise_weight_scale: float = 1.0,
        noise_model_outputs_log_var: bool = True,
        noise_weight_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        # objective
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ):
        super().__init__(
            model=model,
            reduction=reduction,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            eps=eps,
            roi_mode=roi_mode,
            roi_combine=roi_combine,
            roi_threshold=roi_threshold,
            roi_target_prob=roi_target_prob,
            roi_interval=roi_interval,
            roi_beta=roi_beta,
            roi_bandwidth=roi_bandwidth,
            roi_min_weight=roi_min_weight,
            roi_weight_scale=roi_weight_scale,
            roi_weight_fn=roi_weight_fn,
            noise_mode=noise_mode,
            noise_combine=noise_combine,
            noise_penalty_lambda=noise_penalty_lambda,
            noise_min_weight=noise_min_weight,
            noise_weight_scale=noise_weight_scale,
            noise_model_outputs_log_var=noise_model_outputs_log_var,
            noise_weight_fn=noise_weight_fn,
            objective=objective,
        )
        self.objective = objective

    def _get_latent_posterior(self, X: Tensor):
        for name in ("latent_posterior", "posterior_latent", "posterior_f"):
            fn = getattr(self.model, name, None)
            if callable(fn):
                return fn(X)

        inner_model = getattr(self.model, "model", None)
        if inner_model is not None and callable(getattr(inner_model, "posterior", None)):
            return inner_model.posterior(X)

        gp_model = getattr(self.model, "gp_model", None)
        if gp_model is not None and callable(getattr(gp_model, "posterior", None)):
            return gp_model.posterior(X)

        raise AttributeError(
            "Latent posterior accessor was not found. Expected one of:\n"
            "  - model.latent_posterior(X)\n"
            "  - model.posterior_latent(X)\n"
            "  - model.posterior_f(X)\n"
            "  - model.model.posterior(X)\n"
            "  - model.gp_model.posterior(X)"
        )

    def _extract_variance_from_posterior(self, posterior) -> Tensor:
        if hasattr(posterior, "variance"):
            return posterior.variance
        dist = getattr(posterior, "distribution", None)
        if dist is not None and hasattr(dist, "variance"):
            return dist.variance
        mvn = getattr(posterior, "mvn", None)
        if mvn is not None and hasattr(mvn, "variance"):
            return mvn.variance
        raise AttributeError(
            "Could not extract variance from latent posterior. "
            "Expected posterior.variance or posterior.distribution.variance."
        )

    def _latent_stats_and_mean_prob(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        X = self._ensure_q_batch(X)
        Xt = self._apply_input_transform(X)
        orig = Xt.shape[:-1]

        # raw X を渡すことで、モデル側 input_transform に一任する。
        latent_post = self._get_latent_posterior(X)
        mu_f = normalize_pointwise_tensor_to_orig(latent_post.mean, orig, name="latent mean")
        var_f = normalize_pointwise_tensor_to_orig(
            self._extract_variance_from_posterior(latent_post),
            orig,
            name="latent variance",
        ).clamp_min(self.eps)

        post = self.model.posterior(X)
        mean_prob = normalize_pointwise_tensor_to_orig(
            post.mean,
            orig,
            name="posterior mean probability",
        )
        if not (0.0 <= mean_prob.min().item() and mean_prob.max().item() <= 1.0):
            mean_prob = torch.sigmoid(mean_prob)
        mean_prob = mean_prob.clamp(self.eps, 1.0 - self.eps)
        return mu_f, var_f, mean_prob, Xt

    def _apply_score_objective(self, score: Tensor, X: Optional[Tensor], *, name: str) -> Tensor:
        return apply_classification_objective_to_score(self, score, X=X, name=name)

    def _postprocess_pointwise_score(self, score: Tensor, mean_prob: Tensor, Xt: Tensor, X: Tensor, *, name: str) -> Tensor:
        score = self._apply_roi_weight_per_point(score, mean_prob, Xt)
        score = self._apply_noise_weight_per_point(score, Xt)

        pending = self._pending_penalty_per_point(Xt)
        if pending.shape == score.shape:
            score = score - pending
        elif pending.numel() == score.numel():
            score = score - pending.reshape_as(score)
        elif self.pending_penalty_weight > 0.0:
            raise RuntimeError(
                f"Pending penalty shape mismatch in {name}: "
                f"score.shape={tuple(score.shape)}, pending.shape={tuple(pending.shape)}"
            )

        score = align_pointwise_score_to_X(score, Xt, name=f"{name} score before objective")
        return self._apply_score_objective(score, X=X, name=name)


class qHeteroBinaryLatentStraddleAcquisition(_BaseHeteroBinaryLevelSetAcquisition):
    """heteroscedastic classification 用 straddle acquisition。境界に近く、かつ不確実な点を選びます。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        beta: 不確実性または sample deviation をどれだけ重視するかを決める係数。
        threshold: binary classification や level-set で使う境界値。
        **kwargs: 親クラスまたは BoTorch acquisition に渡す追加 keyword arguments。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        level-set estimation で最初に試しやすい acquisition です。
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """

    def __init__(
        self,
        model,
        beta: float = 2.0,
        threshold: float = 0.0,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.beta = float(beta)
        self.threshold = float(threshold)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        X_in = X if X.ndim > 2 else X.unsqueeze(0)
        original_batch_shape = X_in.shape[:-2]

        mu_f, var_f, mean_prob, Xt = self._latent_stats_and_mean_prob(X)
        score = (self.beta ** 0.5) * var_f.sqrt() - (mu_f - self.threshold).abs()
        score = self._postprocess_pointwise_score(
            score, mean_prob, Xt, X, name="qHeteroBinaryLatentStraddle"
        )
        out = self._reduce_q(score)
        self._check_output_shape(out, original_batch_shape, "qHeteroBinaryLatentStraddle")
        return out


class qHeteroBinaryICUAcquisition(_BaseHeteroBinaryLevelSetAcquisition):
    """heteroscedastic classification 用 ICU acquisition。contour / boundary 周辺の不確実性を評価します。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        X_in = X if X.ndim > 2 else X.unsqueeze(0)
        original_batch_shape = X_in.shape[:-2]

        _, _, mean_prob, Xt = self._latent_stats_and_mean_prob(X)
        score = 4.0 * mean_prob * (1.0 - mean_prob)
        score = self._postprocess_pointwise_score(score, mean_prob, Xt, X, name="qHeteroBinaryICU")
        out = self._reduce_q(score)
        self._check_output_shape(out, original_batch_shape, "qHeteroBinaryICU")
        return out


class qHeteroBinaryBoundaryVarianceAcquisition(_BaseHeteroBinaryLevelSetAcquisition):
    """heteroscedastic classification 用 boundary variance acquisition。境界近傍の posterior variance を重視します。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        threshold: binary classification や level-set で使う境界値。
        tau: soft PI や境界近傍重み付けに使う温度・幅パラメータ。
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
        model,
        threshold: float = 0.0,
        tau: float = 1.0,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.threshold = float(threshold)
        self.tau = float(tau)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        X_in = X if X.ndim > 2 else X.unsqueeze(0)
        original_batch_shape = X_in.shape[:-2]

        mu_f, var_f, mean_prob, Xt = self._latent_stats_and_mean_prob(X)
        score = var_f * boundary_kernel_weight(mu_f, self.threshold, tau=self.tau)
        score = self._postprocess_pointwise_score(
            score, mean_prob, Xt, X, name="qHeteroBinaryBoundaryVariance"
        )
        out = self._reduce_q(score)
        self._check_output_shape(out, original_batch_shape, "qHeteroBinaryBoundaryVariance")
        return out


class qHeteroBinaryClassEntropyAcquisition(_BaseHeteroBinaryLevelSetAcquisition):
    """heteroscedastic classification 用 class entropy acquisition。class probability の entropy を評価します。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        X_in = X if X.ndim > 2 else X.unsqueeze(0)
        original_batch_shape = X_in.shape[:-2]

        _, _, mean_prob, Xt = self._latent_stats_and_mean_prob(X)
        score = bernoulli_entropy(mean_prob, eps=self.eps)
        score = self._postprocess_pointwise_score(
            score, mean_prob, Xt, X, name="qHeteroBinaryClassEntropy"
        )
        out = self._reduce_q(score)
        self._check_output_shape(out, original_batch_shape, "qHeteroBinaryClassEntropy")
        return out

__all__ = [
    "qHeteroBinaryLatentStraddleAcquisition",
    "qHeteroBinaryICUAcquisition",
    "qHeteroBinaryBoundaryVarianceAcquisition",
    "qHeteroBinaryClassEntropyAcquisition",
]
