from __future__ import annotations

from typing import Callable, Literal, Optional

import torch
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition.binary.base import (
    ReductionType,
    _BinaryClassificationAcqBase,
)

try:
    from bochan.acquisition.binary.base import (
        ROIWeightMode,
        ROICombineType,
        NoiseWeightMode,
        NoiseCombineType,
    )
except ImportError:  # fallback for older base.py
    ROIWeightMode = Literal["none", "prob_above", "prob_below", "prob_interval", "custom"]
    ROICombineType = Literal["multiply", "add"]
    NoiseWeightMode = Literal["none", "inverse_linear", "inverse_exp", "custom"]
    NoiseCombineType = Literal["multiply", "subtract"]

try:
    from ...objective.binary import BinaryClassificationScoreObjectiveMixin
except ImportError:  # fallback when installed under bochan.acquisition
    from bochan.acquisition.objective.binary import BinaryClassificationScoreObjectiveMixin

from ._utils import (
    _align_pointwise_score_to_X,
    _apply_objective_to_pointwise_score,
)


class _HeteroBALDAcquisitionBinary(BinaryClassificationScoreObjectiveMixin, _BinaryClassificationAcqBase):
    """Heteroscedastic binary classification BALD."""

    def __init__(
        self,
        model,
        num_samples: int = 16,
        reduction: ReductionType = "mean",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        eps: float = 1e-6,
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
        noise_mode: NoiseWeightMode = "inverse_linear",
        noise_combine: NoiseCombineType = "multiply",
        noise_penalty_lambda: float = 1.0,
        noise_min_weight: float = 0.0,
        noise_weight_scale: float = 1.0,
        noise_model_outputs_log_var: bool = True,
        noise_weight_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
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
        )
        self.num_samples = int(num_samples)
        self._set_classification_score_objective(objective)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        X_in = X if X.dim() > 2 else X.unsqueeze(0)
        original_batch_shape = X_in.shape[:-2]

        probs, _, Xt = self._pointwise_latent_probs(X, num_samples=self.num_samples)
        entropy_conditional = self._binary_entropy(probs, self.eps).mean(dim=0)
        mean_prob = probs.mean(dim=0)
        mean_entropy = self._binary_entropy(mean_prob, self.eps)
        score = mean_entropy - entropy_conditional

        score = self._apply_roi_weight_per_point(score, mean_prob, Xt)
        score = self._apply_noise_weight_per_point(score, Xt)
        score = score - self._pending_penalty_per_point(Xt)
        score = _align_pointwise_score_to_X(
            score,
            Xt,
            name="HeteroBALD score before objective",
            reduce_extra="sum",
        )
        score = _apply_objective_to_pointwise_score(
            self,
            score,
            raw_X=X_in,
            expanded_X=Xt,
            name="HeteroBALD",
        )

        out = self._reduce_q(score)
        self._check_output_shape(out, original_batch_shape, "HeteroBALD")
        return out


class _HeteroProbabilityVarianceBinary(BinaryClassificationScoreObjectiveMixin, _BinaryClassificationAcqBase):
    """Heteroscedastic binary classification probability variance acquisition."""

    def __init__(
        self,
        model,
        reduction: ReductionType = "mean",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        eps: float = 1e-6,
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
        noise_mode: NoiseWeightMode = "inverse_linear",
        noise_combine: NoiseCombineType = "multiply",
        noise_penalty_lambda: float = 1.0,
        noise_min_weight: float = 0.0,
        noise_weight_scale: float = 1.0,
        noise_model_outputs_log_var: bool = True,
        noise_weight_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
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
        )
        self._set_classification_score_objective(objective)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        X_in = X if X.dim() > 2 else X.unsqueeze(0)
        original_batch_shape = X_in.shape[:-2]

        post = self.model.posterior(X_in)
        Xt = self._apply_input_transform(X_in)

        p = self._squeeze_last_output_dim(post.mean).clamp(self.eps, 1.0 - self.eps)
        p = _align_pointwise_score_to_X(
            p,
            Xt,
            name="HeteroProbabilityVariance probability",
            reduce_extra="mean",
        )
        score = p * (1.0 - p)

        score = self._apply_roi_weight_per_point(score, p, Xt)
        score = self._apply_noise_weight_per_point(score, Xt)
        score = score - self._pending_penalty_per_point(Xt)
        score = _align_pointwise_score_to_X(
            score,
            Xt,
            name="HeteroProbabilityVariance score before objective",
            reduce_extra="sum",
        )
        score = _apply_objective_to_pointwise_score(
            self,
            score,
            raw_X=X_in,
            expanded_X=Xt,
            name="HeteroProbabilityVariance",
        )

        out = self._reduce_q(score)
        self._check_output_shape(out, original_batch_shape, "HeteroProbabilityVariance")
        return out



# =========================================================
# Unified active-learning family names
# =========================================================
class _HeteroUncertaintySamplingBinary(BinaryClassificationScoreObjectiveMixin, _BinaryClassificationAcqBase):
    """Heteroscedastic binary uncertainty sampling.

    score_type:
        - "entropy": predictive entropy
        - "variance": probability variance p(1-p)
        - "least_confidence": margin / least-confidence uncertainty
    """

    def __init__(
        self,
        model,
        reduction: ReductionType = "mean",
        score_type: Literal["entropy", "variance", "least_confidence"] = "variance",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        eps: float = 1e-6,
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
        noise_mode: NoiseWeightMode = "inverse_linear",
        noise_combine: NoiseCombineType = "multiply",
        noise_penalty_lambda: float = 1.0,
        noise_min_weight: float = 0.0,
        noise_weight_scale: float = 1.0,
        noise_model_outputs_log_var: bool = True,
        noise_weight_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
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
        )
        self.score_type = score_type
        self._set_classification_score_objective(objective)

    def _uncertainty_score(self, p: Tensor) -> Tensor:
        if self.score_type == "variance":
            return p * (1.0 - p)
        if self.score_type == "entropy":
            return self._binary_entropy(p, self.eps)
        if self.score_type == "least_confidence":
            return 1.0 - torch.maximum(p, 1.0 - p)
        raise ValueError(f"Unknown score_type: {self.score_type}")

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        X_in = X if X.dim() > 2 else X.unsqueeze(0)
        original_batch_shape = X_in.shape[:-2]

        post = self.model.posterior(X_in)
        Xt = self._apply_input_transform(X_in)

        p = self._squeeze_last_output_dim(post.mean)
        pmin = p.min().item()
        pmax = p.max().item()
        if not (0.0 <= pmin and pmax <= 1.0):
            p = torch.sigmoid(p)
        p = p.clamp(self.eps, 1.0 - self.eps)
        p = _align_pointwise_score_to_X(
            p,
            Xt,
            name="HeteroUncertaintySampling probability",
            reduce_extra="mean",
        )

        score = self._uncertainty_score(p)
        score = self._apply_roi_weight_per_point(score, p, Xt)
        score = self._apply_noise_weight_per_point(score, Xt)
        score = score - self._pending_penalty_per_point(Xt)
        score = _align_pointwise_score_to_X(
            score,
            Xt,
            name="HeteroUncertaintySampling score before objective",
            reduce_extra="sum",
        )
        score = _apply_objective_to_pointwise_score(
            self,
            score,
            raw_X=X_in,
            expanded_X=Xt,
            name="HeteroUncertaintySampling",
        )

        out = self._reduce_q(score)
        self._check_output_shape(out, original_batch_shape, "HeteroUncertaintySampling")
        return out


class _qHeteroBinaryPredictiveEntropyAcquisition(_HeteroUncertaintySamplingBinary):
    def __init__(self, model, **kwargs):
        kwargs.pop("score_type", None)
        super().__init__(model=model, score_type="entropy", **kwargs)


class _qHeteroBinaryProbabilityVarianceAcquisition(_HeteroUncertaintySamplingBinary):
    def __init__(self, model, **kwargs):
        kwargs.pop("score_type", None)
        super().__init__(model=model, score_type="variance", **kwargs)


class _qHeteroBinaryMarginUncertaintyAcquisition(_HeteroUncertaintySamplingBinary):
    def __init__(self, model, **kwargs):
        kwargs.pop("score_type", None)
        super().__init__(model=model, score_type="least_confidence", **kwargs)


class _qHeteroBinaryIntegratedPosteriorVarianceProxy(_qHeteroBinaryProbabilityVarianceAcquisition):
    """Hetero binary IPV-style proxy based on noise-aware probability variance."""


# Canonical unified names






class qHeteroBinaryPredictiveEntropy(_qHeteroBinaryPredictiveEntropyAcquisition):
    """heteroscedastic classification 用 predictive entropy acquisition。予測分布の曖昧さが大きい点を選びます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        予測が曖昧な点を探索したい場合の基本的な active learning acquisition です。
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """
    pass


class qHeteroBinaryBALD(_HeteroBALDAcquisitionBinary):
    """heteroscedastic classification 用 BALD / mutual-information acquisition。モデル不確実性を減らす情報量の大きい点を選びます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        BALD は predictive entropy から条件付き entropy を引いた情報利得として解釈できます。
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """
    pass


class qHeteroBinaryProbabilityVariance(_qHeteroBinaryProbabilityVarianceAcquisition):
    """heteroscedastic classification 用 variance-based acquisition。posterior / probability / utility の分散が大きい点を選びます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """
    pass


class qHeteroBinaryMarginUncertainty(_qHeteroBinaryMarginUncertaintyAcquisition):
    """heteroscedastic classification 用 margin uncertainty acquisition。決定境界または class 境界に近い点を選びます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """
    pass


class qHeteroBinaryIntegratedPosteriorVariance(_qHeteroBinaryIntegratedPosteriorVarianceProxy):
    """heteroscedastic classification 用 variance-based acquisition。posterior / probability / utility の分散が大きい点を選びます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """
    pass

__all__ = [
    "qHeteroBinaryPredictiveEntropy",
    "qHeteroBinaryBALD",
    "qHeteroBinaryProbabilityVariance",
    "qHeteroBinaryMarginUncertainty",
    "qHeteroBinaryIntegratedPosteriorVariance",
]
