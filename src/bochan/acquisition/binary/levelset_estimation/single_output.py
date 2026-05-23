from __future__ import annotations

import math
from typing import Callable, Literal, Optional

import torch
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition.binary.base import (
    ReductionType,
    _BinaryClassificationAcqBase,
)
from ._utils import (
    align_pointwise_score_to_X,
    apply_classification_objective_to_score,
    bernoulli_entropy,
    boundary_kernel_weight,
)


class qBinaryLatentStraddleAcquisition(_BinaryClassificationAcqBase):
    """classification 用 straddle acquisition。境界に近く、かつ不確実な点を選びます。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        beta: 不確実性または sample deviation をどれだけ重視するかを決める係数。
        threshold: binary classification や level-set で使う境界値。
        reduction: q-batch 方向の集約方法。通常は `mean` または `sum`。
        pending_penalty_weight: X_pending 近傍を避ける penalty の強さ。
        pending_penalty_beta: X_pending penalty の距離減衰率。
        eps: 数値安定化用の微小値。
        objective: posterior samples または計算済み score に作用する objective。InputPerturbation の q*n_w -> q 集約にも使えます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        level-set estimation で最初に試しやすい acquisition です。
    """

    def __init__(
        self,
        model,
        beta: float = 1.0,
        threshold: float = 0.0,
        reduction: ReductionType = "mean",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 5.0,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ):
        super().__init__(
            model=model,
            reduction=reduction,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            eps=eps,
        )
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.objective = objective

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()

        latent_dist, orig, Xt = self._get_latent_dist_and_orig(X)
        mu = self._reshape_pointwise_tensor(latent_dist.mean, orig)
        var = self._reshape_pointwise_tensor(latent_dist.variance, orig).clamp_min(self.eps)
        sigma = var.sqrt()

        score = self.beta * sigma - torch.sqrt((mu - self.threshold).pow(2) + 1e-8)
        score = score - self._pending_penalty_per_point(Xt)

        score = align_pointwise_score_to_X(
            score,
            Xt,
            name="qBinaryLatentStraddle score before objective",
            reduce_extra="sum",
        )
        score = apply_classification_objective_to_score(
            self,
            score,
            X=X,
            name="qBinaryLatentStraddle",
        )

        out = self._reduce_q(score)
        self._check_output_shape(out, X.shape[:-2], "qBinaryLatentStraddle")
        return out


class qBinaryJointLatentStraddleAcquisition(_BinaryClassificationAcqBase):
    """classification 用 joint straddle acquisition。q-batch 全体の境界不確実性を評価します。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        beta: 不確実性または sample deviation をどれだけ重視するかを決める係数。
        target: straddle / margin 系で近づけたい目標値。
        uncertainty_mode: この acquisition / objective の動作を制御するパラメータ。
        boundary_mode: この acquisition / objective の動作を制御するパラメータ。
        tau: soft PI や境界近傍重み付けに使う温度・幅パラメータ。
        jitter: この acquisition / objective の動作を制御するパラメータ。
        eps: 数値安定化用の微小値。
        marginalize_pending: この acquisition / objective の動作を制御するパラメータ。
        same_batch_penalty_weight: 同一 q-batch 内の候補点同士が近すぎる場合の penalty の強さ。
        pending_penalty_weight: X_pending 近傍を避ける penalty の強さ。
        observed_penalty_weight: 観測済み点近傍を避ける penalty の強さ。
        distance_beta: この acquisition / objective の動作を制御するパラメータ。
        duplicate_tol: この acquisition / objective の動作を制御するパラメータ。
        hard_duplicate_penalty: 完全重複またはほぼ重複する候補に対する追加 penalty。
        X_observed: 既に観測済みの点。重複候補の抑制や参照点として使います。
        deepgp_num_samples: この acquisition / objective の動作を制御するパラメータ。
        objective: posterior samples または計算済み score に作用する objective。InputPerturbation の q*n_w -> q 集約にも使えます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    """

    def __init__(
        self,
        model,
        beta: float = 2.0,
        target: float = 0.0,
        uncertainty_mode: Literal["logdet1p", "logdet", "sqrt_trace"] = "logdet1p",
        boundary_mode: Literal["mean_abs", "l2_mean", "max_abs"] = "l2_mean",
        tau: float = 1.0,
        jitter: float = 1e-6,
        eps: float = 1e-10,
        marginalize_pending: bool = True,
        same_batch_penalty_weight: float = 0.1,
        pending_penalty_weight: float = 0.1,
        observed_penalty_weight: float = 0.0,
        distance_beta: float = 20.0,
        duplicate_tol: float = 1e-6,
        hard_duplicate_penalty: float = 1e6,
        X_observed: Optional[Tensor] = None,
        deepgp_num_samples: int = 10,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ):
        super().__init__(
            model=model,
            reduction="sum",
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=distance_beta,
            eps=eps,
        )
        self.beta = float(beta)
        self.target = float(target)
        self.uncertainty_mode = uncertainty_mode
        self.boundary_mode = boundary_mode
        self.tau = float(tau)
        self.jitter = float(jitter)
        self.marginalize_pending = bool(marginalize_pending)
        self.same_batch_penalty_weight = float(same_batch_penalty_weight)
        self.observed_penalty_weight = float(observed_penalty_weight)
        self.distance_beta = float(distance_beta)
        self.duplicate_tol = float(duplicate_tol)
        self.hard_duplicate_penalty = float(hard_duplicate_penalty)
        self.X_observed = None
        self.set_X_observed(X_observed)
        self.deepgp_num_samples = int(deepgp_num_samples)
        self.objective = objective

    def set_X_observed(self, X_observed: Optional[Tensor]) -> None:
        # observed points are constants during acquisition optimization
        self.X_observed = None if X_observed is None else X_observed.detach()

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        # sequential optimize_acqf passes previous candidates as X_pending.
        # They should be treated as constants, not as part of the current graph.
        self.X_pending = None if X_pending is None else X_pending.detach()

    @staticmethod
    def _flatten_points(X: Tensor) -> Tensor:
        return X.reshape(-1, X.shape[-1])

    def _transform_and_flatten_reference(self, Xref: Optional[Tensor]) -> Optional[Tensor]:
        if Xref is None or Xref.numel() == 0:
            return None
        Xref = Xref.detach()
        Xt = self._apply_input_transform(Xref)
        if isinstance(Xt, list):
            Xt = Xt[0]
        if isinstance(Xt, tuple):
            Xt = Xt[0]
        return self._flatten_points(Xt)

    def _latent_mean_and_cov(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """latent posterior の mean / covariance を返す。

        `_BinaryClassificationAcqBase._latent_mean_and_cov` を使うことで、
        InputPerturbation / DeepGP / wrapper latent_posterior / inner model fallback
        の扱いを single-output active-learning 系と揃える。
        """
        return super()._latent_mean_and_cov(X)

    def _joint_uncertainty(self, cov: Tensor) -> Tensor:
        q = cov.shape[-1]
        eye = torch.eye(q, dtype=cov.dtype, device=cov.device)

        if self.uncertainty_mode == "logdet1p":
            tau2 = max(self.tau ** 2, self.eps)
            mat = eye + cov / tau2
            sign, logabsdet = torch.linalg.slogdet(mat)
            if not torch.all(sign > 0):
                raise RuntimeError("Non-positive definite matrix encountered in logdet1p.")
            return 0.5 * logabsdet

        if self.uncertainty_mode == "logdet":
            sign, logabsdet = torch.linalg.slogdet(cov)
            if not torch.all(sign > 0):
                raise RuntimeError("Non-positive definite covariance encountered in logdet.")
            return 0.5 * logabsdet

        if self.uncertainty_mode == "sqrt_trace":
            tr = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1).clamp_min(self.eps)
            return tr.sqrt()

        raise ValueError(f"Unknown uncertainty_mode: {self.uncertainty_mode}")

    def _boundary_distance(self, mu: Tensor) -> Tensor:
        diff = mu - self.target
        if self.boundary_mode == "mean_abs":
            return diff.abs().mean(dim=-1)
        if self.boundary_mode == "l2_mean":
            return diff.pow(2).mean(dim=-1).sqrt()
        if self.boundary_mode == "max_abs":
            return diff.abs().max(dim=-1).values
        raise ValueError(f"Unknown boundary_mode: {self.boundary_mode}")

    def _joint_straddle_score(self, X: Tensor) -> Tensor:
        mu, cov = self._latent_mean_and_cov(X)
        uncertainty = self._joint_uncertainty(cov)
        boundary = self._boundary_distance(mu)
        return self.beta * uncertainty - boundary

    def _same_batch_repulsion(self, Xt: Tensor) -> Tensor:
        batch_shape = Xt.shape[:-2]
        q = Xt.shape[-2]
        d = Xt.shape[-1]

        if q <= 1 or self.same_batch_penalty_weight <= 0.0:
            return torch.zeros(batch_shape, device=Xt.device, dtype=Xt.dtype)

        Xb = Xt.reshape(-1, q, d)
        dmat = torch.cdist(Xb, Xb)
        eye_mask = torch.eye(q, device=Xt.device, dtype=torch.bool).unsqueeze(0)
        dmat = dmat.masked_fill(eye_mask, float("inf"))
        nearest = dmat.min(dim=-1).values

        soft_pen = torch.exp(-self.distance_beta * nearest).sum(dim=-1)
        hard_hits = (nearest <= self.duplicate_tol).to(Xt.dtype).sum(dim=-1)
        total = self.same_batch_penalty_weight * soft_pen + self.hard_duplicate_penalty * hard_hits
        return total.reshape(*batch_shape)

    def _reference_repulsion(self, Xt: Tensor, Xref: Optional[Tensor], weight: float) -> Tensor:
        batch_shape = Xt.shape[:-2]
        q = Xt.shape[-2]
        d = Xt.shape[-1]

        if weight <= 0.0:
            return torch.zeros(batch_shape, device=Xt.device, dtype=Xt.dtype)

        Xref2d = self._transform_and_flatten_reference(Xref)
        if Xref2d is None or Xref2d.numel() == 0:
            return torch.zeros(batch_shape, device=Xt.device, dtype=Xt.dtype)

        Xb = Xt.reshape(-1, q, d)
        dists = torch.cdist(Xb.reshape(-1, d), Xref2d)
        nearest = dists.min(dim=-1).values.reshape(-1, q)

        soft_pen = torch.exp(-self.distance_beta * nearest).sum(dim=-1)
        hard_hits = (nearest <= self.duplicate_tol).to(Xt.dtype).sum(dim=-1)
        total = weight * soft_pen + self.hard_duplicate_penalty * hard_hits
        return total.reshape(*batch_shape)

    def _repulsion_penalty(self, X: Tensor) -> Tensor:
        Xt = self._apply_input_transform(X)
        penalty = self._same_batch_repulsion(Xt)
        penalty = penalty + self._reference_repulsion(
            Xt, getattr(self, "X_pending", None), self.pending_penalty_weight
        )
        penalty = penalty + self._reference_repulsion(
            Xt, self.X_observed, self.observed_penalty_weight
        )
        return penalty

    @staticmethod
    def _expand_pending_to_batch(X_pending: Tensor, batch_shape: torch.Size) -> Tensor:
        if X_pending.ndim == 2:
            m, d = X_pending.shape
            return X_pending.view(*([1] * len(batch_shape)), m, d).expand(*batch_shape, m, d)
        if X_pending.ndim >= 3:
            m, d = X_pending.shape[-2], X_pending.shape[-1]
            Xp = X_pending.reshape(*([1] * len(batch_shape)), m, d)
            return Xp.expand(*batch_shape, m, d)
        raise ValueError(f"Unexpected X_pending shape: {tuple(X_pending.shape)}")

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        batch_shape = X.shape[:-2]
        Xp = getattr(self, "X_pending", None)
        if Xp is not None:
            Xp = Xp.detach()

        if Xp is None or Xp.numel() == 0 or not self.marginalize_pending:
            out = self._joint_straddle_score(X)
            out = out - self._repulsion_penalty(X)
            out = apply_classification_objective_to_score(
                self, out, X=X, name="qBinaryJointLatentStraddle"
            )
            self._check_output_shape(out, batch_shape, "qBinaryJointLatentStraddle")
            return out

        Xp_batch = self._expand_pending_to_batch(Xp, batch_shape)
        score_pending = self._joint_straddle_score(Xp_batch)
        X_all = torch.cat([Xp_batch, X], dim=-2)
        score_all = self._joint_straddle_score(X_all)

        out = score_all - score_pending
        out = out - self._repulsion_penalty(X)
        out = apply_classification_objective_to_score(
            self, out, X=X, name="qBinaryJointLatentStraddle"
        )
        self._check_output_shape(out, batch_shape, "qBinaryJointLatentStraddle")
        return out


class qBinaryICUAcquisition(_BinaryClassificationAcqBase):
    """classification 用 ICU acquisition。contour / boundary 周辺の不確実性を評価します。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        reduction: q-batch 方向の集約方法。通常は `mean` または `sum`。
        pending_penalty_weight: X_pending 近傍を避ける penalty の強さ。
        pending_penalty_beta: X_pending penalty の距離減衰率。
        eps: 数値安定化用の微小値。
        objective: posterior samples または計算済み score に作用する objective。InputPerturbation の q*n_w -> q 集約にも使えます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    """

    def __init__(
        self,
        model,
        reduction: ReductionType = "mean",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 5.0,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ):
        super().__init__(
            model=model,
            reduction=reduction,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            eps=eps,
        )
        self.objective = objective

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xt = self._apply_input_transform(X)
        prob_fn = getattr(self.model, "probability_posterior", None)
        post = prob_fn(X) if callable(prob_fn) else self.model.posterior(X)
        prob = self._reshape_pointwise_tensor(post.mean, Xt.shape[:-1])
        if not (0.0 <= prob.min().item() and prob.max().item() <= 1.0):
            prob = torch.sigmoid(prob)
        prob = prob.clamp(self.eps, 1.0 - self.eps)
        score = 4.0 * prob * (1.0 - prob)
        score = score - self._pending_penalty_per_point(Xt)
        score = align_pointwise_score_to_X(score, Xt, name="qBinaryICU score before objective")
        score = apply_classification_objective_to_score(self, score, X=X, name="qBinaryICU")
        out = self._reduce_q(score)
        self._check_output_shape(out, X.shape[:-2], "qBinaryICU")
        return out


class qBinaryBoundaryVarianceAcquisition(_BinaryClassificationAcqBase):
    """classification 用 boundary variance acquisition。境界近傍の posterior variance を重視します。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        threshold: binary classification や level-set で使う境界値。
        tau: soft PI や境界近傍重み付けに使う温度・幅パラメータ。
        reduction: q-batch 方向の集約方法。通常は `mean` または `sum`。
        pending_penalty_weight: X_pending 近傍を避ける penalty の強さ。
        pending_penalty_beta: X_pending penalty の距離減衰率。
        eps: 数値安定化用の微小値。
        objective: posterior samples または計算済み score に作用する objective。InputPerturbation の q*n_w -> q 集約にも使えます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    """

    def __init__(
        self,
        model,
        threshold: float = 0.0,
        tau: float = 1.0,
        reduction: ReductionType = "mean",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 5.0,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ):
        super().__init__(
            model=model,
            reduction=reduction,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            eps=eps,
        )
        self.threshold = float(threshold)
        self.tau = float(tau)
        self.objective = objective

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        latent_dist, orig, Xt = self._get_latent_dist_and_orig(X)
        mu = self._reshape_pointwise_tensor(latent_dist.mean, orig)
        var = self._reshape_pointwise_tensor(latent_dist.variance, orig).clamp_min(self.eps)
        score = var * boundary_kernel_weight(mu, self.threshold, tau=self.tau)
        score = score - self._pending_penalty_per_point(Xt)
        score = align_pointwise_score_to_X(score, Xt, name="qBinaryBoundaryVariance score before objective")
        score = apply_classification_objective_to_score(self, score, X=X, name="qBinaryBoundaryVariance")
        out = self._reduce_q(score)
        self._check_output_shape(out, X.shape[:-2], "qBinaryBoundaryVariance")
        return out


class qBinaryClassEntropyAcquisition(_BinaryClassificationAcqBase):
    """classification 用 class entropy acquisition。class probability の entropy を評価します。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        reduction: q-batch 方向の集約方法。通常は `mean` または `sum`。
        pending_penalty_weight: X_pending 近傍を避ける penalty の強さ。
        pending_penalty_beta: X_pending penalty の距離減衰率。
        eps: 数値安定化用の微小値。
        objective: posterior samples または計算済み score に作用する objective。InputPerturbation の q*n_w -> q 集約にも使えます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    """

    def __init__(
        self,
        model,
        reduction: ReductionType = "mean",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 5.0,
        eps: float = 1e-6,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ):
        super().__init__(
            model=model,
            reduction=reduction,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            eps=eps,
        )
        self.objective = objective

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xt = self._apply_input_transform(X)
        prob_fn = getattr(self.model, "probability_posterior", None)
        post = prob_fn(X) if callable(prob_fn) else self.model.posterior(X)
        prob = self._reshape_pointwise_tensor(post.mean, Xt.shape[:-1])
        if not (0.0 <= prob.min().item() and prob.max().item() <= 1.0):
            prob = torch.sigmoid(prob)
        score = bernoulli_entropy(prob, eps=self.eps)
        score = score - self._pending_penalty_per_point(Xt)
        score = align_pointwise_score_to_X(score, Xt, name="qBinaryClassEntropy score before objective")
        score = apply_classification_objective_to_score(self, score, X=X, name="qBinaryClassEntropy")
        out = self._reduce_q(score)
        self._check_output_shape(out, X.shape[:-2], "qBinaryClassEntropy")
        return out

__all__ = [
    "qBinaryLatentStraddleAcquisition",
    "qBinaryJointLatentStraddleAcquisition",
    "qBinaryICUAcquisition",
    "qBinaryBoundaryVarianceAcquisition",
    "qBinaryClassEntropyAcquisition",
]
