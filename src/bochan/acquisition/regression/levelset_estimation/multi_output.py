from __future__ import annotations

from typing import Literal, Optional, Sequence

import torch
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from .base import BasePendingPenaltyAcquisition, safe_logdet


ReductionType = Literal["mean", "sum", "max", "min"]
UncertaintyMode = Literal["sum_std", "mean_std", "max_std"]
BoundaryMode = Literal["common_satisfaction", "distance_to_threshold"]




def _normal_cdf(z: Tensor) -> Tensor:
    return 0.5 * (1.0 + torch.erf(z / torch.sqrt(torch.as_tensor(2.0, device=z.device, dtype=z.dtype))))


def _as_1d_thresholds(h: Sequence[float] | Tensor, *, device, dtype) -> Tensor:
    return torch.as_tensor(h, device=device, dtype=dtype).reshape(-1)


def _check_output_dim(mean: Tensor, thresholds: Tensor, name: str) -> None:
    if mean.size(-1) != thresholds.numel():
        raise ValueError(
            f"{name}: number of thresholds ({thresholds.numel()}) does not match "
            f"number of outputs ({mean.size(-1)})."
        )


def _reduce_q(score: Tensor, reduction: ReductionType) -> Tensor:
    if reduction == "mean":
        return score.mean(dim=-1)
    if reduction == "sum":
        return score.sum(dim=-1)
    if reduction == "max":
        return score.max(dim=-1).values
    if reduction == "min":
        return score.min(dim=-1).values
    raise ValueError(f"Unknown reduction: {reduction}")


class qMultiOutputRegressionStraddleAcquisition(BasePendingPenaltyAcquisition):
    """multi-output regression 用 straddle acquisition。境界に近く、かつ不確実な点を選びます。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        beta: 不確実性または sample deviation をどれだけ重視するかを決める係数。
        h: level-set estimation で使う閾値。
        penalty_scale: この acquisition / objective の動作を制御するパラメータ。
        uncertainty_mode: この acquisition / objective の動作を制御するパラメータ。
        boundary_mode: この acquisition / objective の動作を制御するパラメータ。
        reduction: q-batch 方向の集約方法。通常は `mean` または `sum`。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        level-set estimation で最初に試しやすい acquisition です。
    """

    def __init__(
        self,
        model: Model,
        beta: float,
        h: Sequence[float] | Tensor,
        penalty_scale: float = 10.0,
        uncertainty_mode: UncertaintyMode = "sum_std",
        boundary_mode: BoundaryMode = "common_satisfaction",
        reduction: ReductionType = "sum",
    ) -> None:
        super().__init__(model=model, penalty_scale=penalty_scale)
        self.beta = float(beta)
        self.uncertainty_mode = uncertainty_mode
        self.boundary_mode = boundary_mode
        self.reduction = reduction
        self.register_buffer("thresholds", torch.as_tensor(h, dtype=torch.float32).reshape(-1))

    def _uncertainty(self, std: Tensor) -> Tensor:
        if self.uncertainty_mode == "sum_std":
            return std.sum(dim=-1)
        if self.uncertainty_mode == "mean_std":
            return std.mean(dim=-1)
        if self.uncertainty_mode == "max_std":
            return std.max(dim=-1).values
        raise ValueError(f"Unknown uncertainty_mode: {self.uncertainty_mode}")

    def _boundary_penalty(self, mean: Tensor, thresholds: Tensor) -> Tensor:
        thresholds_view = thresholds.view(*((1,) * (mean.ndim - 1)), -1)
        if self.boundary_mode == "common_satisfaction":
            return torch.relu(thresholds_view - mean).sum(dim=-1)
        if self.boundary_mode == "distance_to_threshold":
            return (mean - thresholds_view).abs().sum(dim=-1)
        raise ValueError(f"Unknown boundary_mode: {self.boundary_mode}")

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        mean = posterior.mean
        std = posterior.variance.clamp_min(1e-9).sqrt()

        thresholds = self.thresholds.to(device=mean.device, dtype=mean.dtype)
        _check_output_dim(mean, thresholds, "qMultiOutputRegressionStraddleAcquisition")

        boundary = self._boundary_penalty(mean, thresholds)  # (*batch, q)
        uncertainty = self._uncertainty(std)                 # (*batch, q)
        score = -boundary + self.beta * uncertainty
        out = _reduce_q(score, reduction=self.reduction)
        return self._apply_pending_penalty(X, out)


class qMultiOutputRegressionJointStraddleAcquisition(BasePendingPenaltyAcquisition):
    """multi-output regression 用 joint straddle acquisition。q-batch 全体の境界不確実性を評価します。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        beta: 不確実性または sample deviation をどれだけ重視するかを決める係数。
        h: level-set estimation で使う閾値。
        penalty_scale: この acquisition / objective の動作を制御するパラメータ。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    """

    def __init__(
        self,
        model: Model,
        beta: float,
        h: Sequence[float] | Tensor,
        penalty_scale: float = 20.0,
    ) -> None:
        super().__init__(model=model, penalty_scale=penalty_scale)
        self.beta = float(beta)
        self.register_buffer("thresholds", torch.as_tensor(h, dtype=torch.float32).reshape(-1))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        mean = posterior.mean                    # (*batch, q, m)
        covar = posterior.mvn.covariance_matrix  # (*batch, q*m, q*m)

        thresholds = self.thresholds.to(device=mean.device, dtype=mean.dtype)
        _check_output_dim(mean, thresholds, "qMultiOutputRegressionJointStraddleAcquisition")

        thresholds_view = thresholds.view(*((1,) * (mean.ndim - 1)), -1)
        margin = torch.relu(thresholds_view - mean)
        mean_term = -margin.sum(dim=(-2, -1))
        logdet = safe_logdet(covar)
        score = mean_term + self.beta * logdet
        return self._apply_pending_penalty(X, score)


class _qMultiOutputRegressionJointBoundaryVarianceAcquisition(BasePendingPenaltyAcquisition):
    """多出力回帰モデルに対する joint boundary variance。

    各出力 j に threshold h_j を設定し、q 点すべてで各出力が threshold を
    超える joint probability を計算する。その Bernoulli variance
    p_joint * (1 - p_joint) を acquisition score とする。
    """

    def __init__(
        self,
        model: Model,
        h: Sequence[float] | Tensor,
        penalty_scale: float = 20.0,
    ) -> None:
        super().__init__(model=model, penalty_scale=penalty_scale)
        self.register_buffer("h", torch.as_tensor(h, dtype=torch.float32).reshape(-1))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        mean = posterior.mean
        sigma = posterior.variance.clamp_min(1e-9).sqrt()

        thresholds = self.h.to(device=mean.device, dtype=mean.dtype)
        _check_output_dim(mean, thresholds, "_qMultiOutputRegressionJointBoundaryVarianceAcquisition")

        probs = []
        for i in range(thresholds.numel()):
            mean_i = mean[..., i]
            sigma_i = sigma[..., i]
            z = (mean_i - thresholds[i]) / sigma_i
            p_i = _normal_cdf(z)
            probs.append(p_i.prod(dim=-1))

        joint_prob = torch.stack(probs, dim=-1).prod(dim=-1)
        variance = joint_prob * (1.0 - joint_prob)
        return self._apply_pending_penalty(X, variance)


class qMultiOutputRegressionICUAcquisition(BasePendingPenaltyAcquisition):
    """multi-output regression 用 ICU acquisition。contour / boundary 周辺の不確実性を評価します。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        h: level-set estimation で使う閾値。
        reduction: q-batch 方向の集約方法。通常は `mean` または `sum`。
        output_reduction: multi-output の出力方向の集約方法。
        joint_boundary: この acquisition / objective の動作を制御するパラメータ。
        penalty_scale: この acquisition / objective の動作を制御するパラメータ。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    """

    def __init__(
        self,
        model: Model,
        h: Sequence[float] | Tensor,
        reduction: ReductionType = "mean",
        output_reduction: ReductionType = "mean",
        joint_boundary: bool = False,
        penalty_scale: float = 50.0,
    ) -> None:
        super().__init__(model=model, penalty_scale=penalty_scale)
        self.reduction = reduction
        self.output_reduction = output_reduction
        self.joint_boundary = bool(joint_boundary)
        self.register_buffer("thresholds", torch.as_tensor(h, dtype=torch.float32).reshape(-1))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        mean = posterior.mean
        sigma = posterior.variance.clamp_min(1e-9).sqrt()

        thresholds = self.thresholds.to(device=mean.device, dtype=mean.dtype)
        _check_output_dim(mean, thresholds, "qMultiOutputRegressionICUAcquisition")

        thresholds_view = thresholds.view(*((1,) * (mean.ndim - 1)), -1)
        z = (mean - thresholds_view) / sigma
        poe = _normal_cdf(z)  # (*batch, q, m)

        if self.joint_boundary:
            p_joint_per_q = poe.prod(dim=-1)  # (*batch, q)
            score = p_joint_per_q * (1.0 - p_joint_per_q)
        else:
            score_per_output = poe * (1.0 - poe)
            score = _reduce_q(score_per_output, reduction=self.output_reduction)

        out = _reduce_q(score, reduction=self.reduction)
        return self._apply_pending_penalty(X, out)


# =========================================================



class qMultiOutputRegressionBoundaryVarianceAcquisition(_qMultiOutputRegressionJointBoundaryVarianceAcquisition):
    """multi-output regression 用 boundary variance acquisition。境界近傍の posterior variance を重視します。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    """
    pass

__all__ = [
    "qMultiOutputRegressionStraddleAcquisition",
    "qMultiOutputRegressionJointStraddleAcquisition",
    "qMultiOutputRegressionICUAcquisition",
    "qMultiOutputRegressionBoundaryVarianceAcquisition",
]
