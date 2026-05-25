from __future__ import annotations

import math
from typing import Callable, Literal, Optional

import torch
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from .base import BasePendingPenaltyAcquisition, safe_logdet
from .utils import contour_uncertainty


RiskType = Optional[Literal["var", "cvar"]]


class _RegressionLevelSetScoreObjective(torch.nn.Module):
    """regression level-set acquisition の pointwise score に作用する objective。"""

    def __init__(
        self,
        n_w: Optional[int] = None,
        risk_type: RiskType = None,
        alpha: float = 0.5,
        maximize: bool = True,
        weight: float = 1.0,
        sign: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_w = None if n_w is None else int(n_w)
        self.risk_type = risk_type
        self.alpha = float(alpha)
        self.maximize = bool(maximize)
        self.weight = float(weight)
        self.sign = float(sign)

        if self.risk_type not in (None, "var", "cvar"):
            raise ValueError(f"Unknown risk_type: {self.risk_type}")
        if self.risk_type is not None and self.n_w is None:
            raise ValueError("risk_type is specified, but n_w is None.")
        if self.risk_type is not None and not (0.0 < self.alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1].")

    def forward(self, score: Tensor, X: Optional[Tensor] = None) -> Tensor:
        score = score * self.sign * self.weight

        if self.n_w is None or self.n_w <= 1:
            return score

        if X is not None:
            X_in = X if X.ndim > 2 else X.unsqueeze(0)
            if tuple(score.shape) == tuple(X_in.shape[:-2]):
                return score

        q_expanded = score.shape[-1]
        if q_expanded % self.n_w != 0:
            raise RuntimeError(
                f"score.shape[-1] must be divisible by n_w. "
                f"Got score.shape={tuple(score.shape)}, n_w={self.n_w}."
            )

        q = q_expanded // self.n_w
        score_w = score.reshape(*score.shape[:-1], q, self.n_w)

        if self.risk_type is None:
            return score_w.mean(dim=-1)

        descending = not self.maximize
        sorted_score = torch.sort(score_w, dim=-1, descending=descending).values
        k = max(1, int(math.ceil(self.n_w * self.alpha)))
        tail = sorted_score[..., :k]

        if self.risk_type == "var":
            return tail[..., -1]
        if self.risk_type == "cvar":
            return tail.mean(dim=-1)
        raise ValueError(f"Unknown risk_type: {self.risk_type}")


def _objective_X_for_score(score: Tensor, X: Optional[Tensor]) -> Optional[Tensor]:
    """Return an X argument compatible with score's q-batch semantics.

    Pointwise level-set acquisitions pass ``score.shape = batch_shape x q`` and can
    use the original candidate ``X.shape = batch_shape x q x d`` for objective
    shape verification.

    Joint level-set acquisitions first reduce over q / q*n_w and then pass
    ``score.shape = batch_shape``. In that case the original X would make
    BoTorch's ``MCAcquisitionObjective.__call__`` compare ``score.shape[-1]``
    with ``X.shape[-2]`` and fail, e.g. ``Got 128 and 1`` during initial-condition
    generation. For already joint-reduced scores, use a lightweight shape witness
    whose q-like dimension is the batch length so that score transforms such as
    sign / weight can still be applied without triggering the q=1 check.
    """
    if X is None or X.ndim < 3 or score.ndim == 0:
        return X

    if tuple(score.shape) == tuple(X.shape[:-2]):
        return score.unsqueeze(-1)

    return X


def _apply_regression_levelset_objective_to_score(
    owner,
    score: Tensor,
    X: Optional[Tensor] = None,
    name: str = "RegressionLevelSetAcquisition",
) -> Tensor:
    objective = getattr(owner, "objective", None)
    if objective is None:
        return score
    X_for_objective = _objective_X_for_score(score, X)
    try:
        out = objective(score, X=X_for_objective)
    except TypeError:
        out = objective(score)
    if not torch.is_tensor(out):
        raise RuntimeError(f"{name}: objective must return a Tensor. Got {type(out)}.")
    return out


def _posterior_mean_var(model: Model, X: Tensor, eps: float = 1e-12) -> tuple[Tensor, Tensor]:
    posterior = model.posterior(X)
    mean = posterior.mean.squeeze(-1)
    if hasattr(posterior, "variance"):
        var = posterior.variance.squeeze(-1).clamp_min(eps)
    else:
        var = posterior.mvn.covariance_matrix.diagonal(dim1=-2, dim2=-1).clamp_min(eps)
    return mean, var


def _boundary_kernel_weight(values: Tensor, h: float | Tensor, tau: float) -> Tensor:
    h_t = torch.as_tensor(h, device=values.device, dtype=values.dtype)
    tau_t = torch.as_tensor(tau, device=values.device, dtype=values.dtype).clamp_min(1e-8)
    return torch.exp(-0.5 * ((values - h_t) / tau_t).pow(2))


class qRegressionStraddleAcquisition(BasePendingPenaltyAcquisition):
    """regression 用 straddle acquisition。境界に近く、かつ不確実な点を選びます。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        beta: 不確実性または sample deviation をどれだけ重視するかを決める係数。
        h: level-set estimation で使う閾値。
        reduction: q-batch 方向の集約方法。通常は `mean` または `sum`。
        penalty_scale: この acquisition / objective の動作を制御するパラメータ。
        objective: posterior samples または計算済み score に作用する objective。InputPerturbation の q*n_w -> q 集約にも使えます。
        eps: 数値安定化用の微小値。
    
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
        beta: float = 1.0,
        h: float = 0.0,
        reduction: Literal["mean", "sum", "max"] = "mean",
        penalty_scale: float = 100.0,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        eps: float = 1e-9,
    ) -> None:
        super().__init__(model=model, penalty_scale=penalty_scale)
        self.beta = float(beta)
        self.h = float(h)
        self.reduction = reduction
        self.objective = objective
        self.eps = float(eps)

    def _reduce_q(self, score: Tensor) -> Tensor:
        if self.reduction == "mean":
            return score.mean(dim=-1)
        if self.reduction == "sum":
            return score.sum(dim=-1)
        if self.reduction == "max":
            return score.max(dim=-1).values
        raise ValueError("reduction must be one of 'mean', 'sum', or 'max'.")

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        mean, var = _posterior_mean_var(self.model, X, eps=self.eps)
        score = -(mean - self.h).abs() + self.beta * var.sqrt()
        score = _apply_regression_levelset_objective_to_score(
            self, score, X=X, name="qRegressionStraddle"
        )
        out = self._reduce_q(score)
        return self._apply_pending_penalty(X, out)


class qRegressionJointStraddleAcquisition(BasePendingPenaltyAcquisition):
    """regression 用 joint straddle acquisition。q-batch 全体の境界不確実性を評価します。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        beta: 不確実性または sample deviation をどれだけ重視するかを決める係数。
        h: level-set estimation で使う閾値。
        uncertainty_measure: この acquisition / objective の動作を制御するパラメータ。
        tau: soft PI や境界近傍重み付けに使う温度・幅パラメータ。
        penalty_scale: この acquisition / objective の動作を制御するパラメータ。
        objective: posterior samples または計算済み score に作用する objective。InputPerturbation の q*n_w -> q 集約にも使えます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    """

    def __init__(
        self,
        model: Model,
        beta: float = 1.0,
        h: float = 0.0,
        uncertainty_measure: Literal["logdet", "logdet1p", "trace"] = "logdet",
        tau: float = 1.0,
        penalty_scale: float = 20.0,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(model=model, penalty_scale=penalty_scale)
        self.beta = float(beta)
        self.h = float(h)
        self.uncertainty_measure = uncertainty_measure
        self.tau = float(tau)
        self.objective = objective

    def _uncertainty_score(self, covar: Tensor) -> Tensor:
        if self.uncertainty_measure == "trace":
            return covar.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        if self.uncertainty_measure == "logdet1p":
            q = covar.shape[-1]
            eye = torch.eye(q, device=covar.device, dtype=covar.dtype)
            return safe_logdet(eye + covar / (self.tau ** 2 + 1e-12))
        if self.uncertainty_measure == "logdet":
            return safe_logdet(covar)
        raise ValueError("uncertainty_measure must be 'logdet', 'logdet1p', or 'trace'.")

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        mean = posterior.mean.squeeze(-1)
        covar = posterior.mvn.covariance_matrix

        dist_to_h = (mean - self.h).abs()
        mean_term = -dist_to_h.mean(dim=-1)
        uncertainty = self._uncertainty_score(covar)
        score = mean_term + self.beta * uncertainty
        score = _apply_regression_levelset_objective_to_score(
            self, score, X=X, name="qRegressionJointStraddle"
        )
        return self._apply_pending_penalty(X, score)


class qRegressionICUAcquisition(BasePendingPenaltyAcquisition):
    """regression 用 ICU acquisition。contour / boundary 周辺の不確実性を評価します。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        h: level-set estimation で使う閾値。
        reduction: q-batch 方向の集約方法。通常は `mean` または `sum`。
        penalty_scale: この acquisition / objective の動作を制御するパラメータ。
        objective: posterior samples または計算済み score に作用する objective。InputPerturbation の q*n_w -> q 集約にも使えます。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    """

    def __init__(
        self,
        model: Model,
        h: float,
        reduction: Literal["mean", "sum", "max"] = "mean",
        penalty_scale: float = 50.0,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(model=model, penalty_scale=penalty_scale)
        self.h = float(h)
        self.reduction = reduction
        self.objective = objective

    def _reduce_q(self, score: Tensor) -> Tensor:
        if self.reduction == "mean":
            return score.mean(dim=-1)
        if self.reduction == "sum":
            return score.sum(dim=-1)
        if self.reduction == "max":
            return score.max(dim=-1).values
        raise ValueError("reduction must be one of 'mean', 'sum', or 'max'.")

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        score = contour_uncertainty(self.model, X, self.h)
        score = _apply_regression_levelset_objective_to_score(
            self, score, X=X, name="qRegressionICU"
        )
        out = self._reduce_q(score)
        return self._apply_pending_penalty(X, out)


class qRegressionBoundaryVarianceAcquisition(BasePendingPenaltyAcquisition):
    """regression 用 boundary variance acquisition。境界近傍の posterior variance を重視します。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        h: level-set estimation で使う閾値。
        tau: soft PI や境界近傍重み付けに使う温度・幅パラメータ。
        reduction: q-batch 方向の集約方法。通常は `mean` または `sum`。
        penalty_scale: この acquisition / objective の動作を制御するパラメータ。
        objective: posterior samples または計算済み score に作用する objective。InputPerturbation の q*n_w -> q 集約にも使えます。
        eps: 数値安定化用の微小値。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    """

    def __init__(
        self,
        model: Model,
        h: float,
        tau: float = 1.0,
        reduction: Literal["mean", "sum", "max"] = "mean",
        penalty_scale: float = 50.0,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(model=model, penalty_scale=penalty_scale)
        self.h = float(h)
        self.tau = float(tau)
        self.reduction = reduction
        self.objective = objective
        self.eps = float(eps)

    def _reduce_q(self, score: Tensor) -> Tensor:
        if self.reduction == "mean":
            return score.mean(dim=-1)
        if self.reduction == "sum":
            return score.sum(dim=-1)
        if self.reduction == "max":
            return score.max(dim=-1).values
        raise ValueError("reduction must be one of 'mean', 'sum', or 'max'.")

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        mean, var = _posterior_mean_var(self.model, X, eps=self.eps)
        score = var * _boundary_kernel_weight(mean, self.h, tau=self.tau)
        score = _apply_regression_levelset_objective_to_score(
            self, score, X=X, name="qRegressionBoundaryVariance"
        )
        out = self._reduce_q(score)
        return self._apply_pending_penalty(X, out)

__all__ = [
    "qRegressionStraddleAcquisition",
    "qRegressionJointStraddleAcquisition",
    "qRegressionICUAcquisition",
    "qRegressionBoundaryVarianceAcquisition",
]
