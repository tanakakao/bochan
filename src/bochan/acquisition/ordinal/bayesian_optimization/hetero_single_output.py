from __future__ import annotations

import math
from typing import Callable, Literal, Optional, Sequence

import torch
from torch import Tensor

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform

from ..hetero_utils import (
    _normal_cdf,
    _normal_pdf,
    compute_hetero_ordinal_best_f,
    get_hetero_ordinal_summary,
    reduce_q,
)


RiskType = Optional[Literal["var", "cvar"]]


class _HeteroOrdinalNormalScoreObjective(torch.nn.Module):
    """
    hetero ordinal BO score objective.

    score: (*batch, q) or (*batch, q * n_w) -> optionally (*batch, q)
    """

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


def _apply_hetero_ordinal_normal_objective_to_score(
    owner,
    score: Tensor,
    X: Optional[Tensor] = None,
    name: str = "HeteroOrdinalNormalAcquisition",
) -> Tensor:
    objective = getattr(owner, "objective", None)
    if objective is None:
        return score

    try:
        out = objective(score, X=X)
    except TypeError:
        out = objective(score)

    if not torch.is_tensor(out):
        raise RuntimeError(f"{name}: objective must return a Tensor. Got {type(out)}.")
    return out


class _BaseHeteroOrdinalBOAcquisition(AcquisitionFunction):
    def __init__(
        self,
        model: Model,
        *,
        utility_values: Optional[Sequence[float] | Tensor] = None,
        noise_penalty: float = 0.0,
        variance_scale: float = 1.0,
        tau: float = 1e-6,
        default_sigma: float = 0.0,
        reduce: str = "max",
        eps: float = 1e-12,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(model=model)
        self.utility_values = utility_values
        self.noise_penalty = float(noise_penalty)
        self.variance_scale = float(variance_scale)
        self.tau = float(tau)
        self.default_sigma = float(default_sigma)
        self.reduce = str(reduce)
        self.eps = float(eps)
        self.objective = objective
        self.X_pending: Optional[Tensor] = None

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        self.X_pending = None if X_pending is None else X_pending.detach()

    def _summary(self, X: Tensor) -> dict[str, Tensor]:
        return get_hetero_ordinal_summary(
            self.model,
            X,
            utility_values=self.utility_values,
            noise_penalty=self.noise_penalty,
            variance_scale=self.variance_scale,
            tau=self.tau,
            default_sigma=self.default_sigma,
            eps=self.eps,
        )

    def _apply_objective_to_score(self, score: Tensor, X: Tensor, name: str) -> Tensor:
        return _apply_hetero_ordinal_normal_objective_to_score(self, score, X=X, name=name)


class qHeteroOrdinalExpectedUtility(_BaseHeteroOrdinalBOAcquisition):
    """heteroscedastic ordinal 用 expected utility acquisition。ordinal class を utility に変換して最大化します。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """
    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        summary = self._summary(X)
        score = self._apply_objective_to_score(
            summary["robust_mean"],
            X=X,
            name="qHeteroOrdinalExpectedUtility"
        )
        return reduce_q(score, reduce=self.reduce)


class qHeteroOrdinalExpectedImprovement(_BaseHeteroOrdinalBOAcquisition):
    """heteroscedastic ordinal 用 expected improvement acquisition。現在の best_f からの改善量を評価します。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        best_f: 既存観測点または baseline から計算した現在の最良値。
        **kwargs: 親クラスまたは BoTorch acquisition に渡す追加 keyword arguments。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """
    def __init__(self, model: Model, best_f: float | Tensor, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.best_f = best_f

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        summary = self._summary(X)
        best_f = torch.as_tensor(
            self.best_f,
            device=summary["robust_mean"].device,
            dtype=summary["robust_mean"].dtype,
        )
        z = (summary["robust_mean"] - best_f) / summary["total_std"].clamp_min(self.eps)
        improvement = (
            (summary["robust_mean"] - best_f) * _normal_cdf(z)
            + summary["total_std"] * _normal_pdf(z)
        )
        improvement = self._apply_objective_to_score(
            improvement,
            X=X,
            name="qHeteroOrdinalExpectedImprovement"
        )
        return reduce_q(improvement, reduce=self.reduce)


class qHeteroOrdinalProbabilityOfImprovement(_BaseHeteroOrdinalBOAcquisition):
    """heteroscedastic ordinal 用 probability of improvement acquisition。best_f を上回る確率を評価します。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        best_f: 既存観測点または baseline から計算した現在の最良値。
        **kwargs: 親クラスまたは BoTorch acquisition に渡す追加 keyword arguments。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """
    def __init__(self, model: Model, best_f: float | Tensor, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.best_f = best_f

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        summary = self._summary(X)
        best_f = torch.as_tensor(
            self.best_f,
            device=summary["robust_mean"].device,
            dtype=summary["robust_mean"].dtype,
        )
        z = (summary["robust_mean"] - best_f) / summary["total_std"].clamp_min(self.eps)
        score = _normal_cdf(z)
        score = self._apply_objective_to_score(
            score,
            X=X,
            name="qHeteroOrdinalProbabilityOfImprovement"
        )
        return reduce_q(score, reduce=self.reduce)


class qHeteroOrdinalExpectedUtilityUpperConfidenceBound(_BaseHeteroOrdinalBOAcquisition):
    """heteroscedastic ordinal 用 upper confidence bound acquisition。平均と不確実性を組み合わせて探索します。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        beta: 不確実性または sample deviation をどれだけ重視するかを決める係数。
        **kwargs: 親クラスまたは BoTorch acquisition に渡す追加 keyword arguments。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """
    def __init__(self, model: Model, beta: float = 2.0, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.beta = float(beta)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        summary = self._summary(X)
        score = summary["robust_mean"] + math.sqrt(self.beta) * summary["total_std"]
        score = self._apply_objective_to_score(
            score,
            X=X,
            name="HeteroOrdinalExpectedUtilityUpperConfidenceBound",
        )
        return reduce_q(score, reduce=self.reduce)

__all__ = [
    "qHeteroOrdinalExpectedUtility",
    "qHeteroOrdinalExpectedImprovement",
    "qHeteroOrdinalProbabilityOfImprovement",
    "qHeteroOrdinalExpectedUtilityUpperConfidenceBound",
]
