from __future__ import annotations

import math
from typing import Callable, Literal, Optional

import torch
from torch import Tensor
from torch import nn

from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.acquisition.objective import MCAcquisitionObjective


RiskType = Optional[Literal["var", "cvar"]]
AggregatedRiskMode = Literal["ignore", "error"]

BinaryClassificationScoreShapeMode = Literal[
    "auto",
    "pointwise",       # (*batch, q_like)
    "multioutput_qm",  # (*batch, q_like, m)
    "multioutput_mq",  # (*batch, m, q_like)
    "aggregated",      # (*batch,)
]


# ============================================================
# Common helpers
# ============================================================


def _validate_n_w_risk(
    *,
    n_w: Optional[int],
    risk_type: RiskType,
    alpha: float,
) -> None:
    if n_w is not None and int(n_w) <= 0:
        raise ValueError("n_w must be a positive integer or None.")

    if risk_type not in (None, "var", "cvar"):
        raise ValueError(f"Unknown risk_type: {risk_type!r}.")

    if risk_type is not None and n_w is None:
        raise ValueError("risk_type is specified, but n_w is None.")

    if risk_type is not None and not (0.0 < float(alpha) <= 1.0):
        raise ValueError("alpha must be in (0, 1].")


def _aggregate_score_w(
    score_w: Tensor,
    *,
    n_w: int,
    risk_type: RiskType,
    alpha: float,
    risk_dim: int,
    maximize: bool = True,
) -> Tensor:
    """
    Aggregate the perturbation axis.

    Args:
        score_w:
            Tensor containing an explicit perturbation axis.
        n_w:
            Number of perturbations.
        risk_type:
            None, "var", or "cvar".
        alpha:
            Tail fraction for VaR / CVaR.
        risk_dim:
            Dimension corresponding to n_w.
        maximize:
            If True, smaller score values are treated as the worst tail.
            If False, larger score values are treated as the worst tail.
    """
    if risk_type is None:
        return score_w.mean(dim=risk_dim)

    descending = not maximize
    sorted_score = torch.sort(score_w, dim=risk_dim, descending=descending).values

    k = max(1, int(math.ceil(int(n_w) * float(alpha))))
    tail = sorted_score.narrow(dim=risk_dim, start=0, length=k)

    if risk_type == "var":
        return tail.select(dim=risk_dim, index=k - 1)

    if risk_type == "cvar":
        return tail.mean(dim=risk_dim)

    raise ValueError(f"Unknown risk_type: {risk_type!r}.")


# ============================================================
# 1. Single-output classification score objective
# ============================================================


class BinaryClassificationScoreObjective(MCAcquisitionObjective):
    """
    InputPerturbation で展開された binary classification score を q に戻す objective。

    Expected input:
        samples.shape = (*sample_shape, *batch_shape, q * n_w)
        or
        samples.shape = (*sample_shape, *batch_shape, q * n_w, 1)

    Return:
        samples.shape = (*sample_shape, *batch_shape, q)
    """

    def __init__(
        self,
        n_w: int,
        risk_type: str | None = None,
        alpha: float = 0.8,
        maximize: bool = True,
    ) -> None:
        super().__init__()
        self.n_w = int(n_w)
        self.risk_type = risk_type
        self.alpha = float(alpha)
        self.maximize = bool(maximize)

    def _is_pointwise_score(self, score: Tensor, X: Tensor | None) -> bool:
        if X is None:
            return False

        if score.ndim < 1:
            return False

        q = X.shape[-2]
        q_expanded_expected = q * self.n_w

        if score.shape[-1] != q_expanded_expected:
            return False

        x_batch_shape = X.shape[:-2]

        if len(x_batch_shape) == 0:
            return True

        if score.ndim < len(x_batch_shape) + 1:
            return False

        score_batch_tail = score.shape[-(len(x_batch_shape) + 1):-1]

        return torch.Size(score_batch_tail) == torch.Size(x_batch_shape)

    def _is_aggregated_score(self, score: Tensor, X: Tensor | None) -> bool:
        if X is None:
            return False

        if score.ndim < 1:
            return False

        q = X.shape[-2]

        if score.shape[-1] != q:
            return False

        x_batch_shape = X.shape[:-2]

        if len(x_batch_shape) == 0:
            return True

        if score.ndim < len(x_batch_shape) + 1:
            return False

        score_batch_tail = score.shape[-(len(x_batch_shape) + 1):-1]

        return torch.Size(score_batch_tail) == torch.Size(x_batch_shape)

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        """
        Args:
            samples:
                probability / score samples.
                BoTorch の MCAcquisitionObjective では、この引数名は samples にする。
            X:
                候補点。shape = (*batch_shape, q, d)

        Returns:
            Tensor:
                q*n_w を q に集約した score。
        """
        score = samples

        # binary output dim を許容する
        # [..., q_like, 1] -> [..., q_like]
        if score.ndim >= 1 and score.shape[-1] == 1:
            score = score.squeeze(-1)

        # n_w > 1 のときは pointwise 判定を先に行う。
        # score.shape[-1] == q*n_w の場合、必ず q に集約する。
        if self._is_pointwise_score(score, X):
            q_expanded = score.shape[-1]

            if q_expanded % self.n_w != 0:
                raise RuntimeError(
                    f"Expanded q dimension must be divisible by n_w={self.n_w}. "
                    f"Got q_expanded={q_expanded}."
                )

            q = q_expanded // self.n_w

            # [..., q*n_w] -> [..., q, n_w]
            score = score.reshape(*score.shape[:-1], q, self.n_w)

            if self.risk_type is None:
                return score.mean(dim=-1)

            if self.risk_type == "var":
                return score.var(dim=-1, unbiased=False)

            if self.risk_type == "cvar":
                k = max(1, int(self.alpha * self.n_w))
                sorted_score = score.sort(dim=-1).values

                if self.maximize:
                    # maximize では悪い側 = 小さい値側
                    return sorted_score[..., :k].mean(dim=-1)

                # minimize では悪い側 = 大きい値側
                return sorted_score[..., -k:].mean(dim=-1)

            raise ValueError(f"Unknown risk_type: {self.risk_type}")

        # すでに q に集約済みならそのまま返す
        if self._is_aggregated_score(score, X):
            return score

        raise RuntimeError(
            "ClassificationScoreObjective expected either an aggregated score "
            "or a pointwise score with shape "
            "(*sample_shape, *batch_shape, q_like). "
            f"Got samples.shape={tuple(samples.shape)}, "
            f"score.shape={tuple(score.shape)}, "
            f"X.shape={None if X is None else tuple(X.shape)}."
        )


# ============================================================
# 2. Multi-output classification score objective
# ============================================================


class MultiOutputBinaryClassificationScoreObjective(nn.Module):
    """multi-output classification 用 objective。
    
    Args:
        n_w: InputPerturbation で 1 点あたりに展開される摂動数。
        risk_type: InputPerturbation 集約の risk 種類。`None`、`var`、`cvar`。
        alpha: risk 集約または qNEHVI の近似設定に使うパラメータ。
        aggregated_risk_mode: この acquisition / objective の動作を制御するパラメータ。
        score_shape_mode: この acquisition / objective の動作を制御するパラメータ。
    
    Returns:
        Tensor: 入力 samples または score を変換・集約した objective value。
    
    Notes:
        posterior samples ではなく、計算済み acquisition score に作用する objective と、qEHVI などの samples objective を区別して使ってください。
    """

    def __init__(
        self,
        n_w: Optional[int] = None,
        risk_type: RiskType = None,
        alpha: float = 0.5,
        aggregated_risk_mode: AggregatedRiskMode = "ignore",
        score_shape_mode: BinaryClassificationScoreShapeMode = "auto",
    ) -> None:
        super().__init__()

        self.n_w = None if n_w is None else int(n_w)
        self.risk_type = risk_type
        self.alpha = float(alpha)
        self.aggregated_risk_mode = aggregated_risk_mode
        self.score_shape_mode = score_shape_mode

        _validate_n_w_risk(
            n_w=self.n_w,
            risk_type=self.risk_type,
            alpha=self.alpha,
        )

        if self.aggregated_risk_mode not in ("ignore", "error"):
            raise ValueError("aggregated_risk_mode must be 'ignore' or 'error'.")

        if self.score_shape_mode not in (
            "auto",
            "pointwise",
            "multioutput_qm",
            "multioutput_mq",
            "aggregated",
        ):
            raise ValueError(
                "score_shape_mode must be one of "
                "'auto', 'pointwise', 'multioutput_qm', "
                "'multioutput_mq', or 'aggregated'."
            )

    @staticmethod
    def _ensure_q_batch(X: Tensor) -> Tensor:
        return X if X.dim() > 2 else X.unsqueeze(0)

    def _batch_shape_from_X(self, X: Optional[Tensor]) -> Optional[torch.Size]:
        if X is None:
            return None

        Xq = self._ensure_q_batch(X)
        return Xq.shape[:-2]

    def _q_from_X(self, X: Optional[Tensor]) -> Optional[int]:
        if X is None:
            return None

        Xq = self._ensure_q_batch(X)
        return int(Xq.shape[-2])

    def _infer_score_shape_mode(
        self,
        score: Tensor,
        X: Optional[Tensor],
    ) -> BinaryClassificationScoreShapeMode:
        if self.score_shape_mode != "auto":
            return self.score_shape_mode

        if score.ndim == 0:
            return "aggregated"

        batch_shape = self._batch_shape_from_X(X)
        q = self._q_from_X(X)

        if batch_shape is not None:
            if tuple(score.shape) == tuple(batch_shape):
                return "aggregated"

            if score.ndim >= 1 and tuple(score.shape[:-1]) == tuple(batch_shape):
                return "pointwise"

            if score.ndim >= 2 and tuple(score.shape[:-2]) == tuple(batch_shape):
                if q is not None and self.n_w is not None:
                    q_expanded = q * int(self.n_w)

                    if score.shape[-2] in (q, q_expanded):
                        return "multioutput_qm"

                    if score.shape[-1] in (q, q_expanded):
                        return "multioutput_mq"

                return "multioutput_qm"

        if score.ndim == 1:
            return "pointwise"

        if score.ndim >= 2:
            return "multioutput_qm"

        return "aggregated"

    def _handle_aggregated_score(self, score: Tensor) -> Tensor:
        if (
            self.n_w is not None
            and self.n_w > 1
            and self.aggregated_risk_mode == "error"
        ):
            raise RuntimeError(
                "MultiOutputClassificationScoreObjective received an aggregated / joint score "
                f"with shape={tuple(score.shape)}. InputPerturbation aggregation is only valid "
                "for pointwise scores with shape (*batch, q * n_w), "
                "(*batch, q * n_w, m), or (*batch, m, q * n_w)."
            )

        return score

    def _aggregate_pointwise_score(self, score: Tensor) -> Tensor:
        if self.n_w is None or self.n_w <= 1:
            return score

        q_expanded = score.shape[-1]

        if q_expanded % int(self.n_w) != 0:
            raise RuntimeError(
                "score.shape[-1] must be divisible by n_w. "
                f"Got score.shape={tuple(score.shape)}, n_w={self.n_w}."
            )

        q = q_expanded // int(self.n_w)
        score_w = score.reshape(*score.shape[:-1], q, int(self.n_w))

        return _aggregate_score_w(
            score_w,
            n_w=int(self.n_w),
            risk_type=self.risk_type,
            alpha=self.alpha,
            risk_dim=-1,
            maximize=True,
        )

    def _aggregate_multioutput_qm_score(self, score: Tensor) -> Tensor:
        if self.n_w is None or self.n_w <= 1:
            return score

        if score.ndim < 2:
            raise RuntimeError(
                "multioutput_qm score must have shape (*batch, q_like, m). "
                f"Got shape={tuple(score.shape)}."
            )

        q_expanded = score.shape[-2]
        m = score.shape[-1]

        if q_expanded % int(self.n_w) != 0:
            raise RuntimeError(
                "score.shape[-2] must be divisible by n_w. "
                f"Got score.shape={tuple(score.shape)}, n_w={self.n_w}."
            )

        q = q_expanded // int(self.n_w)
        score_w = score.reshape(*score.shape[:-2], q, int(self.n_w), m)

        return _aggregate_score_w(
            score_w,
            n_w=int(self.n_w),
            risk_type=self.risk_type,
            alpha=self.alpha,
            risk_dim=-2,
            maximize=True,
        )

    def _aggregate_multioutput_mq_score(self, score: Tensor) -> Tensor:
        if score.ndim < 2:
            raise RuntimeError(
                "multioutput_mq score must have shape (*batch, m, q_like). "
                f"Got shape={tuple(score.shape)}."
            )

        if self.n_w is None or self.n_w <= 1:
            return score.transpose(-1, -2)

        m = score.shape[-2]
        q_expanded = score.shape[-1]

        if q_expanded % int(self.n_w) != 0:
            raise RuntimeError(
                "score.shape[-1] must be divisible by n_w. "
                f"Got score.shape={tuple(score.shape)}, n_w={self.n_w}."
            )

        q = q_expanded // int(self.n_w)

        score_w = score.reshape(*score.shape[:-2], m, q, int(self.n_w))
        score_mq = _aggregate_score_w(
            score_w,
            n_w=int(self.n_w),
            risk_type=self.risk_type,
            alpha=self.alpha,
            risk_dim=-1,
            maximize=True,
        )

        return score_mq.transpose(-1, -2)

    def forward(self, score: Tensor, X: Optional[Tensor] = None) -> Tensor:
        if not torch.is_tensor(score):
            raise TypeError(f"score must be a Tensor. Got {type(score)}.")

        mode = self._infer_score_shape_mode(score, X)

        if mode == "aggregated":
            return self._handle_aggregated_score(score)

        if mode == "pointwise":
            return self._aggregate_pointwise_score(score)

        if mode == "multioutput_qm":
            return self._aggregate_multioutput_qm_score(score)

        if mode == "multioutput_mq":
            return self._aggregate_multioutput_mq_score(score)

        raise RuntimeError(f"Unsupported inferred score shape mode: {mode}")


# ============================================================
# 3. Multi-output classification input perturbation objective
#    posterior samples -> q / n aggregation for qEHVI / qNEHVI / qNParEGO
# ============================================================


class MultiOutputBinaryClassificationInputPerturbationObjective(MCMultiOutputObjective):
    """multi-output classification 用 objective。
    
    Args:
        n_w: InputPerturbation で 1 点あたりに展開される摂動数。
        risk_type: InputPerturbation 集約の risk 種類。`None`、`var`、`cvar`。
        alpha: risk 集約または qNEHVI の近似設定に使うパラメータ。
        aggregate_mean_when_no_risk: この acquisition / objective の動作を制御するパラメータ。
        allow_unexpanded: この acquisition / objective の動作を制御するパラメータ。
    
    Returns:
        Tensor: 入力 samples または score を変換・集約した objective value。
    
    Notes:
        posterior samples ではなく、計算済み acquisition score に作用する objective と、qEHVI などの samples objective を区別して使ってください。
    """

    def __init__(
        self,
        n_w: Optional[int] = None,
        risk_type: RiskType = None,
        alpha: float = 0.5,
        aggregate_mean_when_no_risk: bool = True,
        allow_unexpanded: bool = True,
    ) -> None:
        super().__init__()

        self.n_w = None if n_w is None else int(n_w)
        self.risk_type = risk_type
        self.alpha = float(alpha)
        self.aggregate_mean_when_no_risk = bool(aggregate_mean_when_no_risk)
        self.allow_unexpanded = bool(allow_unexpanded)

        _validate_n_w_risk(
            n_w=self.n_w,
            risk_type=self.risk_type,
            alpha=self.alpha,
        )

    def _aggregate_risk_axis(self, samples_w: Tensor) -> Tensor:
        return _aggregate_score_w(
            samples_w,
            n_w=int(self.n_w),
            risk_type=self.risk_type,
            alpha=self.alpha,
            risk_dim=-2,
            maximize=True,
        )

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        if not torch.is_tensor(samples):
            raise TypeError(f"samples must be a Tensor. Got {type(samples)}.")

        if samples.ndim < 2:
            raise RuntimeError(
                "samples must have at least shape (..., q_like, m). "
                f"Got shape={tuple(samples.shape)}."
            )

        if self.n_w is None or self.n_w <= 1:
            return samples

        if self.risk_type is None and not self.aggregate_mean_when_no_risk:
            return samples

        n_w = int(self.n_w)
        m = samples.shape[-1]

        # Baseline: X.shape = (n, d)
        if X is not None and X.ndim == 2:
            n = X.shape[-2]

            if samples.shape[-2] == n:
                return samples

            if samples.ndim >= 3 and samples.shape[-3] == n and samples.shape[-2] == n_w:
                return self._aggregate_risk_axis(samples)

            q_like = samples.shape[-2]

            if q_like == n * n_w:
                samples_w = samples.reshape(*samples.shape[:-2], n, n_w, m)
                return self._aggregate_risk_axis(samples_w)

            if self.allow_unexpanded:
                return samples

            raise RuntimeError(
                "Could not aggregate qNEHVI baseline samples. "
                f"samples.shape={tuple(samples.shape)}, "
                f"X.shape={tuple(X.shape)}, n_w={n_w}."
            )

        # Candidate: X.shape = (*batch, q, d)
        if X is not None and X.ndim >= 3:
            q = X.shape[-2]
            q_like = samples.shape[-2]

            if q_like == q:
                return samples

            if q_like == q * n_w:
                samples_w = samples.reshape(*samples.shape[:-2], q, n_w, m)
                return self._aggregate_risk_axis(samples_w)

            if self.allow_unexpanded:
                return samples

            raise RuntimeError(
                "Could not aggregate candidate samples. "
                f"samples.shape={tuple(samples.shape)}, "
                f"X.shape={tuple(X.shape)}, n_w={n_w}."
            )

        # X is None fallback
        q_expanded = samples.shape[-2]

        if q_expanded % n_w != 0:
            if self.allow_unexpanded:
                return samples

            raise RuntimeError(
                "samples.shape[-2] must be divisible by n_w for "
                "InputPerturbation aggregation. "
                f"Got samples.shape={tuple(samples.shape)}, n_w={n_w}."
            )

        q = q_expanded // n_w
        samples_w = samples.reshape(*samples.shape[:-2], q, n_w, m)

        return self._aggregate_risk_axis(samples_w)


# ============================================================
# 4. Objective mixins
# ============================================================


class BinaryClassificationScoreObjectiveMixin:
    """classification 用 objective。
    
    Returns:
        Tensor: 入力 samples または score を変換・集約した objective value。
    
    Notes:
        posterior samples ではなく、計算済み acquisition score に作用する objective と、qEHVI などの samples objective を区別して使ってください。
    """

    objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]]

    def _set_classification_score_objective(
        self,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        self.objective = objective

    def _apply_objective_to_score(
        self,
        score: Tensor,
        X: Optional[Tensor],
        name: str,
    ) -> Tensor:
        if self.objective is None:
            return score

        out = self.objective(score, X=X)

        if not torch.is_tensor(out):
            raise TypeError(f"{name}: objective must return a Tensor. Got {type(out)}.")

        return out


class MultiOutputBinaryClassificationScoreObjectiveMixin:
    """multi-output classification 用 objective。
    
    Returns:
        Tensor: 入力 samples または score を変換・集約した objective value。
    
    Notes:
        posterior samples ではなく、計算済み acquisition score に作用する objective と、qEHVI などの samples objective を区別して使ってください。
    """

    objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]]

    def _set_multioutput_classification_score_objective(
        self,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        self.objective = objective

    def _apply_objective_to_multioutput_score(
        self,
        score: Tensor,
        X: Optional[Tensor],
        name: str,
    ) -> Tensor:
        if self.objective is None:
            return score

        out = self.objective(score, X=X)

        if not torch.is_tensor(out):
            raise TypeError(f"{name}: objective must return a Tensor. Got {type(out)}.")

        return out

__all__ = [
    "BinaryClassificationScoreObjective",
    "MultiOutputBinaryClassificationScoreObjective",
    "MultiOutputBinaryClassificationInputPerturbationObjective",
    "BinaryClassificationScoreObjectiveMixin",
    "MultiOutputBinaryClassificationScoreObjectiveMixin",
]
