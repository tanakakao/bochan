from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor


def align_pointwise_score_to_X(
    score: Tensor,
    X: Tensor,
    *,
    name: str = "score",
    reduce_extra: str = "sum",
) -> Tensor:
    """pointwise score を X.shape[:-1] に揃える。

    InputPerturbation や binary class 方向が混ざる場合、
    score が [..., q, 1], [..., q, 2], [..., q * 2] になることがある。
    その場合は X.shape[:-1] = [..., q] に整形する。
    """
    target_shape = X.shape[:-1]

    if score.shape == target_shape:
        return score

    # [..., q, 1] -> [..., q]
    if score.ndim == len(target_shape) + 1:
        if score.shape[:-1] == target_shape and score.shape[-1] == 1:
            return score.squeeze(-1)

    # [..., q, 2] -> [..., q]
    if score.ndim == len(target_shape) + 1:
        if score.shape[:-1] == target_shape and score.shape[-1] == 2:
            if reduce_extra == "sum":
                return score.sum(dim=-1)
            if reduce_extra == "mean":
                return score.mean(dim=-1)
            if reduce_extra == "max":
                return score.max(dim=-1).values
            raise ValueError(f"Unknown reduce_extra={reduce_extra!r}")

    # [..., q * 2] -> [..., q, 2] -> [..., q]
    if score.ndim == len(target_shape):
        if score.shape[:-1] == target_shape[:-1]:
            q_target = target_shape[-1]
            q_score = score.shape[-1]
            if q_score == q_target * 2:
                score2 = score.reshape(*target_shape, 2)
                if reduce_extra == "sum":
                    return score2.sum(dim=-1)
                if reduce_extra == "mean":
                    return score2.mean(dim=-1)
                if reduce_extra == "max":
                    return score2.max(dim=-1).values
                raise ValueError(f"Unknown reduce_extra={reduce_extra!r}")

    target_numel = math.prod(target_shape)
    if score.numel() == target_numel:
        return score.reshape(*target_shape)

    raise RuntimeError(
        f"{name}: cannot align score to X. "
        f"score.shape={tuple(score.shape)}, "
        f"X.shape={tuple(X.shape)}, "
        f"target_shape={tuple(target_shape)}"
    )


def is_classification_score_objective(objective) -> bool:
    """自作 ClassificationScoreObjective かどうかを判定する。"""
    cls_name = objective.__class__.__name__
    module_name = objective.__class__.__module__

    return (
        cls_name == "ClassificationScoreObjective"
        or (
            "classification" in module_name
            and hasattr(objective, "n_w")
            and hasattr(objective, "risk_type")
        )
    )


def apply_classification_objective_to_score(
    acqf,
    score: Tensor,
    X: Optional[Tensor],
    *,
    name: str,
) -> Tensor:
    """classification / level-set acquisition の score に objective を適用する。"""
    objective = getattr(acqf, "objective", None)
    if objective is None:
        return score

    # 自作 ClassificationScoreObjective は score = (*batch, q_like) のまま渡す。
    if is_classification_score_objective(objective):
        try:
            out = objective(score, X=X)
        except TypeError:
            out = objective(score)

        if not torch.is_tensor(out):
            raise TypeError(f"{name}: objective must return a Tensor. Got {type(out)}.")
        return out

    # BoTorch MCAcquisitionObjective / RiskMeasureMCObjective には m=1 を付ける。
    score_in = score
    if X is not None and score_in.ndim == X.ndim - 1:
        score_in = score_in.unsqueeze(-1)

    try:
        out = objective(score_in, X=X)
    except RuntimeError as err:
        if hasattr(objective, "_verify_output_shape"):
            old_verify = objective._verify_output_shape
            try:
                objective._verify_output_shape = False
                out = objective(score_in, X=X)
            finally:
                objective._verify_output_shape = old_verify
        else:
            raise err
    except TypeError:
        out = objective(score_in)

    if not torch.is_tensor(out):
        raise TypeError(f"{name}: objective must return a Tensor. Got {type(out)}.")

    if X is not None and out.ndim == X.ndim and out.shape[-1] == 1:
        out = out.squeeze(-1)

    return out


def normalize_pointwise_tensor_to_orig(
    t: Tensor,
    orig: torch.Size,
    *,
    name: str,
) -> Tensor:
    """posterior.mean / variance などを pointwise shape に正規化する。"""
    if t.ndim >= 1 and t.shape[-1] == 1:
        t = t.squeeze(-1)

    if tuple(t.shape) == tuple(orig):
        return t

    # DeepGP の sample / extra leading dims は平均化する。
    while t.ndim > len(orig) and tuple(t.shape[-len(orig):]) == tuple(orig):
        t = t.mean(dim=0)
        if tuple(t.shape) == tuple(orig):
            return t

    if t.numel() == math.prod(orig):
        return t.reshape(*orig)

    raise RuntimeError(
        f"Unexpected {name} shape: got {tuple(t.shape)}, "
        f"numel={t.numel()}, expected orig={tuple(orig)} "
        f"with numel={math.prod(orig)}."
    )


def bernoulli_entropy(p: Tensor, eps: float = 1e-12) -> Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log())


def boundary_kernel_weight(values: Tensor, target: float | Tensor, tau: float) -> Tensor:
    target_t = torch.as_tensor(target, device=values.device, dtype=values.dtype)
    tau_t = torch.as_tensor(tau, device=values.device, dtype=values.dtype).clamp_min(1e-8)
    return torch.exp(-0.5 * ((values - target_t) / tau_t).pow(2))
