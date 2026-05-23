from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def _align_pointwise_score_to_X(
    score: Tensor,
    X: Tensor,
    *,
    name: str = "score",
    reduce_extra: str = "sum",
) -> Tensor:
    """
    Align a pointwise score tensor to X.shape[:-1].

    Expected X shape:
        [..., q, d]

    Supported score shapes:
        [..., q]
        [..., q, 1]
        [..., q, 2]
        [..., q * 2]

    The q * 2 / [..., q, 2] cases are useful when binary class dimensions
    accidentally remain in the score tensor.
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
                score = score.reshape(*target_shape, 2)
                if reduce_extra == "sum":
                    return score.sum(dim=-1)
                if reduce_extra == "mean":
                    return score.mean(dim=-1)
                if reduce_extra == "max":
                    return score.max(dim=-1).values
                raise ValueError(f"Unknown reduce_extra={reduce_extra!r}")

    raise RuntimeError(
        f"{name}: cannot align score to X. "
        f"score.shape={tuple(score.shape)}, "
        f"X.shape={tuple(X.shape)}, "
        f"target_shape={tuple(target_shape)}"
    )


def _is_classification_score_objective(objective) -> bool:
    """
    自作の ClassificationScoreObjective かどうかを判定する。

    ClassificationScoreObjective は score shape = (*batch, q_like)
    を期待するため、score.unsqueeze(-1) してはいけない。
    """
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


def _apply_objective_to_pointwise_score(
    acqf,
    score: Tensor,
    *,
    raw_X: Tensor,
    expanded_X: Tensor,
    name: str,
) -> Tensor:
    """
    pointwise classification acquisition score に objective を適用する。

    Args:
        score:
            expanded_X に対応する pointwise score.
            典型 shape: (*batch, q * n_w)

        raw_X:
            objective 適用後の q shape に対応する raw input.
            典型 shape: (*batch, q, d)

        expanded_X:
            input_transform 後の input.
            典型 shape: (*batch, q * n_w, d)

    Notes:
        - ClassificationScoreObjective は score = (*batch, q_like) を期待する。
        - BoTorch MCAcquisitionObjective / RiskMeasureMCObjective は
          samples = (*batch, q_like, m) を期待する。
    """
    objective = getattr(acqf, "objective", None)
    if objective is None:
        return score

    # ------------------------------------------------------------
    # 1. 自作 ClassificationScoreObjective:
    #    score は (*batch, q_like) のまま渡す。
    # ------------------------------------------------------------
    if _is_classification_score_objective(objective):
        try:
            out = objective(score, X=raw_X)
        except TypeError:
            out = objective(score)

        if not torch.is_tensor(out):
            raise TypeError(
                f"{name}: objective must return a Tensor. Got {type(out)}."
            )

        return out

    # ------------------------------------------------------------
    # 2. BoTorch MCAcquisitionObjective / RiskMeasureMCObjective:
    #    samples は (*batch, q_like, m) を想定するため m=1 を付ける。
    # ------------------------------------------------------------
    score_in = score

    # expanded_X: (*batch, q_like, d)
    # score:      (*batch, q_like)
    if score_in.ndim == expanded_X.ndim - 1:
        score_in = score_in.unsqueeze(-1)

    try:
        out = objective(score_in, X=raw_X)
    except RuntimeError as err:
        message = str(err)

        # BoTorch の shape check が邪魔する場合の保険。
        # one-to-many transform + risk aggregation では、
        # objective output q が raw_X 側に戻ることがある。
        if hasattr(objective, "_verify_output_shape"):
            old_verify = objective._verify_output_shape
            try:
                objective._verify_output_shape = False
                out = objective(score_in, X=raw_X)
            finally:
                objective._verify_output_shape = old_verify
        else:
            raise err
    except TypeError:
        out = objective(score_in)

    if not torch.is_tensor(out):
        raise TypeError(
            f"{name}: objective must return a Tensor. Got {type(out)}."
        )

    # (*batch, q, 1) -> (*batch, q)
    if out.ndim == raw_X.ndim and out.shape[-1] == 1:
        out = out.squeeze(-1)

    return out
