"""Score-objective application and q reduction."""

from __future__ import annotations

import torch
from torch import Tensor

from ._base_common import (
    ReductionType,
    _ensure_q_batch,
    _is_mc_multi_output_objective,
    _looks_like_score_objective,
    _objective_call,
    _reduce,
    _safe_prod,
)


class _RegressionObjectiveMixin:
    # ------------------------------------------------------------
    # Objective / q reduction
    # ------------------------------------------------------------
    def _apply_objective_to_score(
        self,
        score: Tensor,
        *,
        raw_X: Tensor,
        expanded_X: Tensor,
        name: str,
    ) -> Tensor:
        objective = self.objective
        if objective is None:
            return score

        # Classification / ordinal style score objective.
        if _looks_like_score_objective(objective):
            out = _objective_call(objective, score, raw_X)
            if not torch.is_tensor(out):
                raise TypeError(f"{name}: objective must return Tensor. Got {type(out)}.")
            return out

        # BoTorch MC objective / risk measure style.  Treat score as deterministic samples.
        if _is_mc_multi_output_objective(objective):
            pseudo = score
            if pseudo.ndim == expanded_X.ndim - 1:
                pseudo = pseudo.unsqueeze(-1)
            pseudo = pseudo.unsqueeze(0)
            out = _objective_call(objective, pseudo, raw_X)
            if not torch.is_tensor(out):
                raise TypeError(f"{name}: objective must return Tensor. Got {type(out)}.")
            if out.ndim >= 1 and out.shape[0] == 1:
                out = out.squeeze(0)
            if out.ndim == raw_X.ndim and out.shape[-1] == 1:
                out = out.squeeze(-1)
            return out

        # Generic callable: try score-objective style first, then pseudo-sample style.
        try:
            out = _objective_call(objective, score, raw_X)
            if torch.is_tensor(out):
                return out
        except Exception:
            pass

        pseudo = score
        if pseudo.ndim == expanded_X.ndim - 1:
            pseudo = pseudo.unsqueeze(-1)
        pseudo = pseudo.unsqueeze(0)
        out = _objective_call(objective, pseudo, raw_X)
        if not torch.is_tensor(out):
            raise TypeError(f"{name}: objective must return Tensor. Got {type(out)}.")
        if out.ndim >= 1 and out.shape[0] == 1:
            out = out.squeeze(0)
        if out.ndim == raw_X.ndim and out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out

    def _aggregate_n_w_if_needed(
        self,
        score: Tensor,
        *,
        q: int,
        context: str,
    ) -> Tensor:
        if self.n_w is None:
            return score

        expected = q * int(self.n_w)
        if score.shape[-1] == q:
            return score
        if score.shape[-1] != expected:
            raise RuntimeError(
                f"{context}: expected last dimension q={q} or q*n_w={expected}, "
                f"got score.shape={tuple(score.shape)}."
            )

        return score.reshape(*score.shape[:-1], q, int(self.n_w)).mean(dim=-1)

    def _reduce_q(self, score: Tensor) -> Tensor:
        return _reduce(score, dim=-1, mode=self.reduction)

    def _finalize_pointwise_score(
        self,
        score: Tensor,
        X: Tensor,
        Xt: Tensor,
        *,
        name: str,
    ) -> Tensor:
        raw_X = _ensure_q_batch(X)
        original_batch_shape = torch.Size(raw_X.shape[:-2])
        q = int(raw_X.shape[-2])

        score = self._align_pointwise_score_to_X(score, Xt, name=f"{name} score before penalty")
        score = score - self._total_penalty_per_point(Xt)

        score = self._align_pointwise_score_to_X(score, Xt, name=f"{name} score before objective")
        score = self._apply_objective_to_score(
            score,
            raw_X=raw_X,
            expanded_X=Xt,
            name=name,
        )

        score = self._aggregate_n_w_if_needed(score, q=q, context=name)

        out = self._reduce_q(score)

        if out.shape == original_batch_shape:
            return out

        while out.ndim > len(original_batch_shape):
            out = out.mean(dim=0)
            if out.shape == original_batch_shape:
                return out

        if out.shape == original_batch_shape:
            return out

        if out.numel() == _safe_prod(original_batch_shape):
            return out.reshape(original_batch_shape)

        raise RuntimeError(
            f"{name}: output shape mismatch. "
            f"Expected {tuple(original_batch_shape)}, got {tuple(out.shape)}."
        )


# ============================================================
# Pointwise active-learning acquisitions
# ============================================================
# Keep this file focused on active learning / uncertainty reduction.
# Boundary / contour / straddle acquisitions are implemented in
# regression_levelset_estimation_aligned.py.

