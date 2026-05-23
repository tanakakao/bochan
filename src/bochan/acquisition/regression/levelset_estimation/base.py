from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from botorch.acquisition.acquisition import AcquisitionFunction


def safe_logdet(covar: Tensor, jitter: float = 1e-6) -> Tensor:
    """Numerically stable log determinant.

    Args:
        covar: Covariance matrix with shape `batch_shape x n x n`.
        jitter: Diagonal jitter.

    Returns:
        Tensor: Log determinant with shape `batch_shape`.
    """
    n = covar.shape[-1]
    eye = torch.eye(n, device=covar.device, dtype=covar.dtype)
    mat = covar + jitter * eye
    sign, logabsdet = torch.linalg.slogdet(mat)
    if not torch.all(sign > 0):
        mat = covar + (10.0 * jitter) * eye
        sign, logabsdet = torch.linalg.slogdet(mat)
    return logabsdet


class BasePendingPenaltyAcquisition(AcquisitionFunction):
    """Regression level-set acquisition base with pending penalty.

    This base keeps `X_pending` in raw input space and transforms it at
    distance-computation time so candidate and pending points are compared in
    the same feature space.

    Args:
        model: BoTorch-compatible model.
        penalty_scale: Pending penalty strength. This is used as the exponential
            distance-decay scale in `_apply_pending_penalty`.
    """

    def __init__(self, model, penalty_scale: float = 20.0) -> None:
        super().__init__(model=model)
        self.penalty_scale = float(penalty_scale)
        self.X_pending: Optional[Tensor] = None

    def _ensure_q_batch(self, X: Tensor) -> Tensor:
        return X.unsqueeze(-2) if X.ndim == 2 else X

    def _coerce_pending_to_tensor(
        self,
        X_pending,
        *,
        ref: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """Normalize X_pending to Tensor or None."""
        if X_pending is None:
            return None
        if torch.is_tensor(X_pending):
            out = X_pending
        elif isinstance(X_pending, (list, tuple)):
            tensors = []
            for item in X_pending:
                if item is None:
                    continue
                t = self._coerce_pending_to_tensor(item, ref=ref)
                if t is not None and t.numel() > 0:
                    tensors.append(t)
            if len(tensors) == 0:
                return None
            if len(tensors) == 1:
                out = tensors[0]
            else:
                try:
                    out = torch.cat(tensors, dim=-2)
                except RuntimeError:
                    out = torch.cat([t.reshape(-1, t.shape[-1]) for t in tensors], dim=-2)
        else:
            raise TypeError(
                "X_pending must be None, Tensor, list, or tuple. "
                f"Got {type(X_pending)}."
            )
        if ref is not None:
            out = out.to(device=ref.device, dtype=ref.dtype)
        return out

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        """Store pending points in raw input space."""
        self.X_pending = self._coerce_pending_to_tensor(X_pending)

    def _apply_input_transform_for_distance(self, X: Tensor) -> Tensor:
        """Map X to the feature space used for distance penalty."""
        X = self._ensure_q_batch(X)

        it = getattr(self.model, "input_transform", None)
        if it is not None:
            Xt = it(X)
            if isinstance(Xt, tuple):
                Xt = Xt[0]
            return self._ensure_q_batch(Xt)

        models = getattr(self.model, "models", None)
        if models is not None and len(models) > 0:
            it = getattr(models[0], "input_transform", None)
            if it is not None:
                Xt = it(X)
                if isinstance(Xt, tuple):
                    Xt = Xt[0]
                return self._ensure_q_batch(Xt)

        return X

    def _transform_pending_like_candidate(
        self,
        X_pending,
        *,
        ref: Tensor,
    ) -> Optional[Tensor]:
        """Map raw-space X_pending to the same distance space as candidate."""
        Xp = self._coerce_pending_to_tensor(X_pending, ref=ref)
        if Xp is None or Xp.numel() == 0:
            return None
        Xp_t = self._apply_input_transform_for_distance(Xp)
        return self._ensure_q_batch(Xp_t).to(device=ref.device, dtype=ref.dtype)

    def _pending_penalty_per_point(self, Xt: Tensor) -> Tensor:
        """Return pointwise pending penalty with shape `batch_shape x q_like`."""
        Xt = self._ensure_q_batch(Xt)
        Xp_t = self._transform_pending_like_candidate(getattr(self, "X_pending", None), ref=Xt)
        if Xp_t is None or Xp_t.numel() == 0:
            return torch.zeros(Xt.shape[:-1], dtype=Xt.dtype, device=Xt.device)

        d = Xt.shape[-1]
        X2d = Xt.reshape(-1, d)
        Xp2d = Xp_t.reshape(-1, Xp_t.shape[-1])
        if Xp2d.shape[-1] != d:
            raise RuntimeError(
                "X_pending feature dimension mismatch after transform: "
                f"Xt.shape={tuple(Xt.shape)}, X_pending_transformed.shape={tuple(Xp_t.shape)}."
            )

        dist = torch.cdist(X2d, Xp2d).min(dim=-1).values.reshape(*Xt.shape[:-1])
        return torch.exp(-self.penalty_scale * dist)

    def _apply_pending_penalty(self, X: Tensor, score: Tensor) -> Tensor:
        """Subtract pending penalty from an already aggregated acquisition score.

        Args:
            X: Raw candidate tensor.
            score: Acquisition score with shape `batch_shape`.

        Returns:
            Tensor: Score after pending penalty.
        """
        if getattr(self, "X_pending", None) is None:
            return score

        Xt = self._apply_input_transform_for_distance(X)
        point_penalty = self._pending_penalty_per_point(Xt)
        if point_penalty.ndim >= 1:
            penalty = point_penalty.mean(dim=-1)
        else:
            penalty = point_penalty
        return score - penalty
