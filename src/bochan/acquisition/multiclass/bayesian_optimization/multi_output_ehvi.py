from __future__ import annotations

import torch
from torch import Tensor

from .multi_output import qMultiOutputMulticlassExpectedHypervolumeImprovement as _BaseQEHVI


def _ensure_q_batch(X: Tensor) -> Tensor:
    if X.ndim == 1:
        return X.view(1, 1, -1)
    if X.ndim == 2:
        return X.unsqueeze(0)
    return X


def _prod(shape: torch.Size | tuple[int, ...]) -> int:
    out = 1
    for s in shape:
        out *= int(s)
    return out


def _match_pending_to_X(X_pending: Tensor, X: Tensor) -> Tensor:
    Xp = torch.as_tensor(X_pending, device=X.device, dtype=X.dtype).detach()
    if Xp.ndim == 1:
        Xp = Xp.view(1, 1, -1)
    elif Xp.ndim == 2:
        Xp = Xp.unsqueeze(0)
    if Xp.shape[:-2] != X.shape[:-2]:
        Xp = Xp.expand(*X.shape[:-2], *Xp.shape[-2:])
    return Xp


def _finalize_qehvi_output(value: Tensor, X: Tensor) -> Tensor:
    target_shape = tuple(X.shape[:-2])
    if value.shape == target_shape:
        return value
    if value.ndim == 0:
        return value.expand(*target_shape) if len(target_shape) > 0 else value

    # Current failure pattern: qEHVI returns [batch, 1] although BoTorch expects [batch].
    while value.ndim > len(target_shape) and value.shape[-1] == 1:
        value = value.squeeze(-1)
        if value.shape == target_shape:
            return value

    while value.ndim > len(target_shape):
        value = value.mean(dim=-1)
        if value.shape == target_shape:
            return value

    if value.numel() == _prod(target_shape):
        return value.reshape(target_shape)
    if len(target_shape) == 0 and value.numel() == 1:
        return value.reshape(target_shape)
    return value


class qMultiOutputMulticlassExpectedHypervolumeImprovement(_BaseQEHVI):
    """Shape-safe qEHVI wrapper for multiclass multi-output objectives.

    Some BoTorch versions return a trailing singleton shape ``batch_shape x 1``
    for qEHVI when the objective ultimately has one q-candidate after sequential
    optimization. The standard ``t_batch_mode_transform`` assertion expects just
    ``batch_shape``. This wrapper computes qEHVI directly and removes only those
    redundant singleton / sample-like dimensions.
    """

    def forward(self, X: Tensor) -> Tensor:
        Xq = _ensure_q_batch(X)
        X_pending = getattr(self, "X_pending", None)
        if X_pending is not None and torch.as_tensor(X_pending).numel() > 0:
            Xq = torch.cat([Xq, _match_pending_to_X(X_pending, Xq)], dim=-2)

        posterior = self.model.posterior(Xq)
        samples = self.get_posterior_samples(posterior)
        value = self._compute_qehvi(samples=samples, X=Xq)
        return _finalize_qehvi_output(value, Xq)


__all__ = ["qMultiOutputMulticlassExpectedHypervolumeImprovement"]
