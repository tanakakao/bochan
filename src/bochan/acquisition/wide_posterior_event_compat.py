"""Posterior event-shape compatibility for wide multi-task adapters."""

from __future__ import annotations

import torch

from bochan.models.wide_multitask import _WidePosterior


_APPLIED = False


def apply_wide_posterior_event_compat() -> None:
    """Expose public q/output event axes to posterior wrappers.

    Binary epistemic probability posteriors delegate ``event_shape`` to their
    latent posterior. The wide posterior stores a flattened base distribution but
    presents public samples as ``[..., q, m]`` (or ``[..., q, m, C]`` for
    multiclass), so report those public event axes.
    """

    global _APPLIED
    if _APPLIED:
        return

    def event_shape(self: _WidePosterior) -> torch.Size:
        mean = self.mean
        trailing = 2 if self.scalar_task_values else 3
        return torch.Size(mean.shape[-trailing:])

    _WidePosterior.event_shape = property(event_shape)  # type: ignore[attr-defined]
    _APPLIED = True


__all__ = ["apply_wide_posterior_event_compat"]
