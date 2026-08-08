"""Compatibility helpers for loading legacy PFNs4BO checkpoints."""

from __future__ import annotations

from typing import Optional

from torch import Tensor, nn


def apply_pfns4bo_torch_compat() -> None:
    """Restore PyTorch transformer aliases imported by PFNs4BO 0.1.5.

    PFNs4BO's custom encoder layer imported typing/module aliases from the
    private ``torch.nn.modules.transformer`` namespace. Recent PyTorch releases
    no longer re-export those aliases, although the public equivalents still
    exist. The upstream checkpoint needs the legacy module to import during
    unpickling, so provide only the missing names and leave existing PyTorch
    symbols untouched.
    """
    from torch.nn.modules import transformer

    aliases = {
        "Module": nn.Module,
        "Tensor": Tensor,
        "Optional": Optional,
        "MultiheadAttention": nn.MultiheadAttention,
        "Linear": nn.Linear,
        "Dropout": nn.Dropout,
        "LayerNorm": nn.LayerNorm,
    }
    for name, value in aliases.items():
        if not hasattr(transformer, name):
            setattr(transformer, name, value)


__all__ = ["apply_pfns4bo_torch_compat"]
