"""Compatibility helpers for BoTorch noisy hypervolume acquisitions.

BoTorch's cached-Cholesky qNEHVI path assumes that a multitask posterior can be
converted to independent output batches. Correlated Kronecker posteriors do not
satisfy that assumption, so their models expose ``_supports_cache_root=False``.
This module makes custom qNEHVI wrappers respect that capability flag whenever
``cache_root`` is not explicitly supplied by the caller.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, TypeVar


AcquisitionType = TypeVar("AcquisitionType", bound=type)


def resolve_nehvi_cache_root(model: Any, cache_root: bool | None = None) -> bool:
    """Resolve qNEHVI root caching from an explicit value or model capability.

    Args:
        model: Surrogate model passed to the acquisition function.
        cache_root: Explicit caller setting. ``True`` and ``False`` are preserved;
            ``None`` selects the model-aware default.

    Returns:
        Whether BoTorch's cached root-decomposition path should be used.
    """
    if cache_root is not None:
        return bool(cache_root)
    return bool(getattr(model, "_supports_cache_root", True))


def patch_nehvi_cache_root_init(acquisition_cls: AcquisitionType) -> AcquisitionType:
    """Patch a qNEHVI-style class to use a model-aware ``cache_root`` default.

    The patch is idempotent and preserves the original constructor metadata via
    ``functools.wraps``. It is used at package import time so both package-level
    imports and direct ``...multi_output`` imports receive the same behavior.
    """
    if getattr(acquisition_cls, "_bochan_cache_root_compat_patched", False):
        return acquisition_cls

    original_init = acquisition_cls.__init__

    @wraps(original_init)
    def model_aware_init(self, model, *args, **kwargs):
        kwargs["cache_root"] = resolve_nehvi_cache_root(
            model,
            kwargs.get("cache_root"),
        )
        original_init(self, model, *args, **kwargs)

    acquisition_cls.__init__ = model_aware_init
    acquisition_cls._bochan_cache_root_compat_patched = True
    acquisition_cls._bochan_original_init = original_init
    return acquisition_cls


__all__ = [
    "patch_nehvi_cache_root_init",
    "resolve_nehvi_cache_root",
]
