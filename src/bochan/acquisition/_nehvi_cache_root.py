"""Compatibility helpers for BoTorch noisy hypervolume acquisitions.

BoTorch's cached-Cholesky qNEHVI path assumes direct access to a posterior
``distribution`` and a covariance layout that can be converted to independent
output batches. Correlated Kronecker posteriors and bochan's wide posterior
adapters do not satisfy those assumptions. This module makes custom qNEHVI
wrappers respect model capability whenever ``cache_root`` is not explicitly
supplied by the caller.
"""

# ruff: noqa: I001

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, TypeVar


AcquisitionType = TypeVar("AcquisitionType", bound=type)


def _model_supports_cache_root(model: Any) -> bool:
    """Return whether the model's public posterior supports cached Cholesky."""

    explicit = getattr(model, "_supports_cache_root", None)
    if explicit is not None:
        return bool(explicit)

    # Wide-format adapters wrap the base posterior in a shape-preserving view.
    # The view intentionally exposes moments and sampling, but not the raw
    # ``distribution`` object required by BoTorch's cached-Cholesky mixin.
    return not callable(getattr(model, "_wrap_wide_posterior", None))


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
    return _model_supports_cache_root(model)


def _expose_x_baseline_signature(acquisition_cls: AcquisitionType) -> None:
    """Expose ``X_baseline`` to API signature-based context filtering.

    Several bochan qNEHVI wrappers accept acquisition-specific arguments only
    through ``*args`` / ``**kwargs``. The high-level API filters automatically
    supplied context fields by constructor signature, so a hidden
    ``X_baseline`` would otherwise be removed even though the wrapper forwards it
    to BoTorch's qNEHVI implementation.
    """
    try:
        signature = inspect.signature(acquisition_cls.__init__, follow_wrapped=False)
    except (TypeError, ValueError):
        return

    if "X_baseline" in signature.parameters:
        return

    parameters = list(signature.parameters.values())
    insert_at = next(
        (
            index
            for index, parameter in enumerate(parameters)
            if parameter.kind == inspect.Parameter.VAR_KEYWORD
        ),
        len(parameters),
    )
    parameters.insert(
        insert_at,
        inspect.Parameter(
            "X_baseline",
            kind=inspect.Parameter.KEYWORD_ONLY,
            default=None,
        ),
    )
    acquisition_cls.__signature__ = signature.replace(parameters=parameters[1:])


def patch_nehvi_cache_root_init(acquisition_cls: AcquisitionType) -> AcquisitionType:
    """Patch a qNEHVI-style class to use a model-aware ``cache_root`` default.

    The patch is idempotent and preserves the original constructor metadata via
    ``functools.wraps``. It is used at package import time so both package-level
    imports and direct ``...multi_output`` imports receive the same behavior.
    The public class signature additionally exposes ``X_baseline`` so the
    high-level API can supply its default ``train_X`` baseline.
    """
    if getattr(acquisition_cls, "_bochan_cache_root_compat_patched", False):
        _expose_x_baseline_signature(acquisition_cls)
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
    _expose_x_baseline_signature(acquisition_cls)
    return acquisition_cls


__all__ = [
    "patch_nehvi_cache_root_init",
    "resolve_nehvi_cache_root",
]
