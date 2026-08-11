"""Model-aware helpers for BoTorch noisy hypervolume acquisitions."""

from __future__ import annotations

from typing import Any


def _model_supports_cache_root(model: Any) -> bool:
    """Return whether the model's public posterior supports cached Cholesky."""

    explicit = getattr(model, "_supports_cache_root", None)
    if explicit is not None:
        return bool(explicit)

    # Wide-format adapters expose a shape-preserving posterior view without the
    # raw distribution object required by BoTorch's cached-Cholesky mixin.
    return not callable(getattr(model, "_wrap_wide_posterior", None))


def resolve_nehvi_cache_root(model: Any, cache_root: bool | None = None) -> bool:
    """Resolve qNEHVI root caching from caller input and model capability.

    The helper is intentionally pure: acquisition classes call it explicitly in
    their constructors rather than having their ``__init__`` methods patched at
    import time.
    """

    if cache_root is not None:
        return bool(cache_root)
    return _model_supports_cache_root(model)


__all__ = ["resolve_nehvi_cache_root"]
