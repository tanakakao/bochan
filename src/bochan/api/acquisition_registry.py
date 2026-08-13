"""Deprecated flat import facade for :mod:`bochan.api.registry.acquisition`."""

from __future__ import annotations

from .registry.acquisition import available_acqf_names, resolve_acqf_cls

__all__ = ["available_acqf_names", "resolve_acqf_cls"]
