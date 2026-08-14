"""Temporary runtime adapters required by the Web application.

Keep import-time compatibility and routing patches centralized here instead of
spreading installer calls across :mod:`bochan.serving.webapp`. These adapters
should be removed individually as their behavior is absorbed into the normal
Web/visualization implementation.

Unlike dependency-specific compatibility such as the PFNs4BO checkpoint loader,
these adapters patch bochan runtime objects and therefore remain explicit
technical-debt boundaries.
"""

from __future__ import annotations

from .hybrid_bo_routing import install_web_hybrid_objective_bo_routing


def install_web_runtime_adapters() -> None:
    """Install the remaining Web runtime adapters."""

    install_web_hybrid_objective_bo_routing()


__all__ = ["install_web_runtime_adapters"]
