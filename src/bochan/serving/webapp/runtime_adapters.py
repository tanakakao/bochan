"""Temporary runtime adapters required by the Web application.

Keep import-time compatibility and routing patches centralized here instead of
spreading installer calls across :mod:`bochan.serving.webapp`.  These adapters
should be removed individually as their behavior is absorbed into the normal
Web/visualization implementation.

Unlike dependency-specific compatibility such as the PFNs4BO checkpoint loader,
these adapters patch bochan or third-party runtime objects and therefore remain
explicit technical-debt boundaries.
"""

from __future__ import annotations

from .composition_multielement_ternary import install_composition_multielement_ternary
from .composition_pd_compat import install_composition_pd_compat
from .composition_visualization import install_composition_visualization
from .composition_visualization_compat import install_composition_visualization_compat
from .hybrid_bo_routing import install_web_hybrid_objective_bo_routing
from .ternary_plot_grid_compat import install_ternary_plot_grid_compat
from .visualization_feature_types import install_visualization_feature_type_compat


def install_web_runtime_adapters() -> None:
    """Install the remaining Web runtime adapters in their required order."""

    install_visualization_feature_type_compat()
    install_ternary_plot_grid_compat()
    install_composition_visualization()
    install_composition_visualization_compat()
    install_composition_multielement_ternary()
    install_composition_pd_compat()
    install_web_hybrid_objective_bo_routing()


__all__ = ["install_web_runtime_adapters"]
