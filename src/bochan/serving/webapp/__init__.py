"""FastAPI application used by the React web interface."""

from typing import Any

from .composition_element_importance_figures import (
    install_composition_element_importance_figures,
)
from .composition_feature_importance import install_composition_feature_importance
from .composition_feature_importance_views import (
    install_composition_feature_importance_views,
)
from .composition_importance_output_compat import (
    install_composition_importance_output_compat,
)
from .composition_importance_records_compat import (
    install_composition_importance_records_compat,
)
from .composition_multielement_ternary import (
    install_composition_multielement_ternary,
)
from .composition_pd_compat import install_composition_pd_compat
from .composition_visualization import install_composition_visualization
from .composition_visualization_compat import (
    install_composition_visualization_compat,
)
from .hybrid_bo_routing import install_web_hybrid_objective_bo_routing
from .pandas_compat import install_pandas_string_category_compat
from .ternary_plot_grid_compat import install_ternary_plot_grid_compat
from .visualization_feature_types import install_visualization_feature_type_compat

# These Web-only visualization adapters remain outside the tabular optimizer
# path. Composition fitting/candidate handling itself is integrated directly in
# workflows_tabular.py and does not install runtime patches.
install_pandas_string_category_compat()
install_visualization_feature_type_compat()
install_ternary_plot_grid_compat()
install_composition_visualization()
install_composition_visualization_compat()
install_composition_multielement_ternary()
install_composition_pd_compat()
install_web_hybrid_objective_bo_routing()

from .app import WEB_CAPABILITIES, app, create_app as _create_app  # noqa: E402
from .composition_web_routes import register_composition_routes  # noqa: E402

# Composition feature-importance presentation is installed after app/session
# construction. These presentation adapters are independent of tabular fitting.
install_composition_importance_records_compat()
install_composition_feature_importance()
install_composition_feature_importance_views()
install_composition_importance_output_compat()
install_composition_element_importance_figures()

WEB_CAPABILITIES["composition"] = {
    "enabled": True,
    "max_formula_columns": 1,
    "sites": False,
    "ratio_total": 1.0,
    "representations": ["fractions", "clr", "alr", "ilr"],
    "normalizations": ["atomic_fraction", "weight_fraction"],
    "element_constraints": ["=", "<=", ">="],
    "visualization_axes": [
        "element_fraction_1d",
        "element_fraction_2d",
        "ternary",
    ],
    "feature_importance": ["composition_total", "element_perturbation"],
    "validation_endpoint": "/api/v1/composition/validate",
    "optimization_endpoint": "/api/v1/composition/regression/run",
}
register_composition_routes(app)


def create_app(*args: Any, **kwargs: Any) -> Any:
    """Create a Web FastAPI app including the composition validation route."""

    created = _create_app(*args, **kwargs)
    register_composition_routes(created)
    return created


__all__ = ["app", "create_app"]
