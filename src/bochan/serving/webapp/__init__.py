"""FastAPI application used by the React web interface."""

from typing import Any

from .composition_constraint_adapter import install_composition_constraint_adapter
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
from .composition_visualization import install_composition_visualization
from .composition_web_support import install_composition_web_support
from .pandas_compat import install_pandas_string_category_compat
from .visualization_feature_types import install_visualization_feature_type_compat

# Install before importing app.py and workflow modules so runtime DataFrame
# conversions use the Pandas StringDtype compatibility wrapper.
install_pandas_string_category_compat()
install_visualization_feature_type_compat()
install_composition_visualization()

# The adapters must be installed before app.py imports workflows.py and binds the
# composition-unaware workflow function. Keep these imports intentionally delayed.
install_composition_web_support()
install_composition_constraint_adapter()

from . import workflows_tabular as _workflows_tabular  # noqa: E402

# Keep the established internal contract used by the Web workflow wrapper and
# artifact tests even though the callable is composition-aware.
_workflows_tabular.run_regression_web_workflow.__module__ = _workflows_tabular.__name__

# Composition feature importance requires the completed visualization session.
# Attach the raw importance payload, replace coordinate-level PI by the joint
# ``組成全体`` entry for supported regression outputs, preserve other outputs,
# then add the element-wise composition card.
install_composition_feature_importance()
install_composition_feature_importance_views()
install_composition_importance_output_compat()
install_composition_element_importance_figures()

from .app import WEB_CAPABILITIES, app, create_app as _create_app  # noqa: E402
from .composition_web_routes import register_composition_routes  # noqa: E402

WEB_CAPABILITIES["composition"] = {
    "enabled": True,
    "max_formula_columns": 1,
    "sites": False,
    "ratio_total": 1.0,
    "representations": ["fractions", "clr", "alr", "ilr"],
    "normalizations": ["atomic_fraction", "weight_fraction"],
    "element_constraints": ["=", "<=", ">="],
    "visualization_axes": ["element_fraction_1d", "element_fraction_2d", "ternary"],
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
