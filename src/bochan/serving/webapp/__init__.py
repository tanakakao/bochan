"""FastAPI application used by the React web interface."""

from typing import Any

from .app import WEB_CAPABILITIES, app, create_app as _create_app
from .composition_web_routes import register_composition_routes

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
