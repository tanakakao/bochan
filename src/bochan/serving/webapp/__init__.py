"""FastAPI application used by the React web interface."""

from typing import Any

from .composition_web_support import install_composition_web_support

install_composition_web_support()

from .app import app, create_app as _create_app
from .composition_web_routes import register_composition_routes

register_composition_routes(app)


def create_app(*args: Any, **kwargs: Any) -> Any:
    """Create a Web FastAPI app including the composition validation route."""

    created = _create_app(*args, **kwargs)
    register_composition_routes(created)
    return created


__all__ = ["app", "create_app"]
