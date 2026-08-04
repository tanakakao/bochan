"""FastAPI application used by the React web interface."""

from .composition_web_support import install_composition_web_support

install_composition_web_support()

from .app import app, create_app
from .composition_web_routes import register_composition_routes

register_composition_routes(app)

__all__ = ["app", "create_app"]
