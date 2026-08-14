from __future__ import annotations

from pathlib import Path

import bochan.serving.fastapi as fastapi_package
from bochan.serving.fastapi import router
from bochan.serving.fastapi import services


_ALLOWED_ROOT_MODULES = {
    "__init__.py",
    "app.py",
    "converters.py",
    "dependencies.py",
    "router.py",
    "target_categories.py",
}
_ALLOWED_PACKAGES = {"routers", "schemas", "services", "stores"}


def test_fastapi_root_contains_only_transport_boundary_modules() -> None:
    package_dir = Path(fastapi_package.__file__).resolve().parent
    root_modules = {path.name for path in package_dir.glob("*.py")}
    packages = {
        path.name
        for path in package_dir.iterdir()
        if path.is_dir() and (path / "__init__.py").exists()
    }

    assert root_modules == _ALLOWED_ROOT_MODULES
    assert _ALLOWED_PACKAGES.issubset(packages)


def test_fastapi_package_boundaries_do_not_forward_service_functions() -> None:
    assert router.__all__ == []
    assert services.__all__ == []
