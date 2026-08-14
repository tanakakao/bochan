"""Architecture checks for the final Web/FastAPI serving composition."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from bochan.serving.fastapi.app import create_app as create_core_app
from bochan.serving.webapp.app import create_app as create_web_app

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
SERVING_ROOT = SRC_ROOT / "bochan/serving"
WEBAPP_ROOT = SERVING_ROOT / "webapp"
FASTAPI_ROOT = SERVING_ROOT / "fastapi"
_HTTP_METHODS = {"get", "put", "post", "delete", "options", "head", "patch", "trace"}
_ALLOWED_WEBAPP_FASTAPI_IMPORTS = {
    "bochan.serving.fastapi",
    "bochan.serving.fastapi.converters",
    "bochan.serving.fastapi.schemas.tabular",
}


def _openapi_route_keys(app) -> set[tuple[str, str]]:
    routes: set[tuple[str, str]] = set()
    for path, operations in app.openapi()["paths"].items():
        for method in operations:
            if method.lower() in _HTTP_METHODS:
                routes.add((method.upper(), str(path)))
    return routes


def _resolve_relative_import(path: Path, level: int, module: str | None) -> str | None:
    if not path.is_relative_to(SRC_ROOT):
        return None
    package_parts = list(path.relative_to(SRC_ROOT).with_suffix("").parts[:-1])
    parent_hops = max(level - 1, 0)
    if parent_hops > len(package_parts):
        return None
    if parent_hops:
        package_parts = package_parts[:-parent_hops]
    if module:
        package_parts.extend(module.split("."))
    return ".".join(package_parts)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level == 0:
            if node.module:
                imports.add(node.module)
            continue
        resolved = _resolve_relative_import(path, node.level, node.module)
        if resolved is None:
            imports.add("." * node.level + (node.module or ""))
        elif node.module:
            imports.add(resolved)
        else:
            imports.update(f"{resolved}.{alias.name}" for alias in node.names)
    return imports


def test_webapp_composes_core_and_web_routes_without_collisions() -> None:
    core_routes = _openapi_route_keys(create_core_app())
    web_only_routes = _openapi_route_keys(create_web_app(include_core_api=False))
    combined_routes = _openapi_route_keys(create_web_app())

    assert core_routes.isdisjoint(web_only_routes)
    assert combined_routes == core_routes | web_only_routes
    assert ("GET", "/api/v1/health") in combined_routes
    assert ("GET", "/api/v1/capabilities") in combined_routes


def test_webapp_can_run_without_core_api() -> None:
    core_routes = _openapi_route_keys(create_core_app())
    web_only_routes = _openapi_route_keys(create_web_app(include_core_api=False))

    assert core_routes.isdisjoint(web_only_routes)
    assert ("GET", "/api/v1/health") not in web_only_routes
    assert ("GET", "/api/v1/capabilities") in web_only_routes


def test_webapp_custom_prefix_covers_core_and_web_routes() -> None:
    routes = _openapi_route_keys(create_web_app(api_prefix="/bochan-test"))

    assert ("GET", "/bochan-test/health") in routes
    assert ("GET", "/bochan-test/capabilities") in routes
    assert all(path.startswith("/bochan-test/") for _, path in routes)


def test_fastapi_and_webapp_dependency_direction_is_one_way() -> None:
    reverse_dependencies = [
        path.relative_to(ROOT).as_posix()
        for path in FASTAPI_ROOT.rglob("*.py")
        if any(
            imported == "bochan.serving.webapp"
            or imported.startswith("bochan.serving.webapp.")
            for imported in _imports(path)
        )
    ]
    webapp_fastapi_imports = {
        imported
        for path in WEBAPP_ROOT.rglob("*.py")
        for imported in _imports(path)
        if imported == "bochan.serving.fastapi"
        or imported.startswith("bochan.serving.fastapi.")
    }

    assert reverse_dependencies == []
    assert webapp_fastapi_imports <= _ALLOWED_WEBAPP_FASTAPI_IMPORTS
    assert "bochan.serving.fastapi" in _imports(WEBAPP_ROOT / "app.py")
    assert "bochan.serving.workbench.datasets" in _imports(WEBAPP_ROOT / "app.py")


def test_frontend_launcher_and_backend_share_default_api_contract() -> None:
    backend_prefix = inspect.signature(create_web_app).parameters["api_prefix"].default
    frontend = (ROOT / "web/src/api.ts").read_text(encoding="utf-8")
    vite = (ROOT / "web/vite.config.ts").read_text(encoding="utf-8")
    launcher = (ROOT / "start_web.bat").read_text(encoding="utf-8")

    assert backend_prefix == "/api/v1"
    assert 'VITE_API_BASE ?? "/api/v1"' in frontend
    assert '"/api": "http://127.0.0.1:8001"' in vite
    assert '"/health":' not in vite
    assert '"/models":' not in vite
    assert '"/acquisitions":' not in vite
    assert "BACKEND_PORT=8001" in launcher
    assert "HEALTH_URL=http://%BACKEND_HOST%:%BACKEND_PORT%/api/v1/health" in launcher
    assert "bochan.serving.webapp.app:app" in launcher


def test_removed_webapp_modules_stay_removed_and_unimported() -> None:
    removed_modules = {
        "composition_support",
        "composition_visualization",
        "composition_visualization_dispatch",
        "level_set_settings",
        "risk_settings",
        "search_settings",
        "target_settings",
        "model_runtime",
        "model_reuse",
        "model_artifacts",
        "model_artifact_support",
        "workflows_extended",
        "_visualization_sessions_core",
    }
    removed_paths = [WEBAPP_ROOT / f"{name}.py" for name in removed_modules]
    forbidden_imports = {
        f"bochan.serving.webapp.{name}" for name in removed_modules
    }

    stale_imports: list[tuple[str, str]] = []
    for base in (SRC_ROOT, ROOT / "tests"):
        for path in base.rglob("*.py"):
            for imported in _imports(path):
                if imported in forbidden_imports:
                    stale_imports.append((path.relative_to(ROOT).as_posix(), imported))

    assert [path.name for path in removed_paths if path.exists()] == []
    assert stale_imports == []


def test_webapp_ci_path_filters_reference_existing_files() -> None:
    missing: list[tuple[str, str]] = []
    for workflow in (ROOT / ".github/workflows").glob("*.yml"):
        source = workflow.read_text(encoding="utf-8")
        paths = re.findall(
            r'^\s*-\s+["\']([^"\']+)["\']\s*$',
            source,
            flags=re.MULTILINE,
        )
        for raw_path in paths:
            if not raw_path.startswith("src/bochan/serving/webapp/"):
                continue
            if any(char in raw_path for char in "*?["):
                continue
            if not (ROOT / raw_path).exists():
                missing.append((workflow.name, raw_path))

    assert missing == []
