from __future__ import annotations

from pathlib import Path

import bochan.serving.fastapi as fastapi_package
import bochan.serving.fastapi.services as services
from bochan.serving.fastapi import router
from bochan.serving.fastapi.services import candidates, studies, study_results, tabular

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


def test_fastapi_services_package_does_not_forward_service_functions() -> None:
    forwarded = {
        "build_fit_response",
        "candidate_response",
        "compare_candidate_results",
        "compute_feature_importance_response",
        "fit_tabular_optimizer",
        "generate_candidate_result",
        "predict_response",
        "to_dataframe",
    }
    assert forwarded.isdisjoint(vars(services))
    assert services.__all__ == []
    assert router.__all__ == []


def test_fastapi_service_functions_are_owned_by_concrete_modules() -> None:
    assert candidates.generate_candidate_result.__module__ == (
        "bochan.serving.fastapi.services.candidates"
    )
    assert studies.build_study.__module__ == "bochan.serving.fastapi.services.studies"
    assert study_results.history_records.__module__ == (
        "bochan.serving.fastapi.services.study_results"
    )
    assert tabular.fit_tabular_optimizer.__module__ == (
        "bochan.serving.fastapi.services.tabular"
    )
