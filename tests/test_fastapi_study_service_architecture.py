from __future__ import annotations

import importlib.util

from bochan.serving.fastapi.services import studies, study_results


def test_fastapi_study_services_have_canonical_owners() -> None:
    assert studies.build_study.__module__ == "bochan.serving.fastapi.services.studies"
    assert studies.restore_trials.__module__ == "bochan.serving.fastapi.services.studies"
    assert study_results.history_records.__module__ == "bochan.serving.fastapi.services.study_results"
    assert study_results.pareto_records.__module__ == "bochan.serving.fastapi.services.study_results"


def test_removed_root_study_service_does_not_exist() -> None:
    assert importlib.util.find_spec("bochan.serving.fastapi.study_service") is None
