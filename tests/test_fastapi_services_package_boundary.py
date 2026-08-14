from __future__ import annotations

import bochan.serving.fastapi.services as services
from bochan.serving.fastapi.services import candidates, studies, study_results, tabular


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
