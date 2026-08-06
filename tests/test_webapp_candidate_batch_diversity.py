from __future__ import annotations

from types import SimpleNamespace

import pytest

from bochan.serving.webapp.app import RegressionRunRequest
from bochan.serving.webapp.candidate_batch_diversity import (
    candidate_uniqueness_metadata,
    install_web_candidate_batch_diversity,
    prepare_web_candidate_request,
    resolve_web_candidate_sequential,
)


def test_web_q3_normal_search_forces_sequential_pending_selection() -> None:
    effective, strategy = resolve_web_candidate_sequential(
        q=3,
        search_method="normal",
        requested=False,
    )

    assert effective is True
    assert strategy == "sequential_pending"


@pytest.mark.parametrize(
    "method",
    ["nsgaii", "nsga2", "thompson_sampling", "optimize_thompson_sampling"],
)
def test_native_batch_search_methods_keep_their_own_batch_selection(
    method: str,
) -> None:
    effective, strategy = resolve_web_candidate_sequential(
        q=3,
        search_method=method,
        requested=True,
    )

    assert effective is False
    assert strategy == "native_batch"


def test_prepare_web_candidate_request_does_not_mutate_original_request() -> None:
    request = RegressionRunRequest(
        dataset_id="dataset",
        feature_columns=["x"],
        target_column="y",
        target_columns=["y"],
        optimizer={
            "name": "normal",
            "q": 3,
            "num_restarts": 4,
            "raw_samples": 16,
            "sequential": False,
        },
    )

    prepared, metadata = prepare_web_candidate_request(request)

    assert request.optimizer.sequential is False
    assert prepared.optimizer.sequential is True
    assert metadata == {
        "candidate_batch_strategy": "sequential_pending",
        "candidate_sequential_requested": False,
        "candidate_sequential_effective": True,
    }


def test_candidate_uniqueness_metadata_uses_final_encoded_values() -> None:
    metadata = candidate_uniqueness_metadata(
        {
            "candidates": [
                {"encoded_values": {"x": 0.1, "category": 1.0}},
                {"encoded_values": {"x": 0.1, "category": 1.0}},
                {"encoded_values": {"x": 0.2, "category": 1.0}},
            ]
        }
    )

    assert metadata == {
        "candidate_count": 3,
        "candidate_unique_count": 2,
        "candidate_duplicate_count": 1,
    }


def test_installed_adapter_passes_effective_request_to_workflow() -> None:
    def original(request, store):
        del store
        return {
            "candidates": [
                {"encoded_values": {"x": 0.1}},
                {"encoded_values": {"x": 0.2}},
                {"encoded_values": {"x": 0.3}},
            ],
            "metadata": {
                "workflow_sequential": request.optimizer.sequential,
            },
        }

    workflows = SimpleNamespace(_run_regression_web_workflow=original)
    workflows_tabular = SimpleNamespace(run_regression_web_workflow=original)
    install_web_candidate_batch_diversity(workflows, workflows_tabular)

    request = SimpleNamespace(
        optimizer=SimpleNamespace(name="normal", q=3, sequential=False)
    )
    result = workflows._run_regression_web_workflow(request, None)

    assert result["metadata"]["workflow_sequential"] is True
    assert result["metadata"]["candidate_batch_strategy"] == "sequential_pending"
    assert result["metadata"]["candidate_duplicate_count"] == 0
