"""Regression tests for model artifacts that include candidate history."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from bochan.api import CandidateResult, OptimizeConfig  # noqa: E402
from bochan.model_artifact import (  # noqa: E402
    deserialize_model_artifact,
    serialize_model_artifact,
)
from bochan.tabular import TabularBayesianOptimizer  # noqa: E402


def test_model_artifact_keeps_history_without_request_local_postprocess() -> None:
    def final_candidate_postprocess(value):
        return value

    opt_config = OptimizeConfig(
        q=3,
        final_candidate_postprocess=final_candidate_postprocess,
    )
    candidate_result = CandidateResult(
        candidates=None,
        acq_value=None,
        acqf=SimpleNamespace(name="EI"),
        acq_config=SimpleNamespace(name="EI"),
        opt_config=opt_config,
        data_context=SimpleNamespace(),
    )

    optimizer = object.__new__(TabularBayesianOptimizer)
    optimizer.dataset = SimpleNamespace()
    optimizer.bo = SimpleNamespace(
        bundle=SimpleNamespace(task_type="regression", model_type="base"),
        history=[candidate_result],
    )

    content = serialize_model_artifact(
        optimizer,
        backend="tabular",
    )
    payload = deserialize_model_artifact(
        content,
        trust_pickle=True,
        expected_backend="tabular",
    )

    restored_history = payload["optimizer"].bo.history
    assert len(restored_history) == 1
    assert restored_history[0].opt_config.q == 3
    assert restored_history[0].opt_config.final_candidate_postprocess is None
