from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

pytest.importorskip("fastapi")

from bochan.serving.fastapi.routers import tabular as tabular_router
from bochan.serving.fastapi.schemas import TabularBatchCandidateRequest


class _DummyTabularBayesianOptimizer:
    instances: list[_DummyTabularBayesianOptimizer] = []

    def __init__(
        self,
        *,
        model_config: dict[str, Any],
        fit_config: dict[str, Any],
        input_cols: list[str],
        target_cols: list[str],
    ) -> None:
        self.model_config = model_config
        self.fit_config = fit_config
        self.input_cols = input_cols
        self.target_cols = target_cols
        self.fit_data: pd.DataFrame | None = None
        self.candidate_calls: list[tuple[dict[str, Any], dict[str, Any]]] = []
        self.instances.append(self)

    def fit(self, df: pd.DataFrame) -> None:
        self.fit_data = df.copy()

    def candidate(
        self,
        *,
        acq_config: dict[str, Any],
        opt_config: dict[str, Any],
    ) -> tuple[pd.DataFrame, float]:
        self.candidate_calls.append((dict(acq_config), dict(opt_config)))
        return (
            pd.DataFrame(
                [
                    {
                        "raw material 1": 0.1,
                        "raw material 2": 0.2,
                        "raw material 3": 0.3,
                        "temperature": 100.0,
                        "time": 5.0,
                    }
                ]
            ),
            float("nan"),
        )


def _request(**updates: Any) -> TabularBatchCandidateRequest:
    payload: dict[str, Any] = {
        "data": [
            {
                "raw material 1": 0.1,
                "raw material 2": 0.2,
                "raw material 3": 0.3,
                "temperature": 100.0,
                "time": 5.0,
                "property": 0.5,
                "property2": 0.7,
            }
        ],
        "model_types": ["base"],
        "acquisition_names": ["ehvi", "nsgaii"],
        "optimizers": ["torch", "ga"],
        "fit_config": {"maxiter": 1},
        "optimize_config": {"q": 2, "num_restarts": 2, "raw_samples": 4},
    }
    payload.update(updates)
    return TabularBatchCandidateRequest.model_validate(payload)


def _patch_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tabular_router,
        "_load_tabular_dependencies",
        lambda: (pd, _DummyTabularBayesianOptimizer),
    )


def test_tabular_batch_runs_each_optimizer_except_for_nsgaii(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _DummyTabularBayesianOptimizer.instances.clear()
    _patch_dependencies(monkeypatch)

    response = tabular_router._run_batch(_request())

    assert response.n_models == 1
    assert response.n_runs == 3
    assert response.n_success == 3
    assert response.n_failed == 0

    instance = _DummyTabularBayesianOptimizer.instances[0]
    assert instance.model_config["model_type"] == "base"
    assert instance.fit_data is not None
    assert [call[0]["name"] for call in instance.candidate_calls] == [
        "ehvi",
        "ehvi",
        "nsgaii",
    ]
    assert [call[1].get("optimizer") for call in instance.candidate_calls] == [
        "torch",
        "ga",
        None,
    ]

    assert response.results[0].candidates[0]["temperature"] == 100.0
    assert response.results[0].acq_value is None


def test_tabular_batch_rejects_missing_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_dependencies(monkeypatch)
    request = _request(data=[{"raw material 1": 0.1}])

    with pytest.raises(ValueError, match="Missing required columns"):
        tabular_router._run_batch(request)
