from types import SimpleNamespace

import torch

from bochan.api import DataContext
from bochan.serving.fastapi.routers import candidates as candidate_routes


class _Store:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def get(self, model_id):
        if model_id != "model-1":
            raise KeyError(model_id)
        return self.optimizer


def _result(candidate, acq_value, diagnostics):
    context = DataContext()
    object.__setattr__(context, "_acquisition_diagnostics_snapshot", diagnostics)
    return SimpleNamespace(
        candidates=torch.tensor([[candidate]], dtype=torch.double),
        acq_value=torch.tensor(acq_value, dtype=torch.double),
        data_context=context,
    )


def test_compare_candidates_returns_each_results_own_diagnostics(monkeypatch):
    first = _result(
        0.2,
        1.1,
        {
            "baseline_rows": 3,
            "objective_output_indices": [0],
            "baseline_filtered": True,
        },
    )
    second = _result(
        0.8,
        2.2,
        {
            "baseline_rows": 5,
            "objective_output_indices": [1],
            "baseline_filtered": False,
        },
    )
    optimizer = SimpleNamespace(
        last_acquisition_diagnostics={"baseline_rows": 999},
    )

    monkeypatch.setattr(
        candidate_routes,
        "compare_candidate_results",
        lambda *_args, **_kwargs: {"qei": first, "qucb": second},
    )

    response = candidate_routes.compare_candidates(
        "model-1",
        SimpleNamespace(),
        _Store(optimizer),
    )

    assert response.results["qei"].diagnostics == {
        "baseline_rows": 3,
        "objective_output_indices": [0],
        "baseline_filtered": True,
    }
    assert response.results["qucb"].diagnostics == {
        "baseline_rows": 5,
        "objective_output_indices": [1],
        "baseline_filtered": False,
    }
    assert response.results["qei"].diagnostics != optimizer.last_acquisition_diagnostics
    assert response.results["qucb"].diagnostics != optimizer.last_acquisition_diagnostics


def test_compare_candidates_keeps_backward_compatible_null_diagnostics(monkeypatch):
    result = SimpleNamespace(
        candidates=torch.tensor([[0.5]], dtype=torch.double),
        acq_value=torch.tensor(1.0, dtype=torch.double),
        data_context=DataContext(),
    )
    optimizer = SimpleNamespace(last_acquisition_diagnostics={"baseline_rows": 9})

    monkeypatch.setattr(
        candidate_routes,
        "compare_candidate_results",
        lambda *_args, **_kwargs: {"qei": result},
    )

    response = candidate_routes.compare_candidates(
        "model-1",
        SimpleNamespace(),
        _Store(optimizer),
    )

    assert response.results["qei"].diagnostics is None
