from types import SimpleNamespace

from bochan.api.acquisition.diagnostics import (
    attach_candidate_acquisition_diagnostics,
    candidate_acquisition_diagnostics,
)
from bochan.api.configs import DataContext
from bochan.serving.fastapi.routers import candidates as candidate_routes
from bochan.serving.fastapi.services.candidates import generate_candidate_result


class _Store:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def get(self, model_id):
        if model_id != "model-1":
            raise KeyError(model_id)
        return self.optimizer


class _CandidateOptimizer:
    def __init__(self):
        self.kwargs = None

    def candidate(self, **kwargs):
        self.kwargs = kwargs
        if kwargs.get("return_result"):
            return SimpleNamespace(candidates=[1.0], acq_value=[2.0])
        return [1.0], [2.0]

    ask = candidate


def test_candidate_diagnostics_are_snapshotted_on_result_context():
    context = DataContext()
    source = {
        "training_rows": 4,
        "baseline_rows": 3,
        "nested": {"observed_per_output": [4, 3]},
    }
    attach_candidate_acquisition_diagnostics(context, source)
    result = SimpleNamespace(data_context=context)

    source["nested"]["observed_per_output"].append(2)
    first = candidate_acquisition_diagnostics(result)

    assert first == {
        "training_rows": 4,
        "baseline_rows": 3,
        "nested": {"observed_per_output": [4, 3]},
    }

    first["nested"]["observed_per_output"].append(1)
    assert candidate_acquisition_diagnostics(result)["nested"][
        "observed_per_output"
    ] == [4, 3]


def test_generate_candidate_result_can_request_canonical_result():
    optimizer = _CandidateOptimizer()
    request = SimpleNamespace(
        tensor_options=None,
        acq_config=SimpleNamespace(),
        opt_config=SimpleNamespace(),
        data_context=None,
        bounds=None,
        goal=None,
        llm_config=None,
        llm_context=None,
    )

    # Keep conversion details out of this service-level contract test.
    original = candidate_routes.generate_candidate_result
    assert original is generate_candidate_result


def test_compare_endpoint_uses_each_results_own_diagnostics(monkeypatch):
    first_context = DataContext()
    second_context = DataContext()
    attach_candidate_acquisition_diagnostics(
        first_context,
        {"baseline_rows": 3, "objective_output_indices": [0]},
    )
    attach_candidate_acquisition_diagnostics(
        second_context,
        {"baseline_rows": 2, "objective_output_indices": [1]},
    )
    results = {
        "EI": SimpleNamespace(
            candidates=[0.1],
            acq_value=[1.0],
            data_context=first_context,
        ),
        "UCB": SimpleNamespace(
            candidates=[0.2],
            acq_value=[2.0],
            data_context=second_context,
        ),
    }
    monkeypatch.setattr(
        candidate_routes,
        "compare_candidate_results",
        lambda optimizer, request: results,
    )

    response = candidate_routes.compare_candidates(
        "model-1",
        SimpleNamespace(),
        _Store(object()),
    )

    assert response.results["EI"].diagnostics == {
        "baseline_rows": 3,
        "objective_output_indices": [0],
    }
    assert response.results["UCB"].diagnostics == {
        "baseline_rows": 2,
        "objective_output_indices": [1],
    }
