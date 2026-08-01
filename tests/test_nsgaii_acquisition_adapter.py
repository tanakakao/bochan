from __future__ import annotations

import numpy as np
import torch

import bochan.optim.nsgaii_adapter as adapter
from bochan.optim.nsgaii_diversity import select_diverse_nsgaii_candidates
from bochan.optim.nsgaii_outputs import (
    NSGAIIAcquisitionContextAdapter,
    NSGAIIObjectiveOutputAdapter,
)


class _FakeMultiOutputModel:
    num_outputs = 2

    def posterior(self, X: torch.Tensor):
        return type(
            "Posterior",
            (),
            {"mean": torch.cat([X[..., :1], 1.0 - X[..., :1]], dim=-1)},
        )()


class _ScalarEhviLikeAcquisition:
    def __init__(self) -> None:
        self.model = _FakeMultiOutputModel()
        self.objective = lambda Y: Y
        self.constraints = [lambda Y: 0.2 - Y[..., 0]]
        self.ref_point = torch.tensor([0.0, 0.0], dtype=torch.double)


def _clustered_pareto_pool() -> tuple[torch.Tensor, torch.Tensor]:
    candidates = torch.tensor(
        [
            [0.00, 0.00],
            [0.01, 0.00],
            [0.02, 0.00],
            [1.00, 0.00],
            [0.00, 1.00],
            [1.00, 1.00],
        ],
        dtype=torch.double,
    )
    values = torch.tensor(
        [
            [1.00, 0.00],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.60, 0.80],
            [0.80, 0.60],
            [0.00, 1.00],
        ],
        dtype=torch.double,
    )
    return candidates, values


def test_botorch_nsgaii_backend_is_installed() -> None:
    """The project dependency includes BoTorch's optional pymoo backend."""
    from botorch.utils.multi_objective.optimize import optimize_with_nsgaii

    assert callable(optimize_with_nsgaii)


def test_scalar_multiobjective_acquisition_is_adapted_to_posterior_mean(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_optimize_acqf_nsgaii(**kwargs):
        captured.update(kwargs)
        return (
            torch.tensor([[0.2], [0.8]], dtype=torch.double),
            torch.tensor([[0.2, 0.8], [0.8, 0.2]], dtype=torch.double),
        )

    monkeypatch.setattr(
        adapter._base,
        "optimize_acqf_nsgaii",
        fake_optimize_acqf_nsgaii,
    )
    acquisition = _ScalarEhviLikeAcquisition()

    adapter.optimize_acqf_nsgaii(
        acq_function=acquisition,
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        q=2,
    )

    target = captured["acq_function"]
    assert target is not acquisition
    assert target.__class__.__name__ == "NSGAIIAcquisitionContextAdapter"
    assert target.acq_function.__class__.__name__ == "MultiOutputPosteriorMean"

    objective = captured["objective"]
    assert objective.__class__.__name__ == "NSGAIIObjectiveOutputAdapter"
    assert objective.objective is acquisition.objective
    assert objective.acquisition_context is target

    assert captured["constraints"] == acquisition.constraints
    torch.testing.assert_close(captured["ref_point"], acquisition.ref_point)


def test_existing_multioutput_acquisition_is_preserved_inside_context_adapter(
    monkeypatch,
) -> None:
    from botorch.acquisition.multioutput_acquisition import MultiOutputPosteriorMean

    target = MultiOutputPosteriorMean(model=_FakeMultiOutputModel())
    captured: dict[str, object] = {}

    def fake_optimize_acqf_nsgaii(**kwargs):
        captured.update(kwargs)
        return (
            torch.tensor([[0.2], [0.8]], dtype=torch.double),
            torch.tensor([[0.2, 0.8], [0.8, 0.2]], dtype=torch.double),
        )

    monkeypatch.setattr(
        adapter._base,
        "optimize_acqf_nsgaii",
        fake_optimize_acqf_nsgaii,
    )

    adapter.optimize_acqf_nsgaii(
        acq_function=target,
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        q=2,
    )

    acquisition_context = captured["acq_function"]
    assert acquisition_context.__class__.__name__ == "NSGAIIAcquisitionContextAdapter"
    assert acquisition_context.acq_function is target

    objective = captured["objective"]
    assert objective.__class__.__name__ == "NSGAIIObjectiveOutputAdapter"
    assert objective.objective is None
    assert objective.acquisition_context is acquisition_context

    assert captured["constraints"] is None
    assert captured["ref_point"] is None


def test_sequential_true_is_forced_false_for_population_optimization(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_optimize_acqf_nsgaii(**kwargs):
        captured.update(kwargs)
        return (
            torch.tensor([[0.2], [0.8]], dtype=torch.double),
            torch.tensor([[0.2, 0.8], [0.8, 0.2]], dtype=torch.double),
        )

    monkeypatch.setattr(
        adapter._base,
        "optimize_acqf_nsgaii",
        fake_optimize_acqf_nsgaii,
    )

    adapter.optimize_acqf_nsgaii(
        acq_function=_ScalarEhviLikeAcquisition(),
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        q=2,
        sequential=True,
    )

    assert captured["sequential"] is False


def test_objective_adapter_restores_q_for_validation_then_returns_pymoo_shape() -> None:
    """The internal q-axis is removed again before values are returned to PyMOO."""
    from botorch.acquisition.multi_objective.objective import (
        IdentityMCMultiOutputObjective,
    )

    context = NSGAIIAcquisitionContextAdapter(lambda X: X)
    context.last_X = torch.rand(250, 1, 3, dtype=torch.double)
    objective = NSGAIIObjectiveOutputAdapter(
        IdentityMCMultiOutputObjective(),
        context,
    )
    samples = torch.rand(250, 2, dtype=torch.double)

    values = objective(samples)

    assert values.shape == torch.Size([250, 2])
    torch.testing.assert_close(values, samples)


def test_pymoo_problem_aligns_objective_constraints_and_reference_point() -> None:
    """Web reference constraints must remain 2D with deterministic q=1 values."""
    from botorch.acquisition.multi_objective.objective import (
        IdentityMCMultiOutputObjective,
    )
    from botorch.utils.multi_objective.optimize import BotorchPymooProblem

    def deterministic_objectives(X: torch.Tensor) -> torch.Tensor:
        values = torch.cat([X[..., :1], 1.0 - X[..., :1]], dim=-1)
        return values.squeeze(-2)

    acquisition = NSGAIIAcquisitionContextAdapter(deterministic_objectives)
    objective = NSGAIIObjectiveOutputAdapter(
        IdentityMCMultiOutputObjective(),
        acquisition,
    )
    problem = BotorchPymooProblem(
        n_var=1,
        n_obj=2,
        xl=np.array([0.0]),
        xu=np.array([1.0]),
        acqf=acquisition,
        dtype=torch.double,
        device=torch.device("cpu"),
        ref_point=torch.tensor([0.0, 0.0], dtype=torch.double),
        objective=objective,
        constraints=[lambda Y: 0.2 - Y[..., 0]],
    )
    out: dict[str, np.ndarray] = {}

    problem._evaluate(np.linspace(0.0, 1.0, 250).reshape(-1, 1), out)

    assert out["F"].shape == (250, 2)
    assert out["G"].shape == (250, 3)


def test_diverse_selector_separates_near_duplicate_input_conditions() -> None:
    """Final q points should cover the input space as well as the Pareto front."""

    candidates, values = _clustered_pareto_pool()

    selected_X, selected_Y = select_diverse_nsgaii_candidates(
        candidates,
        values,
        q=3,
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        input_weight=0.7,
    )

    assert selected_X.shape == torch.Size([3, 2])
    assert selected_Y.shape == torch.Size([3, 2])
    assert float(torch.pdist(selected_X).min()) > 0.9


def test_nsgaii_requests_larger_pool_then_returns_diverse_q_batch(monkeypatch) -> None:
    """q=3 is selected from a larger Pareto-oriented NSGA-II result pool."""

    captured: dict[str, object] = {}
    pool_X, pool_Y = _clustered_pareto_pool()

    def fake_optimize_acqf_nsgaii(**kwargs):
        captured.update(kwargs)
        return pool_X, pool_Y

    monkeypatch.setattr(
        adapter._base,
        "optimize_acqf_nsgaii",
        fake_optimize_acqf_nsgaii,
    )

    selected_X, selected_Y = adapter.optimize_acqf_nsgaii(
        acq_function=_ScalarEhviLikeAcquisition(),
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        q=3,
        population_size=100,
    )

    assert captured["q"] == 60
    assert selected_X.shape == torch.Size([3, 2])
    assert selected_Y.shape == torch.Size([3, 2])
    assert float(torch.pdist(selected_X).min()) > 0.9


def test_nsgaii_diversity_can_be_disabled(monkeypatch) -> None:
    """Direct API users can retain BoTorch's original q-selection behavior."""

    captured: dict[str, object] = {}
    pool_X, pool_Y = _clustered_pareto_pool()

    def fake_optimize_acqf_nsgaii(**kwargs):
        captured.update(kwargs)
        return pool_X[:3], pool_Y[:3]

    monkeypatch.setattr(
        adapter._base,
        "optimize_acqf_nsgaii",
        fake_optimize_acqf_nsgaii,
    )

    selected_X, selected_Y = adapter.optimize_acqf_nsgaii(
        acq_function=_ScalarEhviLikeAcquisition(),
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        q=3,
        diversify=False,
    )

    assert captured["q"] == 3
    torch.testing.assert_close(selected_X, pool_X[:3])
    torch.testing.assert_close(selected_Y, pool_Y[:3])
