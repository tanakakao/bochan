from __future__ import annotations

import torch

import bochan.optim.nsgaii_adapter as adapter
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


def test_objective_adapter_restores_missing_singleton_q_axis_before_validation() -> None:
    """Deterministic population values keep q=1 before BoTorch validates them."""
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

    assert values.shape == torch.Size([250, 1, 2])
    torch.testing.assert_close(values[:, 0, :], samples)
