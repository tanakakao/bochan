from __future__ import annotations

import torch

from bochan.optim.nsgaii_adapter import optimize_acqf_nsgaii


class _VectorAcquisition:
    def __init__(self) -> None:
        self.model = None

    def __call__(self, X):
        return X.squeeze(-2)


def test_nsgaii_accepts_current_linear_constraint_arguments(monkeypatch) -> None:
    """Verify Bochan forwards current NSGA-II constraint options directly."""
    captured: dict[str, object] = {}

    def fake_optimize_acqf_nsgaii(**kwargs):
        captured.update(kwargs)
        X = torch.tensor([[0.4, 0.6]], dtype=torch.double)
        return X, X.clone()

    monkeypatch.setattr(
        "bochan.optim.nsgaii_adapter._base.optimize_acqf_nsgaii",
        fake_optimize_acqf_nsgaii,
    )
    constraints = [(torch.tensor([0, 1]), torch.tensor([1.0, 1.0]), 1.0)]
    X, Y = optimize_acqf_nsgaii(
        acq_function=_VectorAcquisition(),
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        q=1,
        inequality_constraints=constraints,
        seed=3,
    )

    torch.testing.assert_close(X, torch.tensor([[0.4, 0.6]], dtype=torch.double))
    torch.testing.assert_close(Y, X)
    assert captured["inequality_constraints"] is constraints
    assert captured["seed"] == 3
