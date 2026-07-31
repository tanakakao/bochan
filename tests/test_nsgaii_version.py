from __future__ import annotations

import torch

from bochan.optim.nsgaii_adapter import optimize_acqf_nsgaii


class _TwoOutputAcquisition:
    def __init__(self) -> None:
        self.model = None

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return torch.cat([X[..., :1], 1.0 - X[..., :1]], dim=-1).squeeze(-2)


def test_botorch_nsgaii_backend_is_installed() -> None:
    """The core bochan installation includes BoTorch's optional pymoo backend."""
    from botorch.utils.multi_objective.optimize import optimize_with_nsgaii

    assert callable(optimize_with_nsgaii)


def test_current_nsgaii_signature_receives_public_options(monkeypatch) -> None:
    """Check that the adapter targets the current BoTorch NSGA-II API directly."""
    received: dict[str, object] = {}

    def current_optimize_acqf_nsgaii(**kwargs):
        received.update(kwargs)
        X = torch.tensor([[0.2], [0.8]], dtype=kwargs["bounds"].dtype)
        Y = torch.tensor([[0.2, 0.8], [0.8, 0.2]], dtype=kwargs["bounds"].dtype)
        return X, Y

    monkeypatch.setattr(
        "bochan.optim.nsgaii_adapter._base.optimize_acqf_nsgaii",
        current_optimize_acqf_nsgaii,
    )
    linear_constraint = (
        torch.tensor([0]),
        torch.tensor([1.0], dtype=torch.double),
        0.1,
    )

    def repair(X):
        return X

    optimize_acqf_nsgaii(
        acq_function=_TwoOutputAcquisition(),
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        num_objectives=2,
        q=2,
        inequality_constraints=[linear_constraint],
        max_attempts=4,
        discrete_choices={0: [0.0, 0.5, 1.0]},
        post_processing_func=repair,
    )

    assert received["inequality_constraints"] == [linear_constraint]
    assert received["max_attempts"] == 4
    assert received["discrete_choices"] == {0: [0.0, 0.5, 1.0]}
    assert received["post_processing_func"] is repair
