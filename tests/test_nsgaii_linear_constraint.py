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


def test_base_nsgaii_omits_unsupported_linear_constraint_kwargs(monkeypatch) -> None:
    """Verify BoTorch versions without input constraints do not receive them."""
    import bochan.optim.nsgaii as base

    captured: dict[str, object] = {}

    def fake_optimize_with_nsgaii(
        *,
        acq_function,
        bounds,
        num_objectives,
        q,
        constraints=None,
        post_processing_func=None,
    ):
        captured.update(
            acq_function=acq_function,
            bounds=bounds,
            num_objectives=num_objectives,
            q=q,
            constraints=constraints,
            post_processing_func=post_processing_func,
        )
        X = torch.tensor([[0.25, 0.75]], dtype=torch.double)
        return X, X.clone()

    monkeypatch.setattr(base, "optimize_with_nsgaii", fake_optimize_with_nsgaii)
    constraints = [(torch.tensor([0, 1]), torch.tensor([1.0, 1.0]), 1.0)]

    X, Y = base.optimize_acqf_nsgaii(
        acq_function=_VectorAcquisition(),
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        q=1,
        num_objectives=2,
        inequality_constraints=constraints,
    )

    torch.testing.assert_close(X, torch.tensor([[0.25, 0.75]], dtype=torch.double))
    torch.testing.assert_close(Y, X)
    assert "inequality_constraints" not in captured


def test_base_nsgaii_forwards_supported_linear_constraint_kwargs(monkeypatch) -> None:
    """Verify newer BoTorch signatures still receive input constraints."""
    import bochan.optim.nsgaii as base

    captured: dict[str, object] = {}

    def fake_optimize_with_nsgaii(
        *,
        acq_function,
        bounds,
        num_objectives,
        q,
        inequality_constraints=None,
        **kwargs,
    ):
        captured.update(
            acq_function=acq_function,
            bounds=bounds,
            num_objectives=num_objectives,
            q=q,
            inequality_constraints=inequality_constraints,
            **kwargs,
        )
        X = torch.tensor([[0.3, 0.7]], dtype=torch.double)
        return X, X.clone()

    monkeypatch.setattr(base, "optimize_with_nsgaii", fake_optimize_with_nsgaii)
    constraints = [(torch.tensor([0, 1]), torch.tensor([1.0, 1.0]), 1.0)]

    X, Y = base.optimize_acqf_nsgaii(
        acq_function=_VectorAcquisition(),
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        q=1,
        num_objectives=2,
        inequality_constraints=constraints,
    )

    torch.testing.assert_close(X, torch.tensor([[0.3, 0.7]], dtype=torch.double))
    torch.testing.assert_close(Y, X)
    assert captured["inequality_constraints"] is not None
