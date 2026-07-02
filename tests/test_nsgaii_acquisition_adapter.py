from __future__ import annotations

import torch

import bochan.optim.nsgaii_adapter as adapter


class _FakeMultiOutputModel:
    num_outputs = 2

    def posterior(self, X: torch.Tensor):
        return type("Posterior", (), {"mean": torch.cat([X[..., :1], 1.0 - X[..., :1]], dim=-1)})()


class _ScalarEhviLikeAcquisition:
    def __init__(self) -> None:
        self.model = _FakeMultiOutputModel()
        self.objective = lambda Y: Y
        self.constraints = [lambda Y: 0.2 - Y[..., 0]]
        self.ref_point = torch.tensor([0.0, 0.0], dtype=torch.double)


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
    assert target.__class__.__name__ == "MultiOutputPosteriorMean"
    assert captured["objective"] is acquisition.objective
    assert captured["constraints"] == acquisition.constraints
    torch.testing.assert_close(captured["ref_point"], acquisition.ref_point)


def test_existing_multioutput_acquisition_is_preserved(monkeypatch) -> None:
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

    assert captured["acq_function"] is target
    assert captured["objective"] is None
    assert captured["constraints"] is None
    assert captured["ref_point"] is None
