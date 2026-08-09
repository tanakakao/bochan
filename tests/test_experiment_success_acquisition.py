from __future__ import annotations

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf

from bochan.acquisition.feasible import ExperimentSuccessWeightedAcquisition


class _DummyModel:
    pass


class _ConstantAcquisition(AcquisitionFunction):
    def __init__(self, value: float) -> None:
        super().__init__(model=_DummyModel())
        self.value = float(value)
        self.X_pending = None

    def forward(self, X):
        return torch.full(
            X.shape[:-2],
            self.value,
            dtype=X.dtype,
            device=X.device,
        )

    def set_X_pending(self, X_pending=None):
        self.X_pending = X_pending


class _ProbabilityPosterior:
    def __init__(self, mean) -> None:
        self.mean = mean


class _IncreasingSuccessModel:
    def probability_posterior(self, X):
        return _ProbabilityPosterior(X[..., :1])


def test_success_weighting_guides_optimize_acqf_to_high_success_region() -> None:
    acqf = ExperimentSuccessWeightedAcquisition(
        acqf=_ConstantAcquisition(1.0),
        success_model=_IncreasingSuccessModel(),
        min_success_probability=0.7,
        eta=0.05,
    )
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    candidate, value = optimize_acqf(
        acqf,
        bounds=bounds,
        q=1,
        num_restarts=4,
        raw_samples=32,
    )

    assert candidate.item() > 0.9
    assert value.item() > 0.95


def test_lower_success_never_improves_negative_base_acquisition() -> None:
    acqf = ExperimentSuccessWeightedAcquisition(
        acqf=_ConstantAcquisition(-1.0),
        success_model=_IncreasingSuccessModel(),
        min_success_probability=0.7,
        eta=0.05,
    )
    low = torch.tensor([[[0.1]]], dtype=torch.double)
    high = torch.tensor([[[0.9]]], dtype=torch.double)

    low_value = acqf(low)
    high_value = acqf(high)

    assert low_value.item() < high_value.item()


def test_success_weighting_has_finite_candidate_gradient() -> None:
    acqf = ExperimentSuccessWeightedAcquisition(
        acqf=_ConstantAcquisition(1.0),
        success_model=_IncreasingSuccessModel(),
        min_success_probability=0.7,
        eta=0.05,
    )
    X = torch.tensor([[[0.6]]], dtype=torch.double, requires_grad=True)

    value = acqf(X).sum()
    value.backward()

    assert X.grad is not None
    assert torch.isfinite(X.grad).all()
    assert X.grad.item() > 0.0
