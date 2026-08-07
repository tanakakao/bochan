from __future__ import annotations

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.transforms import t_batch_mode_transform

import bochan.acquisition.non_gaussian.active_learning.multi_output as ng_multi
import bochan.acquisition.non_gaussian.active_learning.single_output as ng_single
import bochan.acquisition.regression.active_learning._integrated as reg_inner
import bochan.acquisition.regression.active_learning.hetero_single_output as hetero_reg
import bochan.acquisition.regression.active_learning.integrated_variance as reg_outer
from bochan.acquisition.regression.active_learning import qRegressionNegIntegratedPosteriorVariance

DTYPE = torch.double
BOUNDS = torch.tensor([[0.0], [1.0]], dtype=DTYPE)
MC_POINTS = torch.linspace(0.0, 1.0, 7, dtype=DTYPE).unsqueeze(-1)


class _DummyModel(torch.nn.Module):
    supports_non_gaussian_nipv = True

    def fantasize(self, *args, **kwargs):
        return self


class _DummyPendingAcquisition(AcquisitionFunction):
    def __init__(self, model=None, X_pending=None, **kwargs) -> None:
        super().__init__(model=model)
        self.X_pending = X_pending

    def set_X_pending(self, X_pending=None) -> None:
        self.X_pending = X_pending

    @t_batch_mode_transform()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return -((X - 0.5) ** 2).sum(dim=(-1, -2))


def _run_sequential(acquisition: AcquisitionFunction) -> None:
    candidates, values = optimize_acqf(
        acq_function=acquisition,
        bounds=BOUNDS,
        q=2,
        num_restarts=2,
        raw_samples=16,
        sequential=True,
        options={"maxiter": 25},
    )
    assert candidates.shape == torch.Size([2, 1])
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(values).all()
    assert acquisition.X_pending is None


def test_public_regression_nipv_supports_sequential_optimize_acqf() -> None:
    train_x = torch.tensor([[0.0], [0.35], [0.7], [1.0]], dtype=DTYPE)
    train_y = torch.sin(train_x * 4.0)
    model = SingleTaskGP(train_x, train_y)
    model.eval()
    acquisition = qRegressionNegIntegratedPosteriorVariance(
        model=model,
        mc_points=MC_POINTS,
    )

    _run_sequential(acquisition)


@pytest.mark.parametrize(
    ("module", "target_name", "constructor"),
    [
        (
            reg_inner,
            "_BoTorchQNegIntegratedPosteriorVariance",
            lambda: reg_inner.qRegressionNegIntegratedPosteriorVariance(model=_DummyModel(), mc_points=MC_POINTS),
        ),
        (
            reg_outer,
            "_BoTorchNegIntegratedPosteriorVariance",
            lambda: reg_outer.qRegressionNegIntegratedPosteriorVariance(model=_DummyModel(), mc_points=MC_POINTS),
        ),
        (
            hetero_reg,
            "_BoTorchQNegIntegratedPosteriorVariance",
            lambda: hetero_reg.qHeteroRegressionNegIntegratedPosteriorVariance(
                model=_DummyModel(), mc_points=MC_POINTS
            ),
        ),
        (
            ng_single,
            "qRegressionNegIntegratedPosteriorVariance",
            lambda: ng_single.qNonGaussianNegIntegratedResponseMeanVariance(model=_DummyModel(), mc_points=MC_POINTS),
        ),
    ],
)
def test_delegating_single_output_nipv_wrappers_expose_pending_contract(
    monkeypatch: pytest.MonkeyPatch,
    module,
    target_name: str,
    constructor,
) -> None:
    monkeypatch.setattr(module, target_name, _DummyPendingAcquisition)
    acquisition = constructor()
    assert acquisition.X_pending is None
    pending = torch.tensor([[0.2]], dtype=DTYPE)
    acquisition.set_X_pending(pending)
    torch.testing.assert_close(acquisition.X_pending, pending)
    acquisition.set_X_pending(None)
    assert acquisition.X_pending is None
    _run_sequential(acquisition)


def test_multi_output_non_gaussian_nipv_wrapper_exposes_pending_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ng_multi._single,
        "qNonGaussianIntegratedResponseMeanVarianceProxy",
        _DummyPendingAcquisition,
    )
    acquisition = ng_multi.qMultiOutputNonGaussianNegIntegratedResponseMeanVariance(
        model=_DummyModel(),
        mc_points=MC_POINTS,
    )
    assert acquisition.X_pending is None
    pending = torch.tensor([[0.3]], dtype=DTYPE)
    acquisition.set_X_pending(pending)
    torch.testing.assert_close(acquisition.X_pending, pending)
    acquisition.set_X_pending(None)
    assert acquisition.X_pending is None
    _run_sequential(acquisition)


def test_direct_nipv_implementations_already_own_pending_state() -> None:
    from bochan.acquisition.ordinal.active_learning import (
        qOrdinalFantasyNegIntegratedPosteriorVariance,
    )
    from bochan.acquisition.regression.active_learning import (
        qMultiOutputRegressionNegIntegratedPosteriorVariance,
    )

    assert "self.X_pending" in __import__("inspect").getsource(qMultiOutputRegressionNegIntegratedPosteriorVariance)
    assert "self.X_pending" in __import__("inspect").getsource(qOrdinalFantasyNegIntegratedPosteriorVariance)
