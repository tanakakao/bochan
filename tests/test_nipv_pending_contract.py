from __future__ import annotations

import torch
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler

from bochan.acquisition.non_gaussian.active_learning.hetero_multi_output import (
    qHeteroMultiOutputNonGaussianNegIntegratedResponseMeanVariance,
)
from bochan.acquisition.non_gaussian.active_learning.hetero_single_output import (
    qHeteroNonGaussianNegIntegratedResponseMeanVariance,
)
from bochan.acquisition.non_gaussian.active_learning.multi_output import (
    qMultiOutputNonGaussianNegIntegratedResponseMeanVariance,
)
from bochan.acquisition.non_gaussian.active_learning.single_output import (
    qNonGaussianNegIntegratedResponseMeanVariance,
)
from bochan.acquisition.regression.active_learning import (
    qRegressionNegIntegratedPosteriorVariance,
)
from bochan.acquisition.regression.active_learning._integrated import (
    qRegressionNegIntegratedPosteriorVariance as qInnerRegressionNIPV,
)
from bochan.acquisition.regression.active_learning.hetero_single_output import (
    qHeteroRegressionNegIntegratedPosteriorVariance,
)
from bochan.models.regression.non_gaussian.gamma.base import GammaGPModel
from bochan.models.regression.non_gaussian.multioutput import NonGaussianModelList

DTYPE = torch.double
BOUNDS = torch.tensor([[0.0], [1.0]], dtype=DTYPE)
MC_POINTS = torch.linspace(0.05, 0.95, 5, dtype=DTYPE).unsqueeze(-1)
INITIAL_PENDING = torch.tensor([[0.15]], dtype=DTYPE)
UPDATED_PENDING = torch.tensor([[0.35]], dtype=DTYPE)


def _regression_model() -> SingleTaskGP:
    train_x = torch.tensor([[0.0], [0.3], [0.6], [1.0]], dtype=DTYPE)
    train_y = torch.sin(train_x * 4.0)
    model = SingleTaskGP(train_x, train_y)
    model.eval()
    return model


def _gamma_model(offset: float = 0.0) -> GammaGPModel:
    train_x = torch.tensor([[0.05], [0.25], [0.5], [0.75], [0.95]], dtype=DTYPE)
    train_y = 1.0 + float(offset) + 0.75 * train_x
    model = GammaGPModel(
        train_x,
        train_y,
        num_inducing=3,
    )
    model.eval()
    model.likelihood.eval()
    return model


def _assert_pending_round_trip(acquisition) -> None:
    torch.testing.assert_close(acquisition.X_pending, INITIAL_PENDING)
    acquisition.set_X_pending(UPDATED_PENDING)
    torch.testing.assert_close(acquisition.X_pending, UPDATED_PENDING)
    acquisition.set_X_pending(None)
    assert acquisition.X_pending is None


def _run_sequential(acquisition, *, maxiter: int = 15) -> None:
    torch.manual_seed(0)
    candidates, values = optimize_acqf(
        acq_function=acquisition,
        bounds=BOUNDS,
        q=2,
        num_restarts=1,
        raw_samples=8,
        sequential=True,
        options={"maxiter": maxiter, "batch_limit": 1},
    )
    assert candidates.shape == torch.Size([2, 1])
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(values).all()
    assert acquisition.X_pending is None


def test_inner_regression_nipv_owns_real_pending_contract() -> None:
    acquisition = qInnerRegressionNIPV(
        model=_regression_model(),
        mc_points=MC_POINTS,
        X_pending=INITIAL_PENDING,
    )

    _assert_pending_round_trip(acquisition)
    _run_sequential(acquisition)


def test_public_regression_nipv_supports_real_sequential_optimize_acqf() -> None:
    acquisition = qRegressionNegIntegratedPosteriorVariance(
        model=_regression_model(),
        mc_points=MC_POINTS,
        X_pending=INITIAL_PENDING,
    )

    _assert_pending_round_trip(acquisition)
    _run_sequential(acquisition)


def test_hetero_regression_nipv_supports_real_sequential_optimize_acqf() -> None:
    acquisition = qHeteroRegressionNegIntegratedPosteriorVariance(
        model=_regression_model(),
        mc_points=MC_POINTS,
        X_pending=INITIAL_PENDING,
    )

    _assert_pending_round_trip(acquisition)
    _run_sequential(acquisition)


def test_non_gaussian_gamma_nipv_supports_real_sequential_optimize_acqf() -> None:
    acquisition = qNonGaussianNegIntegratedResponseMeanVariance(
        model=_gamma_model(),
        mc_points=MC_POINTS,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([1]), seed=123),
        X_pending=INITIAL_PENDING,
    )
    assert acquisition.uses_proxy is False

    _assert_pending_round_trip(acquisition)
    _run_sequential(acquisition, maxiter=10)


def test_multi_output_non_gaussian_nipv_supports_real_sequential_optimize_acqf() -> None:
    model = NonGaussianModelList(
        _gamma_model(offset=0.0),
        _gamma_model(offset=0.5),
    )
    acquisition = qMultiOutputNonGaussianNegIntegratedResponseMeanVariance(
        model=model,
        mc_points=MC_POINTS,
        num_samples=4,
        seed=123,
        X_pending=INITIAL_PENDING,
    )
    assert acquisition.uses_proxy is True

    _assert_pending_round_trip(acquisition)
    _run_sequential(acquisition, maxiter=8)


def test_hetero_non_gaussian_aliases_inherit_real_pending_contract() -> None:
    single = qHeteroNonGaussianNegIntegratedResponseMeanVariance(
        model=_gamma_model(),
        mc_points=MC_POINTS,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([1]), seed=456),
        X_pending=INITIAL_PENDING,
    )
    multi = qHeteroMultiOutputNonGaussianNegIntegratedResponseMeanVariance(
        model=NonGaussianModelList(
            _gamma_model(offset=0.0),
            _gamma_model(offset=0.5),
        ),
        mc_points=MC_POINTS,
        num_samples=4,
        seed=456,
        X_pending=INITIAL_PENDING,
    )

    _assert_pending_round_trip(single)
    _assert_pending_round_trip(multi)
