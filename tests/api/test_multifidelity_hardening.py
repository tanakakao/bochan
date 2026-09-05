from __future__ import annotations

import pytest
import torch

from bochan.api import BayesianOptimizer, FitConfig, ModelConfig, MultiOutputConfig
from bochan.models.multifidelity import FidelityCostConfig, build_fidelity_cost_utility
from bochan.models.regression.gaussian import GaussianMultiFidelityGP


def _identity_fit(target, **kwargs):
    return target


def _data(dtype=torch.double, device="cpu"):
    X = torch.tensor(
        [
            [0.0, 0.25],
            [0.0, 1.00],
            [0.5, 0.50],
            [0.5, 1.00],
            [1.0, 0.25],
            [1.0, 1.00],
        ],
        dtype=dtype,
        device=device,
    )
    Y = torch.sin(2.0 * X[:, :1]) + 0.2 * X[:, 1:2]
    return X, Y


def _config(*, multi_output: bool = False):
    return ModelConfig(
        task_type="regression",
        model_type="multifidelity_gp",
        input_type="normal",
        outcome_transform=False,
        model_kwargs={
            "fidelity_features": [-1],
            "target_fidelities": {-1: 1.0},
        },
        multi_output_config=(
            MultiOutputConfig(output_names=["a", "b"]) if multi_output else None
        ),
    )


def test_negative_cost_index_resolves_against_model_dimension():
    config = FidelityCostConfig(
        fixed_cost=1.0,
        fidelity_weights={-1: 4.0},
    )
    cost_model, _ = build_fidelity_cost_utility(
        config,
        fidelity_features=(2,),
        d=3,
    )
    X = torch.tensor([[0.2, 7.0, 0.5]], dtype=torch.double)
    cost = cost_model(X)
    assert torch.allclose(cost, torch.tensor([3.0], dtype=torch.double))


def test_negative_cost_index_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="out of range"):
        build_fidelity_cost_utility(
            FidelityCostConfig(fidelity_weights={-4: 1.0}),
            fidelity_features=(2,),
            d=3,
        )


def test_known_noise_tell_requires_new_yvar_and_refit_preserves_noise():
    X, Y = _data()
    Yvar = torch.full_like(Y, 0.01)
    optimizer = BayesianOptimizer(
        _config(),
        fit_config=FitConfig(fit_func=_identity_fit),
    ).fit(X, Y, Yvar)

    new_X = torch.tensor([[0.75, 0.5]], dtype=torch.double)
    new_Y = torch.tensor([[0.9]], dtype=torch.double)
    with pytest.raises(ValueError, match="new_Yvar is required"):
        optimizer.tell(new_X, new_Y, refit=False)

    new_Yvar = torch.tensor([[0.02]], dtype=torch.double)
    optimizer.tell(new_X, new_Y, new_Yvar, refit=True)

    assert optimizer.train_X.shape[0] == X.shape[0] + 1
    assert optimizer.train_Yvar.shape == optimizer.train_Y.shape
    assert torch.allclose(optimizer.train_Yvar[-1], new_Yvar[-1])
    assert isinstance(optimizer.model, GaussianMultiFidelityGP)
    assert optimizer.model.fidelity_features == (1,)
    assert optimizer.model.target_fidelities == {1: 1.0}


def test_refit_preserves_resolved_fidelity_metadata():
    X, Y = _data()
    optimizer = BayesianOptimizer(
        _config(),
        fit_config=FitConfig(fit_func=_identity_fit),
    ).fit(X, Y)

    before = (
        optimizer.model.fidelity_features,
        dict(optimizer.model.target_fidelities),
    )
    optimizer.refit()
    after = (
        optimizer.model.fidelity_features,
        dict(optimizer.model.target_fidelities),
    )
    assert after == before


def test_multioutput_known_noise_builds_independent_fixed_noise_mf_models():
    X, Y1 = _data()
    Y = torch.cat([Y1, 0.5 * Y1 + 0.1], dim=-1)
    Yvar = torch.full_like(Y, 0.01)
    optimizer = BayesianOptimizer(
        _config(multi_output=True),
        fit_config=FitConfig(fit_func=_identity_fit),
    ).fit(X, Y, Yvar)

    assert len(optimizer.model.models) == 2
    for model in optimizer.model.models:
        assert model.fidelity_features == (1,)
        assert model.target_fidelities == {1: 1.0}


def test_dtype_and_device_are_preserved_through_fit_and_predict():
    X, Y = _data(dtype=torch.float64)
    optimizer = BayesianOptimizer(
        _config(),
        fit_config=FitConfig(fit_func=_identity_fit),
    ).fit(X, Y)
    X_test = torch.tensor([[0.25, 1.0]], dtype=X.dtype, device=X.device)
    mean = optimizer.predict(X_test, return_type="mean")
    assert mean.dtype == X.dtype
    assert mean.device == X.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_multifidelity_fit_predict_smoke():
    X, Y = _data(device="cuda")
    optimizer = BayesianOptimizer(
        _config(),
        fit_config=FitConfig(fit_func=_identity_fit),
    ).fit(X, Y)
    X_test = torch.tensor([[0.25, 1.0]], dtype=X.dtype, device=X.device)
    mean = optimizer.predict(X_test, return_type="mean")
    assert mean.is_cuda
