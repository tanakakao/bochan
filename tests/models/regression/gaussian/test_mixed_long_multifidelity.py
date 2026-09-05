from __future__ import annotations

import pytest
import torch
from gpytorch.kernels import ProductKernel, ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

from bochan.models.multifidelity import FidelitySpec
from bochan.models.regression.gaussian import GaussianMixedMultiFidelityGP


def _training_data():
    train_X = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.25],
            [0.0, 1.0, 1.0, 1.00],
            [0.5, 0.0, 1.0, 0.50],
            [0.5, 1.0, 0.0, 1.00],
            [1.0, 0.0, 0.5, 0.25],
            [1.0, 1.0, 0.5, 1.00],
        ],
        dtype=torch.double,
    )
    train_Y = (
        torch.sin(train_X[:, :1] * 2.0)
        + 0.5 * train_X[:, 1:2]
        + train_X[:, 2:3]
        + 0.2 * train_X[:, 3:4]
    )
    bounds = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
        dtype=torch.double,
    )
    return train_X, train_Y, bounds


def test_constructs_mixed_multifidelity_model_and_metadata():
    train_X, train_Y, bounds = _training_data()
    model = GaussianMixedMultiFidelityGP(
        train_X,
        train_Y,
        cat_dims=(1,),
        fidelity_spec=FidelitySpec(
            fidelity_features=(-1,),
            target_fidelities={-1: 1.0},
        ),
        bounds=bounds,
    )

    assert model.cat_dims == (1,)
    assert model.cont_dims == (0, 2)
    assert model.fidelity_features == (3,)
    assert model.input_mode == "mixed"
    assert model.fidelity_metadata == {
        "fidelity_mode": "feature",
        "fidelity_features": [3],
        "target_fidelities": {3: 1.0},
        "input_mode": "mixed",
        "cat_dims": [1],
    }


def test_default_kernel_separates_continuous_categorical_and_fidelity_axes():
    train_X, train_Y, bounds = _training_data()
    model = GaussianMixedMultiFidelityGP(
        train_X,
        train_Y,
        cat_dims=(1,),
        fidelity_spec=FidelitySpec(fidelity_features=(-1,)),
        bounds=bounds,
    )

    assert isinstance(model.covar_module, ProductKernel)
    assert len(model.covar_module.kernels) == 2

    data_kernel = model.covar_module.kernels[0]
    assert isinstance(data_kernel, ScaleKernel)
    assert isinstance(data_kernel.base_kernel, ProductKernel)

    continuous_kernel, categorical_kernel = data_kernel.base_kernel.kernels
    assert tuple(continuous_kernel.active_dims.tolist()) == (0, 2)
    assert tuple(categorical_kernel.active_dims.tolist()) == (1,)

    fidelity_kernel = model.covar_module.kernels[1]
    assert tuple(fidelity_kernel.active_dims.tolist()) == (3,)


def test_known_noise_and_prediction_are_supported():
    train_X, train_Y, bounds = _training_data()
    train_Yvar = torch.full_like(train_Y, 0.01)
    model = GaussianMixedMultiFidelityGP(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        cat_dims=(-3,),
        fidelity_spec=FidelitySpec(fidelity_features=(-1,)),
        bounds=bounds,
    )

    assert isinstance(model.likelihood, FixedNoiseGaussianLikelihood)
    test_X = torch.tensor(
        [[0.25, 0.0, 0.5, 1.0], [0.75, 1.0, 0.5, 0.5]],
        dtype=torch.double,
    )
    mean, std = model.predict(test_X)
    assert mean.shape == torch.Size([2, 1])
    assert std.shape == torch.Size([2, 1])
    assert torch.isfinite(mean).all()
    assert torch.isfinite(std).all()


def test_rejects_categorical_fidelity_collision():
    train_X, train_Y, bounds = _training_data()
    with pytest.raises(ValueError, match="must be disjoint"):
        GaussianMixedMultiFidelityGP(
            train_X,
            train_Y,
            cat_dims=(-1,),
            fidelity_spec=FidelitySpec(fidelity_features=(-1,)),
            bounds=bounds,
        )


def test_rejects_empty_categorical_dimensions():
    train_X, train_Y, bounds = _training_data()
    with pytest.raises(ValueError, match="cat_dims must contain at least one"):
        GaussianMixedMultiFidelityGP(
            train_X,
            train_Y,
            cat_dims=(),
            fidelity_spec=FidelitySpec(fidelity_features=(-1,)),
            bounds=bounds,
        )


def test_make_mll_targets_mixed_model():
    train_X, train_Y, bounds = _training_data()
    model = GaussianMixedMultiFidelityGP(
        train_X,
        train_Y,
        cat_dims=(1,),
        fidelity_spec=FidelitySpec(fidelity_features=(-1,)),
        bounds=bounds,
    )

    mll = model.make_mll()
    assert mll.model is model
    assert mll.likelihood is model.likelihood
