from __future__ import annotations

import pytest
import torch
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

from bochan.models.multifidelity import FidelitySpec
from bochan.models.regression.gaussian import GaussianMultiFidelityGP


def _training_data():
    train_X = torch.tensor(
        [
            [0.0, 0.0, 0.25],
            [0.0, 1.0, 1.00],
            [0.5, 0.0, 0.50],
            [0.5, 1.0, 1.00],
            [1.0, 0.0, 0.25],
            [1.0, 1.0, 1.00],
        ],
        dtype=torch.double,
    )
    train_Y = (
        torch.sin(train_X[:, :1] * 2.0) + train_X[:, 1:2] + 0.2 * train_X[:, 2:3]
    )
    bounds = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.double
    )
    return train_X, train_Y, bounds


def test_constructs_from_negative_fidelity_index_and_exposes_metadata():
    train_X, train_Y, bounds = _training_data()
    model = GaussianMultiFidelityGP(
        train_X,
        train_Y,
        fidelity_spec=FidelitySpec(
            fidelity_features=(-1,), target_fidelities={-1: 1.0}
        ),
        bounds=bounds,
    )

    assert model.fidelity_features == (2,)
    assert model.target_fidelities == {2: 1.0}
    assert model.fidelity_metadata == {
        "fidelity_mode": "feature",
        "fidelity_features": [2],
        "target_fidelities": {2: 1.0},
        "input_mode": "continuous",
        "cat_dims": [],
    }
    assert model.train_inputs[0].shape == train_X.shape
    assert model.train_targets.shape == torch.Size([train_X.shape[0]])


def test_known_noise_uses_fixed_noise_likelihood():
    train_X, train_Y, bounds = _training_data()
    train_Yvar = torch.full_like(train_Y, 0.01)
    model = GaussianMultiFidelityGP(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        fidelity_spec=FidelitySpec(fidelity_features=(-1,)),
        bounds=bounds,
    )

    assert isinstance(model.likelihood, FixedNoiseGaussianLikelihood)
    assert model.train_Yvar_raw is not None
    assert torch.equal(model.train_Yvar_raw, train_Yvar)


def test_accepts_input_and_outcome_transforms_and_predicts():
    train_X, train_Y, bounds = _training_data()
    model = GaussianMultiFidelityGP(
        train_X,
        train_Y,
        fidelity_spec=FidelitySpec(fidelity_features=(-1,)),
        bounds=bounds,
        input_transform=Normalize(d=3, bounds=bounds),
        outcome_transform=Standardize(m=1),
    )

    test_X = torch.tensor([[0.25, 0.5, 1.0], [0.75, 0.5, 0.5]], dtype=torch.double)
    posterior = model.posterior(test_X)
    mean, std = model.predict(test_X)

    assert posterior.mean.shape == torch.Size([2, 1])
    assert mean.shape == torch.Size([2, 1])
    assert std.shape == torch.Size([2, 1])
    assert torch.isfinite(mean).all()
    assert torch.isfinite(std).all()
    assert bool((std >= 0).all())


def test_make_mll_targets_the_model_likelihood():
    train_X, train_Y, bounds = _training_data()
    model = GaussianMultiFidelityGP(
        train_X,
        train_Y,
        fidelity_spec=FidelitySpec(fidelity_features=(-1,)),
        bounds=bounds,
    )

    mll = model.make_mll()
    assert mll.model is model
    assert mll.likelihood is model.likelihood


@pytest.mark.parametrize(
    ("train_Y", "train_Yvar", "message"),
    [
        (torch.zeros(6), None, "train_Y must have shape"),
        (torch.zeros(6, 2), None, "train_Y must have shape"),
        (torch.zeros(6, 1), torch.zeros(6, 2), "train_Yvar must have the same shape"),
        (torch.zeros(6, 1), -torch.ones(6, 1), "train_Yvar must be finite and non-negative"),
    ],
)
def test_rejects_invalid_scalar_or_noise_contract(train_Y, train_Yvar, message):
    train_X, _, bounds = _training_data()
    with pytest.raises(ValueError, match=message):
        GaussianMultiFidelityGP(
            train_X,
            train_Y.double(),
            train_Yvar=None if train_Yvar is None else train_Yvar.double(),
            fidelity_spec=FidelitySpec(fidelity_features=(-1,)),
            bounds=bounds,
        )


def test_rejects_target_outside_bounds_through_shared_spec():
    train_X, train_Y, bounds = _training_data()
    with pytest.raises(ValueError, match="outside bounds"):
        GaussianMultiFidelityGP(
            train_X,
            train_Y,
            fidelity_spec=FidelitySpec(
                fidelity_features=(-1,), target_fidelities={-1: 2.0}
            ),
            bounds=bounds,
        )
