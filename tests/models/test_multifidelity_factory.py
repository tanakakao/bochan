from __future__ import annotations

import pytest
import torch

from bochan.models.multifidelity import FidelitySpec, create_fidelity_surrogate
from bochan.models.regression.gaussian import (
    GaussianMixedMultiFidelityGP,
    GaussianMultiFidelityGP,
)


def _continuous_data():
    train_X = torch.tensor(
        [
            [0.0, 0.25],
            [0.2, 0.50],
            [0.5, 1.00],
            [0.8, 0.50],
            [1.0, 1.00],
        ],
        dtype=torch.double,
    )
    train_Y = torch.sin(train_X[:, :1] * 2.0) + 0.1 * train_X[:, 1:2]
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    return train_X, train_Y, bounds


def _mixed_data():
    train_X = torch.tensor(
        [
            [0.0, 0.0, 0.25],
            [0.2, 1.0, 0.50],
            [0.5, 0.0, 1.00],
            [0.8, 1.0, 0.50],
            [1.0, 0.0, 1.00],
            [0.6, 1.0, 1.00],
        ],
        dtype=torch.double,
    )
    train_Y = train_X[:, :1] + 0.5 * train_X[:, 1:2] + 0.1 * train_X[:, 2:3]
    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.double)
    return train_X, train_Y, bounds


def test_dispatches_continuous_surrogate():
    train_X, train_Y, bounds = _continuous_data()
    model = create_fidelity_surrogate(
        train_X,
        train_Y,
        input_mode="continuous",
        fidelity_spec=FidelitySpec(
            fidelity_features=(-1,), target_fidelities={-1: 1.0}
        ),
        bounds=bounds,
    )

    assert isinstance(model, GaussianMultiFidelityGP)
    assert not isinstance(model, GaussianMixedMultiFidelityGP)
    assert model.fidelity_metadata == {
        "fidelity_mode": "feature",
        "fidelity_features": [1],
        "target_fidelities": {1: 1.0},
        "input_mode": "continuous",
        "cat_dims": [],
    }


def test_normal_alias_dispatches_continuous_surrogate():
    train_X, train_Y, bounds = _continuous_data()
    model = create_fidelity_surrogate(
        train_X,
        train_Y,
        input_mode="normal",
        fidelity_spec=FidelitySpec(fidelity_features=(-1,)),
        bounds=bounds,
    )

    assert isinstance(model, GaussianMultiFidelityGP)
    assert model.input_mode == "continuous"


def test_dispatches_mixed_surrogate_and_preserves_metadata():
    train_X, train_Y, bounds = _mixed_data()
    model = create_fidelity_surrogate(
        train_X,
        train_Y,
        input_mode="mixed",
        cat_dims=(1,),
        fidelity_spec=FidelitySpec(
            fidelity_features=(-1,), target_fidelities={-1: 1.0}
        ),
        bounds=bounds,
    )

    assert isinstance(model, GaussianMixedMultiFidelityGP)
    assert model.cat_dims == (1,)
    assert model.cont_dims == (0,)
    assert model.fidelity_metadata == {
        "fidelity_mode": "feature",
        "fidelity_features": [2],
        "target_fidelities": {2: 1.0},
        "input_mode": "mixed",
        "cat_dims": [1],
    }


def test_forwards_known_noise_to_selected_model():
    train_X, train_Y, bounds = _continuous_data()
    train_Yvar = torch.full_like(train_Y, 0.02)
    model = create_fidelity_surrogate(
        train_X,
        train_Y,
        train_Yvar=train_Yvar,
        fidelity_spec=FidelitySpec(fidelity_features=(-1,)),
        bounds=bounds,
    )

    assert model.train_Yvar_raw is not None
    assert torch.equal(model.train_Yvar_raw, train_Yvar)


def test_rejects_cat_dims_for_continuous_mode():
    train_X, train_Y, bounds = _continuous_data()
    with pytest.raises(ValueError, match="cat_dims is only valid"):
        create_fidelity_surrogate(
            train_X,
            train_Y,
            input_mode="continuous",
            cat_dims=(0,),
            fidelity_spec=FidelitySpec(fidelity_features=(-1,)),
            bounds=bounds,
        )


def test_requires_cat_dims_for_mixed_mode():
    train_X, train_Y, bounds = _mixed_data()
    with pytest.raises(ValueError, match="cat_dims is required"):
        create_fidelity_surrogate(
            train_X,
            train_Y,
            input_mode="mixed",
            fidelity_spec=FidelitySpec(fidelity_features=(-1,)),
            bounds=bounds,
        )


def test_rejects_unknown_input_mode():
    train_X, train_Y, bounds = _continuous_data()
    with pytest.raises(ValueError, match="input_mode must be one of"):
        create_fidelity_surrogate(
            train_X,
            train_Y,
            input_mode="other",
            fidelity_spec=FidelitySpec(fidelity_features=(-1,)),
            bounds=bounds,
        )
