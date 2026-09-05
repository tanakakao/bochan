from __future__ import annotations

import torch

from bochan.api import ModelConfig, MultiOutputConfig
from bochan.api.configs import OptimizeConfig
from bochan.api.modeling.build import build_model
from bochan.models.multifidelity.optimization import (
    merge_target_fidelities_into_opt_config,
    prepare_continuous_fidelity_optimization,
)
from bochan.models.multifidelity.spec import FidelitySpec


def _continuous_data():
    train_X = torch.tensor(
        [
            [0.0, 0.25, 0.50],
            [0.2, 0.50, 0.75],
            [0.4, 1.00, 1.00],
            [0.6, 0.25, 1.00],
            [0.8, 0.50, 0.50],
            [1.0, 1.00, 0.75],
        ],
        dtype=torch.double,
    )
    y = (
        train_X[:, :1]
        + 0.2 * train_X[:, 1:2]
        + 0.1 * train_X[:, 2:3]
    )
    return train_X, y


def test_fidelity_spec_resolves_multiple_negative_dimensions():
    bounds = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        dtype=torch.double,
    )
    resolved = FidelitySpec(
        fidelity_features=(-2, -1),
        target_fidelities={-2: 1.0, -1: 0.9},
    ).resolve(d=3, bounds=bounds)

    assert resolved.fidelity_features == (1, 2)
    assert resolved.target_fidelities == {1: 1.0, 2: 0.9}


def test_continuous_multifidelity_gp_supports_two_fidelity_features():
    train_X, train_Y = _continuous_data()
    bundle = build_model(
        train_X,
        train_Y,
        ModelConfig(
            task_type="regression",
            model_type="multifidelity_gp",
            input_type="normal",
            model_kwargs={
                "fidelity_features": [-2, -1],
                "target_fidelities": {-2: 1.0, -1: 1.0},
            },
        ),
    )

    assert bundle.model.fidelity_features == (1, 2)
    assert bundle.model.target_fidelities == {1: 1.0, 2: 1.0}
    assert bundle.model._init_args["data_fidelities"] == [1, 2]
    posterior = bundle.model.posterior(train_X[:2])
    assert posterior.mean.shape == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()


def test_mixed_multifidelity_gp_supports_two_fidelity_features():
    train_X = torch.tensor(
        [
            [0.0, 0.0, 0.25, 0.50],
            [0.2, 1.0, 0.50, 0.75],
            [0.4, 0.0, 1.00, 1.00],
            [0.6, 1.0, 0.25, 1.00],
            [0.8, 0.0, 0.50, 0.50],
            [1.0, 1.0, 1.00, 0.75],
        ],
        dtype=torch.double,
    )
    train_Y = (
        train_X[:, :1]
        + 0.1 * train_X[:, 1:2]
        + 0.2 * train_X[:, 2:3]
        + 0.1 * train_X[:, 3:4]
    )
    bundle = build_model(
        train_X,
        train_Y,
        ModelConfig(
            task_type="regression",
            model_type="multifidelity_gp",
            input_type="mixed",
            cat_dims=[1],
            model_kwargs={
                "fidelity_features": [-2, -1],
                "target_fidelities": {-2: 1.0, -1: 1.0},
            },
        ),
    )

    assert bundle.model.fidelity_features == (2, 3)
    assert bundle.model.cat_dims == (1,)
    assert bundle.model.cont_dims == (0,)
    posterior = bundle.model.posterior(train_X[:2])
    assert torch.isfinite(posterior.mean).all()


def test_multioutput_models_share_multiple_fidelity_metadata():
    train_X, y1 = _continuous_data()
    train_Y = torch.cat([y1, 1.5 - y1], dim=-1)
    bundle = build_model(
        train_X,
        train_Y,
        ModelConfig(
            task_type="regression",
            model_type="multifidelity_gp",
            input_type="normal",
            model_kwargs={
                "fidelity_features": [-2, -1],
                "target_fidelities": {-2: 1.0, -1: 1.0},
            },
            multi_output_config=MultiOutputConfig(),
        ),
    )

    assert len(bundle.model.models) == 2
    assert all(model.fidelity_features == (1, 2) for model in bundle.model.models)
    assert all(model.target_fidelities == {1: 1.0, 2: 1.0} for model in bundle.model.models)


def test_target_fixed_optimization_merges_all_fidelity_dimensions():
    train_X, train_Y = _continuous_data()
    bundle = build_model(
        train_X,
        train_Y,
        ModelConfig(
            task_type="regression",
            model_type="multifidelity_gp",
            model_kwargs={
                "fidelity_features": [-2, -1],
                "target_fidelities": {-2: 1.0, -1: 0.8},
            },
        ),
    )

    resolved = merge_target_fidelities_into_opt_config(
        OptimizeConfig(),
        model=bundle.model,
    )
    assert resolved.fixed_features == {1: 1.0, 2: 0.8}


def test_joint_multidimensional_fidelity_search_is_supported_in_phase60():
    train_X, train_Y = _continuous_data()
    bundle = build_model(
        train_X,
        train_Y,
        ModelConfig(
            task_type="regression",
            model_type="multifidelity_gp",
            model_kwargs={
                "fidelity_features": [-2, -1],
                "target_fidelities": {-2: 1.0, -1: 1.0},
            },
        ),
    )
    bounds = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        dtype=torch.double,
    )

    resolved = prepare_continuous_fidelity_optimization(
        OptimizeConfig(optimize_fidelity=True),
        model=bundle.model,
        bounds=bounds,
    )

    assert resolved.optimize_fidelity is True
    assert resolved.fixed_features is None
    assert resolved.fixed_features_list is None
