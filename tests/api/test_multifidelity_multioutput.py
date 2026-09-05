from __future__ import annotations

import pytest
import torch

from bochan.api import ModelConfig, MultiOutputConfig, OptimizeConfig
from bochan.api.modeling import build_model
from bochan.models.multifidelity import shared_multifidelity_metadata
from bochan.models.multifidelity.optimization import (
    enumerate_discrete_fidelities_into_opt_config,
    merge_target_fidelities_into_opt_config,
    prepare_continuous_fidelity_optimization,
    target_fidelity_fixed_features,
)


def _training_data():
    train_X = torch.tensor(
        [
            [0.0, 0.25],
            [0.2, 0.50],
            [0.4, 1.00],
            [0.6, 0.25],
            [0.8, 0.50],
            [1.0, 1.00],
        ],
        dtype=torch.double,
    )
    x = train_X[:, :1]
    fidelity = train_X[:, 1:2]
    train_Y = torch.cat(
        [
            x + 0.2 * fidelity,
            (1.0 - x) + 0.1 * fidelity,
        ],
        dim=-1,
    )
    return train_X, train_Y


def _config(**model_kwargs):
    kwargs = {
        "fidelity_features": [-1],
        "target_fidelities": {-1: 1.0},
    }
    kwargs.update(model_kwargs)
    return ModelConfig(
        task_type="regression",
        model_type="multifidelity_gp",
        input_type="normal",
        model_kwargs=kwargs,
        multi_output_config=MultiOutputConfig(),
    )


def test_builds_independent_multifidelity_model_list():
    train_X, train_Y = _training_data()
    bundle = build_model(train_X, train_Y, _config())

    assert bundle.metadata["multi_output"] is True
    assert len(bundle.model.models) == 2
    assert all(tuple(model.fidelity_features) == (1,) for model in bundle.model.models)
    assert all(model.target_fidelities == {1: 1.0} for model in bundle.model.models)

    metadata = shared_multifidelity_metadata(bundle.model.models)
    assert metadata is not None
    assert metadata["multi_output_fidelity"] == "independent"
    assert metadata["fidelity_features"] == (1,)
    assert metadata["target_fidelities"] == {1: 1.0}


def test_model_list_target_fidelity_is_inferred_from_submodels():
    train_X, train_Y = _training_data()
    bundle = build_model(train_X, train_Y, _config())

    assert target_fidelity_fixed_features(bundle.model) == {1: 1.0}

    resolved = merge_target_fidelities_into_opt_config(
        OptimizeConfig(),
        model=bundle.model,
    )
    assert resolved.fixed_features == {1: 1.0}


def test_model_list_supports_discrete_fidelity_search():
    train_X, train_Y = _training_data()
    bundle = build_model(train_X, train_Y, _config())
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

    resolved = enumerate_discrete_fidelities_into_opt_config(
        OptimizeConfig(fidelity_values=[0.25, 0.5, 1.0]),
        model=bundle.model,
        bounds=bounds,
    )

    assert resolved.fixed_features_list == [
        {1: 0.25},
        {1: 0.5},
        {1: 1.0},
    ]


def test_model_list_supports_continuous_fidelity_search():
    train_X, train_Y = _training_data()
    bundle = build_model(train_X, train_Y, _config())
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

    resolved = prepare_continuous_fidelity_optimization(
        OptimizeConfig(optimize_fidelity=True),
        model=bundle.model,
        bounds=bounds,
    )
    resolved = merge_target_fidelities_into_opt_config(
        resolved,
        model=bundle.model,
    )

    assert resolved.optimize_fidelity is True
    assert resolved.fixed_features is None


def test_shared_contract_rejects_mismatched_targets():
    class _Model:
        fidelity_features = (1,)
        input_mode = "continuous"
        cat_dims = ()

        def __init__(self, target):
            self.target_fidelities = {1: target}

    with pytest.raises(ValueError, match="same target_fidelities"):
        shared_multifidelity_metadata([_Model(1.0), _Model(0.8)])


def test_shared_contract_rejects_partial_multifidelity_outputs():
    class _MF:
        fidelity_features = (1,)
        target_fidelities = {1: 1.0}
        input_mode = "continuous"
        cat_dims = ()

    class _Plain:
        pass

    with pytest.raises(ValueError, match="every output"):
        shared_multifidelity_metadata([_MF(), _Plain()])
