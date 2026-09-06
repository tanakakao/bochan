from __future__ import annotations

import pytest
import torch
from botorch.models import SingleTaskGP

from bochan.api.acquisition.multifidelity import _resolve_cost_aware_utility
from bochan.api.acquisition.multifidelity_momf import _resolve_cost_call
from bochan.models.multifidelity import (
    FidelityCostConfig,
    build_fidelity_cost_utility,
    build_learned_fidelity_cost_model,
    evaluate_fidelity_cost_mean,
)


class _ObjectiveModel:
    fidelity_features = (1,)


def _cost_data():
    X = torch.tensor(
        [
            [0.0, 0.1],
            [0.2, 0.3],
            [0.5, 0.6],
            [0.8, 1.0],
        ],
        dtype=torch.double,
    )
    cost = (1.0 + 4.0 * X[:, 1].square()).unsqueeze(-1)
    return X, cost


def test_learned_gp_requires_complete_cost_observations():
    X, cost = _cost_data()
    with pytest.raises(ValueError, match="both train_X and train_cost"):
        FidelityCostConfig(kind="learned_gp", train_X=X)
    with pytest.raises(ValueError, match="both train_X and train_cost"):
        FidelityCostConfig(kind="learned_gp", train_cost=cost)
    with pytest.raises(ValueError, match="requires cost_model or train_X/train_cost"):
        FidelityCostConfig(kind="learned_gp")


def test_learned_gp_validates_positive_aligned_cost_data():
    X, cost = _cost_data()
    with pytest.raises(ValueError, match="same number of rows"):
        build_learned_fidelity_cost_model(
            FidelityCostConfig(
                kind="learned_gp",
                train_X=X,
                train_cost=cost[:-1],
                fit_model=False,
            ),
            d=2,
        )
    bad_cost = cost.clone()
    bad_cost[0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        build_learned_fidelity_cost_model(
            FidelityCostConfig(
                kind="learned_gp",
                train_X=X,
                train_cost=bad_cost,
                fit_model=False,
            ),
            d=2,
        )


def test_learned_gp_builds_single_task_cost_surrogate():
    X, cost = _cost_data()
    config = FidelityCostConfig(
        kind="learned_gp",
        train_X=X,
        train_cost=cost,
        fit_model=False,
    )
    model = build_learned_fidelity_cost_model(config, d=2)

    assert isinstance(model, SingleTaskGP)
    posterior = model.posterior(X[:2])
    assert posterior.mean.shape == (2, 1)
    assert posterior.mean.dtype == X.dtype


def test_learned_log_cost_utility_maps_predictions_to_positive_cost():
    X, cost = _cost_data()
    config = FidelityCostConfig(
        kind="learned_gp",
        train_X=X,
        train_cost=cost,
        fit_model=False,
        log_cost=True,
        use_mean=False,
        min_cost=0.05,
    )
    cost_model, utility = build_fidelity_cost_utility(
        config,
        fidelity_features=(1,),
        d=2,
    )
    query = torch.tensor([[[0.3, 0.4], [0.7, 0.9]]], dtype=torch.double)
    mean_cost = evaluate_fidelity_cost_mean(
        cost_model,
        utility,
        query,
        min_cost=config.min_cost,
    )

    assert mean_cost.shape == (1, 2, 1)
    assert bool((mean_cost >= config.min_cost).all())
    assert utility._use_mean is False


def test_learned_gp_can_reuse_prebuilt_cost_model():
    X, cost = _cost_data()
    base = FidelityCostConfig(
        kind="learned_gp",
        train_X=X,
        train_cost=cost,
        fit_model=False,
    )
    model = build_learned_fidelity_cost_model(base, d=2)
    reused = FidelityCostConfig(
        kind="learned_gp",
        cost_model=model,
        log_cost=True,
        use_mean=True,
    )

    resolved = build_learned_fidelity_cost_model(reused, d=2)
    assert resolved is model


def test_mf_acquisition_resolver_accepts_learned_cost_gp():
    X, cost = _cost_data()
    config = FidelityCostConfig(
        kind="learned_gp",
        train_X=X,
        train_cost=cost,
        fit_model=False,
    )
    kwargs = {"cost_config": config}
    cost_model, utility = _resolve_cost_aware_utility(_ObjectiveModel(), kwargs, d=2)

    assert isinstance(cost_model, SingleTaskGP)
    assert kwargs["cost_aware_utility"] is utility


def test_momf_uses_posterior_mean_for_learned_cost_gp():
    X, cost = _cost_data()
    config = FidelityCostConfig(
        kind="learned_gp",
        train_X=X,
        train_cost=cost,
        fit_model=False,
        log_cost=True,
    )
    cost_call, cost_model = _resolve_cost_call(
        model=_ObjectiveModel(),
        d=2,
        kwargs={"cost_config": config},
    )
    query = torch.tensor([[[0.25, 0.4], [0.75, 0.9]]], dtype=torch.double)
    predicted = cost_call(query)

    assert isinstance(cost_model, SingleTaskGP)
    assert predicted.shape == (1, 2, 1)
    assert bool((predicted > 0).all())
