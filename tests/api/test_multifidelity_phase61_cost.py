from __future__ import annotations

import pytest
import torch

from bochan.api.acquisition.multifidelity import _resolve_cost_aware_utility
from bochan.api.acquisition.multifidelity_momf import _resolve_cost_call
from bochan.models.multifidelity import FidelityCostConfig, build_fidelity_cost_utility


class _Model:
    fidelity_features = (1,)


def test_affine_cost_contract_remains_backward_compatible():
    config = FidelityCostConfig(
        fixed_cost=2.0,
        fidelity_weights={-1: 3.0},
    )
    cost_model, utility = build_fidelity_cost_utility(
        config,
        fidelity_features=(1,),
        d=2,
    )
    X = torch.tensor([[0.2, 0.5]], dtype=torch.double)

    assert torch.allclose(cost_model(X), torch.tensor([3.5], dtype=torch.double))
    assert utility.cost_model is cost_model


def test_fixed_cost_is_independent_of_candidate_and_preserves_dtype():
    config = FidelityCostConfig(kind="fixed", fixed_cost=4.25)
    cost_model, utility = build_fidelity_cost_utility(
        config,
        fidelity_features=(1,),
        d=2,
    )
    X = torch.tensor(
        [[[0.1, 0.2], [0.8, 1.0]]],
        dtype=torch.double,
    )
    cost = cost_model(X)

    assert cost.shape == (1, 2, 1)
    assert cost.dtype == X.dtype
    assert torch.allclose(cost, torch.full((1, 2, 1), 4.25, dtype=X.dtype))
    assert utility.cost_model is cost_model


def test_callable_cost_supports_nonlinear_fidelity_cost():
    config = FidelityCostConfig(
        kind="callable",
        cost_callable=lambda X: torch.exp(2.0 * X[..., -1]),
    )
    cost_model, utility = build_fidelity_cost_utility(
        config,
        fidelity_features=(1,),
        d=2,
    )
    X = torch.tensor(
        [[[0.1, 0.0], [0.8, 0.5]]],
        dtype=torch.double,
    )
    cost = cost_model(X)

    assert cost.shape == (1, 2, 1)
    assert torch.allclose(cost.squeeze(-1), torch.exp(2.0 * X[..., -1]))
    assert utility.cost_model is cost_model


def test_callable_cost_scalar_is_broadcast_over_q_batch():
    config = FidelityCostConfig(kind="callable", cost_callable=lambda X: 3.0)
    cost_model, _ = build_fidelity_cost_utility(
        config,
        fidelity_features=(1,),
        d=2,
    )
    X = torch.zeros(2, 3, 2, dtype=torch.double)

    assert cost_model(X).shape == (2, 3, 1)
    assert torch.allclose(cost_model(X), torch.full((2, 3, 1), 3.0, dtype=X.dtype))


def test_callable_cost_rejects_invalid_shape_and_nonfinite_values():
    X = torch.zeros(2, 3, 2, dtype=torch.double)
    bad_shape, _ = build_fidelity_cost_utility(
        FidelityCostConfig(
            kind="callable",
            cost_callable=lambda X: torch.zeros(2, 2, dtype=X.dtype),
        ),
        fidelity_features=(1,),
        d=2,
    )
    with pytest.raises(ValueError, match="cost_callable must return"):
        bad_shape(X)

    nonfinite, _ = build_fidelity_cost_utility(
        FidelityCostConfig(
            kind="callable",
            cost_callable=lambda X: torch.full(X.shape[:-1], float("nan"), dtype=X.dtype),
        ),
        fidelity_features=(1,),
        d=2,
    )
    with pytest.raises(ValueError, match="finite"):
        nonfinite(X)


def test_cost_kind_specific_validation_is_explicit():
    with pytest.raises(ValueError, match="requires a callable"):
        FidelityCostConfig(kind="callable")
    with pytest.raises(ValueError, match="only valid for kind='affine'"):
        FidelityCostConfig(kind="fixed", fidelity_weights={-1: 1.0})
    with pytest.raises(ValueError, match="must be 'affine', 'fixed', or 'callable'"):
        FidelityCostConfig(kind="learned")


def test_mf_acquisition_resolver_accepts_fixed_and_callable_costs():
    for config in (
        FidelityCostConfig(kind="fixed", fixed_cost=2.0),
        FidelityCostConfig(
            kind="callable",
            cost_callable=lambda X: 1.0 + X[..., -1] ** 2,
        ),
    ):
        kwargs = {"cost_config": config}
        cost_model, utility = _resolve_cost_aware_utility(_Model(), kwargs, d=2)
        assert cost_model is not None
        assert utility is not None
        assert kwargs["cost_aware_utility"] is utility


def test_momf_cost_call_uses_generalized_known_cost_builder():
    X = torch.tensor([[[0.2, 0.25], [0.8, 1.0]]], dtype=torch.double)
    for config, expected in (
        (FidelityCostConfig(kind="fixed", fixed_cost=2.5), torch.full((1, 2, 1), 2.5)),
        (
            FidelityCostConfig(
                kind="callable",
                cost_callable=lambda X: 1.0 + X[..., -1],
            ),
            1.0 + X[..., -1:],
        ),
    ):
        cost_call, cost_model = _resolve_cost_call(
            model=_Model(),
            d=2,
            kwargs={"cost_config": config},
        )
        actual = cost_call(X)
        assert cost_model is not None
        assert torch.allclose(actual, expected.to(dtype=X.dtype))
