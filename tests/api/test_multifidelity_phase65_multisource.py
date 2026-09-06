from __future__ import annotations

import pytest
import torch

from bochan.api import AcquisitionConfig, ModelConfig, OptimizeConfig
from bochan.api.acquisition.multifidelity import build_multifidelity_acquisition
from bochan.api.acquisition.service import resolve_acquisition_class
from bochan.api.modeling.build import build_model
from bochan.models.multifidelity import (
    FidelityCostConfig,
    GaussianMultiSourceGP,
    InformationSourceSpec,
    build_fidelity_cost_utility,
)
from bochan.models.multifidelity.optimization import (
    enumerate_discrete_fidelities_into_opt_config,
    merge_target_fidelities_into_opt_config,
)


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor(
        [
            [0.05, 0.0],
            [0.20, 1.0],
            [0.35, 2.0],
            [0.50, 0.0],
            [0.65, 1.0],
            [0.80, 2.0],
            [0.95, 2.0],
        ],
        dtype=torch.double,
    )
    x = X[:, 0]
    source = X[:, 1]
    bias = torch.where(source == 0, -0.25, torch.where(source == 1, 0.12, 0.0))
    Y = (torch.sin(3.0 * x) + bias).unsqueeze(-1)
    return X, Y


def test_information_source_spec_resolves_negative_index_and_names() -> None:
    X, _ = _data()
    resolved = InformationSourceSpec(
        source_feature=-1,
        source_values=(0, 1, 2),
        target_source=2,
        source_names={0: "simulation_a", 1: "simulation_b", 2: "experiment"},
    ).resolve(d=2, train_X=X)

    assert resolved.source_feature == 1
    assert resolved.source_values == (0, 1, 2)
    assert resolved.target_source == 2
    assert resolved.source_names[2] == "experiment"


def test_information_source_spec_can_infer_observed_sources() -> None:
    X, _ = _data()
    resolved = InformationSourceSpec(source_feature=-1, target_source=2).resolve(
        d=2,
        train_X=X,
    )
    assert resolved.source_values == (0, 1, 2)


def test_multisource_gp_uses_icm_task_axis_and_mf_compatibility_metadata() -> None:
    X, Y = _data()
    model = GaussianMultiSourceGP(
        X,
        Y,
        source_feature=-1,
        source_values=(0, 1, 2),
        target_source=2,
        rank=2,
    )

    assert model.information_source_feature == 1
    assert model.information_source_values == (0, 1, 2)
    assert model.target_information_source == 2
    assert model.fidelity_mode == "information_source"
    assert model.fidelity_features == (1,)
    assert model.target_fidelities == {1: 2.0}

    posterior = model.posterior(torch.tensor([[0.4, 0.0], [0.4, 2.0]], dtype=torch.double))
    assert posterior.mean.shape[-2:] == (2, 1)
    assert torch.isfinite(posterior.mean).all()


def test_multisource_model_type_alias_builds_from_model_config() -> None:
    X, Y = _data()
    bundle = build_model(
        X,
        Y,
        ModelConfig(
            task_type="regression",
            model_type="multisource_gp",
            input_type="normal",
            model_kwargs={
                "source_feature": -1,
                "source_values": [0, 1, 2],
                "target_source": 2,
            },
        ),
    )
    assert isinstance(bundle.model, GaussianMultiSourceGP)
    assert bundle.model.target_information_source == 2


def test_multisource_gp_rejects_noninteger_and_undeclared_sources() -> None:
    X, Y = _data()
    X_noninteger = X.clone()
    X_noninteger[0, -1] = 0.5
    with pytest.raises(ValueError, match="integer task/source id"):
        GaussianMultiSourceGP(
            X_noninteger,
            Y,
            source_feature=-1,
            source_values=(0, 1, 2),
            target_source=2,
        )

    with pytest.raises(ValueError, match="not declared"):
        GaussianMultiSourceGP(
            X,
            Y,
            source_feature=-1,
            source_values=(0, 1),
            target_source=1,
        )


def test_discrete_source_cost_maps_source_ids_without_imposing_order() -> None:
    config = FidelityCostConfig(
        kind="discrete_source",
        source_feature=-1,
        source_costs={0: 1.0, 1: 4.0, 2: 20.0},
    )
    cost_model, utility = build_fidelity_cost_utility(
        config,
        fidelity_features=(1,),
        d=2,
    )
    X = torch.tensor(
        [[[0.2, 0.0], [0.4, 2.0], [0.8, 1.0]]],
        dtype=torch.double,
    )
    expected = torch.tensor([[[1.0], [20.0], [4.0]]], dtype=torch.double)

    assert torch.allclose(cost_model(X), expected)
    assert utility.cost_model is cost_model


def test_discrete_source_cost_rejects_unknown_source_at_evaluation() -> None:
    cost_model, _ = build_fidelity_cost_utility(
        FidelityCostConfig(
            kind="discrete_source",
            source_feature=1,
            source_costs={0: 1.0, 2: 20.0},
        ),
        fidelity_features=(1,),
        d=2,
    )
    with pytest.raises(ValueError, match="No source cost configured"):
        cost_model(torch.tensor([[0.2, 1.0]], dtype=torch.double))


def test_existing_discrete_fidelity_optimizer_enumerates_information_sources() -> None:
    X, Y = _data()
    model = GaussianMultiSourceGP(
        X,
        Y,
        source_feature=-1,
        source_values=(0, 1, 2),
        target_source=2,
    )
    config = OptimizeConfig(fidelity_values=[0.0, 1.0, 2.0])
    bounds = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.double)

    resolved = enumerate_discrete_fidelities_into_opt_config(
        config,
        model=model,
        bounds=bounds,
    )
    assert resolved.fixed_features_list == [{1: 0.0}, {1: 1.0}, {1: 2.0}]

    target = merge_target_fidelities_into_opt_config(
        OptimizeConfig(),
        model=model,
    )
    assert target.fixed_features == {1: 2.0}


class _RoutingOptimizer:
    def _check_fitted(self) -> None:
        return None

    def _acquisition_routing_context(self):
        return "regression", "multisource_gp", False


def test_mfkg_and_mfmes_route_to_multisource_model() -> None:
    optimizer = _RoutingOptimizer()
    for name in ("mfkg", "mfmes"):
        resolved = resolve_acquisition_class(
            optimizer,
            AcquisitionConfig(name=name),
        )
        assert resolved.acqf_factory is build_multifidelity_acquisition
