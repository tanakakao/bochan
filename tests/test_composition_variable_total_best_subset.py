from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from bochan.api import OptimizeConfig
from bochan.composition import CompositionTransformer
from bochan.tabular import TabularBayesianOptimizer
from bochan.tabular.composition.variable_total_support import (
    CompositionVariableTotalDecisionBridge,
    prepare_variable_total_best_subset_config,
)


def _transformer(representation: str) -> CompositionTransformer:
    transformer = CompositionTransformer(
        elements=["Al", "Ti", "V", "Nb"],
        representation=representation,
        reference_element="Nb" if representation == "alr" else None,
        pseudocount=1e-8,
        prefix="alloy",
    )
    transformer.fit(["Al4Ti3V2Nb", "Al3Ti2V3Nb2"])
    return transformer


def _layout(transformer: CompositionTransformer) -> tuple[str, ...]:
    return (
        "temperature",
        *transformer.representation_feature_names_,
        "alloy__total",
        "pressure",
    )


def _model_bounds(transformer: CompositionTransformer) -> torch.Tensor:
    width = len(transformer.representation_feature_names_)
    coordinate_lower = [0.0] * width if transformer.representation == "fractions" else [-8.0] * width
    coordinate_upper = [1.0] * width if transformer.representation == "fractions" else [8.0] * width
    return torch.tensor(
        [
            [800.0, *coordinate_lower, 40.0, 1.0],
            [1200.0, *coordinate_upper, 90.0, 5.0],
        ],
        dtype=torch.double,
    )


@pytest.mark.parametrize("representation", ["fractions", "clr", "alr", "ilr"])
def test_variable_total_bridge_maps_absolute_amounts_to_model_space(
    representation: str,
) -> None:
    transformer = _transformer(representation)
    bridge = CompositionVariableTotalDecisionBridge.from_transformer(
        transformer,
        _layout(transformer),
        total_feature="alloy__total",
    )
    raw = torch.tensor(
        [[900.0, 20.0, 15.0, 10.0, 5.0, 2.0]],
        dtype=torch.double,
        requires_grad=True,
    )

    model = bridge.decision_to_model(raw)
    expected_coordinates = transformer.simplex_transform_.transform(
        np.asarray([[0.4, 0.3, 0.2, 0.1]], dtype=float)
    )
    total_index = bridge.model_feature_names.index("alloy__total")

    assert bridge.decision_feature_names == (
        "temperature",
        "alloy__amount__Al",
        "alloy__amount__Ti",
        "alloy__amount__V",
        "alloy__amount__Nb",
        "pressure",
    )
    assert model[0, total_index].item() == pytest.approx(50.0)
    np.testing.assert_allclose(
        model[
            0,
            bridge.base.coordinate_start : bridge.base.coordinate_stop,
        ]
        .detach()
        .numpy(),
        expected_coordinates[0],
        rtol=1e-10,
        atol=1e-10,
    )

    restored = bridge.model_to_decision(model)
    torch.testing.assert_close(restored, raw, rtol=1e-7, atol=1e-7)

    model.square().sum().backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert float(raw.grad[..., list(bridge.amount_indices)].abs().sum()) > 0.0


def test_variable_total_bridge_uses_absolute_component_bounds() -> None:
    transformer = _transformer("ilr")
    bridge = CompositionVariableTotalDecisionBridge.from_transformer(
        transformer,
        _layout(transformer),
        total_feature="alloy__total",
    )
    bounds = bridge.decision_bounds(
        _model_bounds(transformer),
        component_bounds={
            "Al": (5.0, 60.0),
            "Ti": (0.0, 50.0),
            "V": (0.0, 40.0),
            "Nb": (0.0, 30.0),
        },
        total_bounds=(40.0, 90.0),
    )

    assert bounds.shape == (2, 6)
    assert bounds[:, 0].tolist() == [800.0, 1200.0]
    assert bounds[:, -1].tolist() == [1.0, 5.0]
    torch.testing.assert_close(
        bounds[:, 1:5],
        torch.tensor(
            [
                [5.0, 0.0, 0.0, 0.0],
                [60.0, 50.0, 40.0, 30.0],
            ],
            dtype=torch.double,
        ),
    )


def _site(representation: str = "ilr") -> dict[str, object]:
    return {
        "column": "formula",
        "elements": ("Al", "Ti", "V", "Nb"),
        "representation": representation,
        "normalization": "atomic_fraction",
        "reference_element": "Nb" if representation == "alr" else None,
        "pseudocount": 1e-8,
        "prefix": "alloy",
        "total": 65.0,
        "variable_total": True,
        "total_bounds": (40.0, 90.0),
        "total_feature": "alloy__total",
        "bounds": {
            "Al": (5.0, 70.0),
            "Ti": (0.0, 70.0),
            "V": (0.0, 70.0),
            "Nb": (0.0, 70.0),
        },
        "steps": {},
        "min_components": 3,
        "max_components": 3,
        "required_components": ("Al",),
        "forbidden_components": (),
        "support_selection": "best_subset",
        "best_subset_strategy": "exact",
        "best_subset_max_combinations": 20,
    }


@pytest.mark.parametrize("representation", ["fractions", "clr", "alr", "ilr"])
def test_variable_total_config_uses_amount_support_and_total_constraints(
    representation: str,
) -> None:
    transformer = _transformer(representation)
    bridge, config, bounds = prepare_variable_total_best_subset_config(
        OptimizeConfig(),
        site_name="alloy",
        site_config=_site(representation),
        transformer=transformer,
        model_feature_names=_layout(transformer),
        model_bounds=_model_bounds(transformer),
        dtype=torch.double,
    )

    assert bridge.amount_indices == (1, 2, 3, 4)
    assert bounds.shape == (2, 6)
    repair = config.repair_config
    assert repair is not None
    assert repair.support_selection == "best_subset"
    assert repair.k == 2
    assert tuple(repair.comp_idx or ()) == (2, 3, 4)

    # Required Al stays outside the optional k-sparse group. The site total is
    # no longer a model decision dimension: it is the sum of the four amounts.
    assert bridge.total_feature_name not in bridge.decision_feature_names
    amount_indices = set(bridge.amount_indices)
    total_constraints = [
        item
        for item in config.inequality_constraints or ()
        if set(torch.as_tensor(item[0]).reshape(-1).tolist()) == amount_indices
    ]
    assert len(total_constraints) >= 2


def test_variable_total_config_expands_total_feature_constraints() -> None:
    transformer = _transformer("ilr")
    bridge, config, _bounds = prepare_variable_total_best_subset_config(
        OptimizeConfig(
            equality_constraints=[
                (["alloy__total", "pressure"], [1.0, -10.0], 0.0)
            ]
        ),
        site_name="alloy",
        site_config=_site("ilr"),
        transformer=transformer,
        model_feature_names=_layout(transformer),
        model_bounds=_model_bounds(transformer),
        dtype=torch.double,
    )

    equality = list(config.equality_constraints or ())
    assert len(equality) == 1
    indices, coefficients, rhs = equality[0]
    assert set(torch.as_tensor(indices).reshape(-1).tolist()) == {
        *bridge.amount_indices,
        bridge.process_index_map[bridge.model_feature_names.index("pressure")],
    }
    assert coefficients.shape[-1] == 5
    assert rhs == pytest.approx(0.0)


def test_variable_total_best_subset_rejects_component_steps_for_now() -> None:
    transformer = _transformer("ilr")
    site = _site("ilr")
    site["steps"] = {"Al": 5.0}
    with pytest.raises(ValueError, match="step grids"):
        prepare_variable_total_best_subset_config(
            OptimizeConfig(),
            site_name="alloy",
            site_config=site,
            transformer=transformer,
            model_feature_names=_layout(transformer),
            model_bounds=_model_bounds(transformer),
            dtype=torch.double,
        )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Al": [20.0, 24.0, 28.0, 22.0, 30.0, 26.0, 18.0, 32.0],
            "Ti": [15.0, 18.0, 14.0, 20.0, 16.0, 22.0, 24.0, 12.0],
            "V": [10.0, 8.0, 12.0, 14.0, 9.0, 11.0, 16.0, 13.0],
            "Nb": [5.0, 10.0, 6.0, 8.0, 12.0, 7.0, 9.0, 15.0],
            "temperature": [850.0, 900.0, 950.0, 1000.0, 1050.0, 1100.0, 1150.0, 1200.0],
            "property": [0.8, 1.0, 1.1, 1.25, 1.4, 1.55, 1.7, 1.85],
        }
    )


def _optimizer(representation: str) -> TabularBayesianOptimizer:
    return TabularBayesianOptimizer(
        task_type="regression",
        model_type="base",
        fit_config={"maxiter": 32},
        input_cols=["Al", "Ti", "V", "Nb", "temperature"],
        target_cols="property",
        composition_sites={
            "alloy": {
                "element_columns": {
                    "Al": "Al",
                    "Ti": "Ti",
                    "V": "V",
                    "Nb": "Nb",
                },
                "representation": representation,
                "reference_element": "Nb" if representation == "alr" else None,
                "pseudocount": 1e-8,
                "total_bounds": [40.0, 90.0],
                "bounds": {
                    "Al": [5.0, 70.0],
                    "Ti": [0.0, 70.0],
                    "V": [0.0, 70.0],
                    "Nb": [0.0, 70.0],
                },
                "min_components": 3,
                "max_components": 3,
                "required_components": ["Al"],
                "support_selection": "best_subset",
                "best_subset_strategy": "exact",
                "best_subset_max_combinations": 20,
            }
        },
        bounds={"temperature": [800.0, 1250.0]},
    )


@pytest.mark.parametrize("representation", ["fractions", "clr", "alr", "ilr"])
def test_tabular_candidate_jointly_optimizes_variable_total_and_support(
    representation: str,
) -> None:
    optimizer = _optimizer(representation).fit(_frame())
    result = optimizer.candidate(
        acq_name="logei",
        q=1,
        num_restarts=2,
        raw_samples=16,
        optimizer_kwargs={
            "best_subset_strategy": "exact",
            "options": {"maxiter": 12, "batch_limit": 2},
        },
        return_result=True,
    )

    raw = result.raw_composition_candidates
    bridge = result.composition_raw_bridge
    amounts = bridge.amount_values(raw)
    total = float(amounts.sum().item())

    assert int((amounts > 1e-7).sum().item()) == 3
    assert amounts[..., 0].item() > 0.0
    assert 40.0 - 1e-6 <= total <= 90.0 + 1e-6

    total_index = optimizer.dataset.feature_names.index("alloy__total")
    assert result.candidates[..., total_index].item() == pytest.approx(total, abs=1e-6)
    assert torch.isfinite(result.candidates).all()
    assert torch.isfinite(torch.as_tensor(result.acq_value)).all()

    model_frame = optimizer.candidates_to_dataframe(result.candidates)
    restored = optimizer.inverse_compositions(model_frame, repair=True)
    restored_amounts = restored.loc[0, ["Al", "Ti", "V", "Nb"]].to_numpy(dtype=float)
    assert int((restored_amounts > 1e-7).sum()) == 3
    assert restored.loc[0, "Al"] > 0.0
    assert restored_amounts.sum() == pytest.approx(total, abs=1e-5)
    assert restored.loc[0, "alloy__total"] == pytest.approx(total, abs=1e-5)
    assert 800.0 <= float(restored.loc[0, "temperature"]) <= 1250.0
