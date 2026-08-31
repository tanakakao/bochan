from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from bochan.api import OptimizeConfig
from bochan.composition import CompositionTransformer
from bochan.tabular.composition.config import normalize_composition_sites
from bochan.tabular.composition.grid import (
    CompositionGridFinalPostprocess,
    CompositionVariableTotalGridFinalPostprocess,
)
from bochan.tabular.composition.logratio_support import (
    prepare_logratio_best_subset_config,
)
from bochan.tabular.composition.support import resolve_composition_best_subset
from bochan.tabular.composition.variable_total_support import (
    prepare_variable_total_best_subset_config,
)


def _fraction_site(**overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "column": "formula",
        "elements": ["Al", "Ti", "V", "Cr", "Nb"],
        "representation": "fractions",
        "total": 1.0,
        "min_components": 2,
        "max_components": 4,
        "required_components": [],
        "forbidden_components": ["Cr"],
        "support_selection": "best_subset",
        "best_subset_strategy": "exact",
        "bounds": {
            "Al": [0.0, 1.0],
            "Ti": [0.0, 1.0],
            "V": [0.0, 1.0],
            "Cr": [0.0, 1.0],
            "Nb": [0.0, 1.0],
        },
        "steps": {
            "Al": 0.1,
            "Ti": 0.1,
            "V": 0.1,
            "Nb": 0.1,
        },
    }
    raw.update(overrides)
    site = normalize_composition_sites({"alloy": raw})["alloy"]
    site.setdefault("variable_total", False)
    return site


def _fraction_transformer() -> SimpleNamespace:
    return SimpleNamespace(
        fitted_elements=("Al", "Ti", "V", "Cr", "Nb"),
        prefix="alloy",
    )


def _fraction_features() -> list[str]:
    return [
        "alloy__fraction__Al",
        "alloy__fraction__Ti",
        "alloy__fraction__V",
        "alloy__fraction__Cr",
        "alloy__fraction__Nb",
        "temperature",
    ]


def test_fixed_fraction_is_preserved_by_variable_cardinality_step_grid() -> None:
    config = resolve_composition_best_subset(
        OptimizeConfig(fixed_features={"alloy__fraction__Al": 0.2}),
        composition_sites={"alloy": _fraction_site()},
        composition_transformers={"alloy": _fraction_transformer()},
        feature_names=_fraction_features(),
    )

    callback = config.final_candidate_postprocess
    assert isinstance(callback, CompositionGridFinalPostprocess)
    assert config.fixed_features is not None
    assert config.fixed_features["alloy__fraction__Al"] == pytest.approx(0.2)
    assert callback.minimum_cardinality == 2
    assert callback.exact_k == 4

    candidate = torch.tensor(
        [[0.2, 0.47, 0.33, 0.0, 0.0, 913.0]],
        dtype=torch.double,
    )
    projected = callback(candidate)
    fractions = projected[0, :5]

    assert fractions[0].item() == pytest.approx(0.2, abs=1e-10)
    assert fractions.sum().item() == pytest.approx(1.0, abs=1e-10)
    assert int((fractions > 1e-10).sum().item()) == 3
    assert fractions[3].item() == pytest.approx(0.0)
    assert fractions[4].item() == pytest.approx(0.0)


def test_fixed_fraction_off_step_grid_is_rejected() -> None:
    with pytest.raises(ValueError, match="not on the configured step grid"):
        resolve_composition_best_subset(
            OptimizeConfig(fixed_features={"alloy__fraction__Al": 0.25}),
            composition_sites={"alloy": _fraction_site()},
            composition_transformers={"alloy": _fraction_transformer()},
            feature_names=_fraction_features(),
        )


def _logratio_transformer(representation: str) -> CompositionTransformer:
    transformer = CompositionTransformer(
        elements=["Al", "Ti", "V", "Cr", "Nb"],
        representation=representation,
        reference_element="Nb" if representation == "alr" else None,
        pseudocount=1e-8,
        prefix="alloy",
    )
    transformer.fit(["AlTiVNb", "Al2TiV", "AlTiCrNb"])
    return transformer


def _logratio_layout(transformer: CompositionTransformer) -> list[str]:
    return [
        "temperature",
        *transformer.representation_feature_names_,
        "pressure",
    ]


def _logratio_bounds(transformer: CompositionTransformer) -> torch.Tensor:
    width = len(transformer.representation_feature_names_)
    return torch.tensor(
        [
            [800.0, *([-8.0] * width), 1.0],
            [1200.0, *([8.0] * width), 5.0],
        ],
        dtype=torch.double,
    )


@pytest.mark.parametrize("representation", ["clr", "alr", "ilr"])
def test_logratio_raw_bridge_preserves_fixed_fraction_on_step_grid(
    representation: str,
) -> None:
    transformer = _logratio_transformer(representation)
    raw_site = _fraction_site(representation=representation)
    bridge, config, _raw_bounds = prepare_logratio_best_subset_config(
        OptimizeConfig(fixed_features={"alloy__fraction__Al": 0.2}),
        site_name="alloy",
        site_config=raw_site,
        transformer=transformer,
        model_feature_names=_logratio_layout(transformer),
        model_bounds=_logratio_bounds(transformer),
        dtype=torch.double,
        device=None,
    )

    callback = config.final_candidate_postprocess
    assert isinstance(callback, CompositionGridFinalPostprocess)
    al_index = bridge.fraction_indices[0]
    assert config.fixed_features is not None
    assert config.fixed_features[al_index] == pytest.approx(0.2)

    raw = torch.zeros((1, bridge.decision_dim), dtype=torch.double)
    raw[0, 0] = 900.0
    raw[0, list(bridge.fraction_indices)] = torch.tensor(
        [0.2, 0.47, 0.33, 0.0, 0.0],
        dtype=torch.double,
    )
    raw[0, -1] = 2.0
    projected = callback(raw)
    fractions = projected[0, list(bridge.fraction_indices)]

    assert fractions[0].item() == pytest.approx(0.2, abs=1e-10)
    assert fractions.sum().item() == pytest.approx(1.0, abs=1e-10)
    assert torch.isfinite(bridge.decision_to_model(projected)).all()


def _variable_transformer(representation: str = "ilr") -> CompositionTransformer:
    transformer = CompositionTransformer(
        elements=["Al", "Ti", "V", "Nb"],
        representation=representation,
        reference_element="Nb" if representation == "alr" else None,
        pseudocount=1e-8,
        prefix="alloy",
    )
    transformer.fit(["Al4Ti3V2Nb", "Al3Ti2V3Nb2"])
    return transformer


def _variable_layout(transformer: CompositionTransformer) -> tuple[str, ...]:
    return (
        "temperature",
        *transformer.representation_feature_names_,
        "alloy__total",
        "pressure",
    )


def _variable_model_bounds(transformer: CompositionTransformer) -> torch.Tensor:
    width = len(transformer.representation_feature_names_)
    return torch.tensor(
        [
            [800.0, *([-8.0] * width), 40.0, 1.0],
            [1200.0, *([8.0] * width), 90.0, 5.0],
        ],
        dtype=torch.double,
    )


def _variable_site() -> dict[str, Any]:
    return {
        "column": "formula",
        "elements": ("Al", "Ti", "V", "Nb"),
        "representation": "ilr",
        "normalization": "atomic_fraction",
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
        "steps": {"Al": 5.0, "Ti": 5.0, "V": 5.0, "Nb": 5.0},
        "min_components": 3,
        "max_components": 3,
        "required_components": (),
        "forbidden_components": (),
        "support_selection": "best_subset",
        "best_subset_strategy": "exact",
        "best_subset_max_combinations": 20,
    }


def test_variable_total_fixed_amount_is_preserved_by_step_grid() -> None:
    transformer = _variable_transformer()
    bridge, config, _bounds = prepare_variable_total_best_subset_config(
        OptimizeConfig(fixed_features={"alloy__amount__Al": 20.0}),
        site_name="alloy",
        site_config=_variable_site(),
        transformer=transformer,
        model_feature_names=_variable_layout(transformer),
        model_bounds=_variable_model_bounds(transformer),
        dtype=torch.double,
    )

    callback = config.final_candidate_postprocess
    assert isinstance(callback, CompositionVariableTotalGridFinalPostprocess)
    al_index = bridge.amount_indices[0]
    assert config.fixed_features is not None
    assert config.fixed_features[al_index] == pytest.approx(20.0)

    raw = torch.tensor(
        [[900.0, 20.0, 17.0, 13.0, 0.0, 2.0]],
        dtype=torch.double,
    )
    projected = callback(raw)
    amounts = projected[0, list(bridge.amount_indices)]

    assert amounts[0].item() == pytest.approx(20.0, abs=1e-10)
    assert 40.0 <= amounts.sum().item() <= 90.0
    assert int((amounts > 1e-10).sum().item()) == 3
    for value in amounts[amounts > 1e-10].tolist():
        assert value / 5.0 == pytest.approx(round(value / 5.0), abs=1e-8)


def test_variable_total_fixed_amount_off_step_grid_is_rejected() -> None:
    transformer = _variable_transformer()
    with pytest.raises(ValueError, match="not on the configured step grid"):
        prepare_variable_total_best_subset_config(
            OptimizeConfig(fixed_features={"alloy__amount__Al": 22.5}),
            site_name="alloy",
            site_config=_variable_site(),
            transformer=transformer,
            model_feature_names=_variable_layout(transformer),
            model_bounds=_variable_model_bounds(transformer),
            dtype=torch.double,
        )
