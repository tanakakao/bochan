from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from bochan.api import CandidateRepairConfig, OptimizeConfig
from bochan.api.support.best_subset import enumerate_best_subset_supports
from bochan.composition import CompositionSearchSpace
from bochan.tabular.composition.config import normalize_composition_sites
from bochan.tabular.composition.support import resolve_composition_best_subset
from bochan.tabular.data import resolve_optimize_config_columns


def _raw_site(**overrides):
    config = {
        "column": "formula",
        "elements": ["Al", "Ti", "V", "Cr", "Nb"],
        "representation": "fractions",
        "min_components": 3,
        "max_components": 3,
        "required_components": ["Al"],
        "forbidden_components": ["Cr"],
        "support_selection": "best_subset",
        "bounds": {
            "Al": [0.05, 0.8],
            "Ti": [0.0, 0.8],
            "V": [0.0, 0.8],
            "Cr": [0.0, 0.8],
            "Nb": [0.0, 0.8],
        },
    }
    config.update(overrides)
    return config


def _site(**overrides):
    site = normalize_composition_sites({"alloy": _raw_site(**overrides)})["alloy"]
    site.setdefault("variable_total", False)
    return site


def _transformer():
    return SimpleNamespace(
        fitted_elements=("Al", "Ti", "V", "Cr", "Nb"),
        prefix="alloy",
    )


def _features():
    return [
        "alloy__fraction__Al",
        "alloy__fraction__Ti",
        "alloy__fraction__V",
        "alloy__fraction__Cr",
        "alloy__fraction__Nb",
        "temperature",
    ]


def _resolve(site, config=None):
    return resolve_composition_best_subset(
        config or OptimizeConfig(),
        composition_sites={"alloy": site},
        composition_transformers={"alloy": _transformer()},
        feature_names=_features(),
    )


def test_forbidden_components_are_normalized_into_zero_bounds_and_repair() -> None:
    site = _site()
    assert site["support_selection"] == "best_subset"
    assert site["forbidden_components"] == ("Cr",)
    assert site["bounds"]["Cr"] == (0.0, 0.0)

    space = CompositionSearchSpace(
        components=site["elements"],
        total=site["total"],
        bounds=site["bounds"],
        min_active_components=site["min_components"],
        max_active_components=site["max_components"],
        required_components=site["required_components"],
    )
    repaired = space.repair({"Al": 0.2, "Ti": 0.2, "V": 0.1, "Cr": 0.5, "Nb": 0.0})
    assert repaired["Cr"] == pytest.approx(0.0)
    assert repaired["Al"] > 0.0
    assert sum(value > 1e-8 for value in repaired.values()) == 3
    assert sum(repaired.values()) == pytest.approx(1.0)


def test_site_config_rejects_invalid_support_and_required_forbidden_overlap() -> None:
    with pytest.raises(ValueError, match="support_selection"):
        normalize_composition_sites({"alloy": _raw_site(support_selection="unknown")})
    with pytest.raises(ValueError, match="require and forbid"):
        normalize_composition_sites(
            {
                "alloy": _raw_site(
                    required_components=["Al", "Cr"],
                    forbidden_components=["Cr"],
                )
            }
        )


def test_fraction_best_subset_resolves_to_generic_core_support_search() -> None:
    config = _resolve(_site(), OptimizeConfig(optimizer_kwargs={"best_subset_strategy": "exact"}))
    repair = config.repair_config
    assert repair is not None
    assert repair.support_selection == "best_subset"
    assert repair.k == 2
    assert repair.comp_idx == [
        "alloy__fraction__Ti",
        "alloy__fraction__V",
        "alloy__fraction__Nb",
    ]
    assert config.fixed_features == {"alloy__fraction__Cr": 0.0}
    assert repair.final_sum_constraint is not None

    resolved = resolve_optimize_config_columns(
        config,
        _features(),
        dtype=torch.double,
        device=None,
    )
    assert resolved.fixed_features == {3: 0.0}
    assert resolved.repair_config is not None
    assert resolved.repair_config.comp_idx == [1, 2, 4]
    assert enumerate_best_subset_supports(resolved) == [(1, 2), (1, 4), (2, 4)]


def test_required_components_are_free_valued_not_fixed() -> None:
    site = _site(
        required_components=[],
        forbidden_components=[],
        bounds={
            "Al": [0.1, 0.8],
            "Ti": [0.0, 0.8],
            "V": [0.0, 0.8],
            "Cr": [0.0, 0.8],
            "Nb": [0.0, 0.8],
        },
    )
    config = _resolve(site)
    assert config.repair_config is not None
    assert "alloy__fraction__Al" not in config.repair_config.comp_idx
    assert not config.fixed_features or "alloy__fraction__Al" not in config.fixed_features
    assert any(
        indices == ["alloy__fraction__Al"] and float(rhs) == pytest.approx(0.1)
        for indices, _, rhs in config.inequality_constraints or ()
    )


def test_explicit_forbidden_and_upper_zero_are_removed_from_support() -> None:
    site = _site(
        forbidden_components=["Cr"],
        bounds={
            "Al": [0.05, 0.8],
            "Ti": [0.0, 0.8],
            "V": [0.0, 0.0],
            "Cr": [0.0, 0.8],
            "Nb": [0.0, 0.8],
        },
    )
    config = _resolve(site)
    assert config.fixed_features == {
        "alloy__fraction__V": 0.0,
        "alloy__fraction__Cr": 0.0,
    }
    assert config.repair_config is not None
    assert config.repair_config.comp_idx == ["alloy__fraction__Ti", "alloy__fraction__Nb"]
    assert config.repair_config.k == 2


@pytest.mark.parametrize("representation", ["ilr", "clr", "alr"])
def test_log_ratio_coordinates_are_not_mistaken_for_element_support(representation: str) -> None:
    with pytest.raises(ValueError, match="do not correspond one-to-one to element presence"):
        _resolve(_site(representation=representation))


def test_scope_guards_are_explicit() -> None:
    with pytest.raises(ValueError, match="min_components == max_components"):
        _resolve(_site(min_components=2, max_components=3))

    variable = _site()
    variable["variable_total"] = True
    with pytest.raises(ValueError, match="fixed-total sites only"):
        _resolve(variable)

    with pytest.raises(ValueError, match="continuous fractions"):
        _resolve(_site(steps={"Ti": 0.05}))

    site = _site()
    with pytest.raises(ValueError, match="one composition site"):
        resolve_composition_best_subset(
            OptimizeConfig(),
            composition_sites={"a": site, "b": dict(site)},
            composition_transformers={"a": _transformer(), "b": _transformer()},
            feature_names=_features(),
        )

    with pytest.raises(ValueError, match="owns CandidateRepairConfig.comp_idx"):
        _resolve(
            site,
            OptimizeConfig(
                repair_config=CandidateRepairConfig(comp_idx=["temperature"], k=1)
            ),
        )


def test_infeasible_exact_support_bounds_are_rejected_before_optimizer_calls() -> None:
    site = _site(
        bounds={
            "Al": [0.05, 0.2],
            "Ti": [0.0, 0.1],
            "V": [0.0, 0.2],
            "Cr": [0.0, 0.8],
            "Nb": [0.0, 0.8],
        }
    )
    with pytest.raises(ValueError, match="active upper bounds"):
        _resolve(site)
