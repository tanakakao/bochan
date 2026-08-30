from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from bochan.api import OptimizeConfig
from bochan.api.support.best_subset import enumerate_best_subset_supports
from bochan.composition import CompositionSearchSpace
from bochan.tabular.composition.config import normalize_composition_sites
from bochan.tabular.composition.support import resolve_composition_best_subset
from bochan.tabular.data import resolve_optimize_config_columns


def _site_config(**overrides):
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


def _normalized_site(**overrides):
    site = normalize_composition_sites({"alloy": _site_config(**overrides)})["alloy"]
    site.setdefault("variable_total", False)
    return site


def _transformer(elements=("Al", "Ti", "V", "Cr", "Nb")):
    return SimpleNamespace(fitted_elements=tuple(elements), prefix="alloy")


def _feature_names():
    return [
        "alloy__fraction__Al",
        "alloy__fraction__Ti",
        "alloy__fraction__V",
        "alloy__fraction__Cr",
        "alloy__fraction__Nb",
        "temperature",
    ]


def test_composition_site_normalizes_best_subset_and_forbidden_bounds() -> None:
    site = _normalized_site()

    assert site["support_selection"] == "best_subset"
    assert site["required_components"] == ("Al",)
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


def test_composition_site_rejects_required_forbidden_overlap() -> None:
    with pytest.raises(ValueError, match="require and forbid"):
        normalize_composition_sites(
            {
                "alloy": _site_config(
                    required_components=["Al", "Cr"],
                    forbidden_components=["Cr"],
                )
            }
        )


def test_composition_best_subset_resolves_optional_element_group_and_core_supports() -> None:
    site = _normalized_site()
    config = resolve_composition_best_subset(
        OptimizeConfig(optimizer_kwargs={"best_subset_strategy": "exact"}),
        composition_sites={"alloy": site},
        composition_transformers={"alloy": _transformer()},
        feature_names=_feature_names(),
    )

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
    assert repair.final_sum_constraint == (
        (
            "alloy__fraction__Al",
            "alloy__fraction__Ti",
            "alloy__fraction__V",
            "alloy__fraction__Cr",
            "alloy__fraction__Nb",
        ),
        1.0,
    )

    resolved = resolve_optimize_config_columns(
        config,
        _feature_names(),
        dtype=torch.double,
        device=None,
    )
    resolved_repair = resolved.repair_config
    assert resolved_repair is not None
    assert resolved_repair.comp_idx == [1, 2, 4]
    assert resolved.fixed_features == {3: 0.0}
    assert enumerate_best_subset_supports(resolved) == [(1, 2), (1, 4), (2, 4)]


def test_positive_lower_bound_becomes_required_without_fixing_its_value() -> None:
    site = _normalized_site(
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
    config = resolve_composition_best_subset(
        OptimizeConfig(),
        composition_sites={"alloy": site},
        composition_transformers={"alloy": _transformer()},
        feature_names=_feature_names(),
    )

    repair = config.repair_config
    assert repair is not None
    assert "alloy__fraction__Al" not in repair.comp_idx
    assert repair.k == 2
    assert not config.fixed_features or "alloy__fraction__Al" not in config.fixed_features
    assert any(
        constraint[0] == ["alloy__fraction__Al"] and constraint[2] == pytest.approx(0.1)
        for constraint in config.inequality_constraints or ()
    )


def test_upper_zero_and_explicit_forbidden_are_fixed_to_zero() -> None:
    site = _normalized_site(
        forbidden_components=["Cr"],
        bounds={
            "Al": [0.05, 0.8],
            "Ti": [0.0, 0.8],
            "V": [0.0, 0.0],
            "Cr": [0.0, 0.8],
            "Nb": [0.0, 0.8],
        },
    )
    config = resolve_composition_best_subset(
        OptimizeConfig(),
        composition_sites={"alloy": site},
        composition_transformers={"alloy": _transformer()},
        feature_names=_feature_names(),
    )

    assert config.fixed_features == {
        "alloy__fraction__V": 0.0,
        "alloy__fraction__Cr": 0.0,
    }
    repair = config.repair_config
    assert repair is not None
    assert repair.comp_idx == ["alloy__fraction__Ti", "alloy__fraction__Nb"]
    assert repair.k == 2


@pytest.mark.parametrize("representation", ["ilr", "clr", "alr"])
def test_composition_best_subset_rejects_log_ratio_coordinate_support(representation: str) -> None:
    site = _normalized_site(representation=representation)
    with pytest.raises(ValueError, match="do not correspond one-to-one to element presence"):
        resolve_composition_best_subset(
            OptimizeConfig(),
            composition_sites={"alloy": site},
            composition_transformers={"alloy": _transformer()},
            feature_names=_feature_names(),
        )


def test_composition_best_subset_requires_exact_component_count() -> None:
    site = _normalized_site(min_components=2, max_components=3)
    with pytest.raises(ValueError, match="min_components == max_components"):
        resolve_composition_best_subset(
            OptimizeConfig(),
            composition_sites={"alloy": site},
            composition_transformers={"alloy": _transformer()},
            feature_names=_feature_names(),
        )


def test_composition_best_subset_rejects_variable_total_and_component_steps() -> None:
    variable = _normalized_site()
    variable["variable_total"] = True
    with pytest.raises(ValueError, match="fixed-total sites only"):
        resolve_composition_best_subset(
            OptimizeConfig(),
            composition_sites={"alloy": variable},
            composition_transformers={"alloy": _transformer()},
            feature_names=_feature_names(),
        )

    stepped = _normalized_site(steps={"Ti": 0.05})
    with pytest.raises(ValueError, match="continuous fractions"):
        resolve_composition_best_subset(
            OptimizeConfig(),
            composition_sites={"alloy": stepped},
            composition_transformers={"alloy": _transformer()},
            feature_names=_feature_names(),
        )


def test_composition_best_subset_rejects_multiple_sites_and_generic_sparse_group() -> None:
    site = _normalized_site()
    with pytest.raises(ValueError, match="one composition site"):
        resolve_composition_best_subset(
            OptimizeConfig(),
            composition_sites={"a": site, "b": dict(site)},
            composition_transformers={"a": _transformer(), "b": _transformer()},
            feature_names=_feature_names(),
        )

    from bochan.api import CandidateRepairConfig

    with pytest.raises(ValueError, match="owns CandidateRepairConfig.comp_idx"):
        resolve_composition_best_subset(
            OptimizeConfig(repair_config=CandidateRepairConfig(comp_idx=["temperature"], k=1)),
            composition_sites={"alloy": site},
            composition_transformers={"alloy": _transformer()},
            feature_names=_feature_names(),
        )


def test_composition_best_subset_rejects_supports_that_cannot_reach_unit_sum() -> None:
    site = _normalized_site(
        bounds={
            "Al": [0.05, 0.2],
            "Ti": [0.0, 0.2],
            "V": [0.0, 0.8],
            "Cr": [0.0, 0.8],
            "Nb": [0.0, 0.8],
        }
    )
    with pytest.raises(ValueError, match="active upper bounds"):
        resolve_composition_best_subset(
            OptimizeConfig(),
            composition_sites={"alloy": site},
            composition_transformers={"alloy": _transformer()},
            feature_names=_feature_names(),
        )
