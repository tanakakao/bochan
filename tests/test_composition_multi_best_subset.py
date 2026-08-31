from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from bochan.api import OptimizeConfig
from bochan.api.support.multi_group_best_subset import (
    BEST_SUBSET_GROUPS_KWARG,
    enumerate_grouped_best_subset_supports,
)
from bochan.tabular.composition.multi_support import (
    resolve_multiple_composition_best_subset,
)
from bochan.tabular.data import resolve_optimize_config_columns


def _site(
    prefix: str,
    elements: tuple[str, ...],
    *,
    min_components: int = 2,
    max_components: int = 2,
    required: tuple[str, ...] | None = None,
    forbidden: tuple[str, ...] = (),
    steps=None,
    strategy: str = "exact",
):
    required = (elements[0],) if required is None else required
    bounds = {element: (0.0, 1.0) for element in elements}
    for element in forbidden:
        bounds[element] = (0.0, 0.0)
    return {
        "column": f"{prefix}_formula",
        "elements": elements,
        "representation": "fractions",
        "normalization": "atomic_fraction",
        "prefix": prefix,
        "total": 1.0,
        "variable_total": False,
        "bounds": bounds,
        "steps": steps,
        "min_components": min_components,
        "max_components": max_components,
        "required_components": required,
        "forbidden_components": forbidden,
        "support_selection": "best_subset",
        "best_subset_strategy": strategy,
        "best_subset_max_combinations": 100,
        "best_subset_beam_width": 4,
        "best_subset_beam_steps": 3,
        "best_subset_max_evaluations": 50,
    }


def _transformer(prefix: str, elements: tuple[str, ...]):
    return SimpleNamespace(fitted_elements=elements, prefix=prefix)


def _fixture(*, steps: bool = False):
    a_elements = ("Al", "Ti", "V")
    b_elements = ("Fe", "Ni", "Co")
    step_values_a = {element: 0.5 for element in a_elements} if steps else None
    step_values_b = {element: 0.25 for element in b_elements} if steps else None
    sites = {
        "alloy_a": _site(
            "alloy_a",
            a_elements,
            steps=step_values_a,
        ),
        "alloy_b": _site(
            "alloy_b",
            b_elements,
            min_components=2,
            max_components=3,
            steps=step_values_b,
        ),
    }
    transformers = {
        "alloy_a": _transformer("alloy_a", a_elements),
        "alloy_b": _transformer("alloy_b", b_elements),
    }
    feature_names = [
        "alloy_a__fraction__Al",
        "alloy_a__fraction__Ti",
        "alloy_a__fraction__V",
        "alloy_b__fraction__Fe",
        "alloy_b__fraction__Ni",
        "alloy_b__fraction__Co",
        "temperature",
    ]
    return sites, transformers, feature_names


def _resolve(opt_config: OptimizeConfig | None = None, *, steps: bool = False):
    sites, transformers, feature_names = _fixture(steps=steps)
    config = resolve_multiple_composition_best_subset(
        opt_config or OptimizeConfig(),
        selected_sites=("alloy_a", "alloy_b"),
        composition_sites=sites,
        composition_transformers=transformers,
        feature_names=feature_names,
    )
    return config, feature_names


def test_multiple_fraction_sites_create_independent_sparse_groups() -> None:
    config, feature_names = _resolve()

    groups = config.optimizer_kwargs[BEST_SUBSET_GROUPS_KWARG]
    assert groups == (
        {"name": "alloy_a", "comp_idx": [1, 2], "min_k": 1, "max_k": 1},
        {"name": "alloy_b", "comp_idx": [4, 5], "min_k": 1, "max_k": 2},
    )
    assert config.repair_config is not None
    assert config.repair_config.comp_idx == [
        "alloy_a__fraction__Ti",
        "alloy_a__fraction__V",
        "alloy_b__fraction__Ni",
        "alloy_b__fraction__Co",
    ]
    assert config.repair_config.k == 3
    assert config.repair_config.final_sum_constraint is None

    resolved = resolve_optimize_config_columns(
        config,
        feature_names,
        dtype=torch.double,
        device=None,
    )
    assert resolved.repair_config is not None
    assert resolved.repair_config.comp_idx == [1, 2, 4, 5]
    supports = enumerate_grouped_best_subset_supports(resolved)
    assert len(supports) == 6
    assert all(sum(index in {1, 2} for index in support) == 1 for support in supports)
    assert {sum(index in {4, 5} for index in support) for support in supports} == {1, 2}


def test_multiple_fraction_sites_keep_one_sum_constraint_per_site() -> None:
    config, _feature_names = _resolve()

    equalities = list(config.equality_constraints or ())
    assert any(
        tuple(indices)
        == (
            "alloy_a__fraction__Al",
            "alloy_a__fraction__Ti",
            "alloy_a__fraction__V",
        )
        and float(rhs) == pytest.approx(1.0)
        for indices, _coefficients, rhs in equalities
    )
    assert any(
        tuple(indices)
        == (
            "alloy_b__fraction__Fe",
            "alloy_b__fraction__Ni",
            "alloy_b__fraction__Co",
        )
        and float(rhs) == pytest.approx(1.0)
        for indices, _coefficients, rhs in equalities
    )


def test_multiple_fraction_sites_merge_forbidden_features() -> None:
    sites, transformers, feature_names = _fixture()
    sites["alloy_a"] = _site(
        "alloy_a",
        ("Al", "Ti", "V"),
        forbidden=("V",),
    )
    sites["alloy_b"] = _site(
        "alloy_b",
        ("Fe", "Ni", "Co"),
        forbidden=("Co",),
    )

    config = resolve_multiple_composition_best_subset(
        OptimizeConfig(),
        selected_sites=("alloy_a", "alloy_b"),
        composition_sites=sites,
        composition_transformers=transformers,
        feature_names=feature_names,
    )

    assert config.fixed_features == {
        "alloy_a__fraction__V": 0.0,
        "alloy_b__fraction__Co": 0.0,
    }


def test_multiple_sites_reject_conflicting_site_search_policies_without_override() -> None:
    sites, transformers, feature_names = _fixture()
    sites["alloy_b"] = dict(sites["alloy_b"], best_subset_strategy="beam")

    with pytest.raises(ValueError, match="conflicting best_subset_strategy"):
        resolve_multiple_composition_best_subset(
            OptimizeConfig(),
            selected_sites=("alloy_a", "alloy_b"),
            composition_sites=sites,
            composition_transformers=transformers,
            feature_names=feature_names,
        )

    config = resolve_multiple_composition_best_subset(
        OptimizeConfig(optimizer_kwargs={"best_subset_strategy": "auto"}),
        selected_sites=("alloy_a", "alloy_b"),
        composition_sites=sites,
        composition_transformers=transformers,
        feature_names=feature_names,
    )
    assert config.optimizer_kwargs["best_subset_strategy"] == "auto"


def test_two_independent_step_grids_are_chained_without_moving_process_values() -> None:
    config, _feature_names = _resolve(steps=True)
    callback = config.final_candidate_postprocess
    assert callback is not None

    candidates = torch.tensor(
        [
            [0.55, 0.45, 0.0, 0.55, 0.0, 0.45, 913.0],
            [0.55, 0.0, 0.45, 0.55, 0.45, 0.0, 987.0],
        ],
        dtype=torch.double,
    )
    projected = callback(candidates)

    assert projected[:, 6].tolist() == pytest.approx([913.0, 987.0], abs=1e-10)
    assert projected[:, :3].sum(dim=-1).tolist() == pytest.approx([1.0, 1.0], abs=1e-10)
    assert projected[:, 3:6].sum(dim=-1).tolist() == pytest.approx([1.0, 1.0], abs=1e-10)
    assert projected[0, :3].tolist() == pytest.approx([0.5, 0.5, 0.0], abs=1e-10)
    assert projected[1, :3].tolist() == pytest.approx([0.5, 0.0, 0.5], abs=1e-10)
    assert projected[0, 3:6].tolist() == pytest.approx([0.5, 0.0, 0.5], abs=1e-10)
    assert projected[1, 3:6].tolist() == pytest.approx([0.5, 0.5, 0.0], abs=1e-10)


def test_cross_site_linear_constraint_is_rejected_when_both_sites_are_stepped() -> None:
    sites, transformers, feature_names = _fixture(steps=True)
    config = OptimizeConfig(
        equality_constraints=[
            (
                ["alloy_a__fraction__Al", "alloy_b__fraction__Fe"],
                [1.0, -1.0],
                0.0,
            )
        ]
    )

    with pytest.raises(ValueError, match="cannot couple two composition Best Subset sites"):
        resolve_multiple_composition_best_subset(
            config,
            selected_sites=("alloy_a", "alloy_b"),
            composition_sites=sites,
            composition_transformers=transformers,
            feature_names=feature_names,
        )
