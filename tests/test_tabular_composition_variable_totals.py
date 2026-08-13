from __future__ import annotations

import pandas as pd
import pytest

from bochan.tabular import TabularBayesianOptimizer


def _single_site_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A_La": [30.0, 42.0],
            "A_Sr": [20.0, 18.0],
            "temperature": [900.0, 950.0],
            "property": [10.0, 12.0],
        }
    )


def _single_site_optimizer(**site_overrides) -> TabularBayesianOptimizer:
    site = {
        "element_columns": {"La": "A_La", "Sr": "A_Sr"},
        "representation": "ilr",
        "total_bounds": [30.0, 70.0],
        "min_components": 2,
        "max_components": 2,
    }
    site.update(site_overrides)
    return TabularBayesianOptimizer(
        input_cols=["A_La", "A_Sr", "temperature"],
        target_cols="property",
        bounds={"temperature": [850.0, 1000.0]},
        composition_sites={"A": site},
    )


def test_variable_total_is_added_as_model_feature() -> None:
    optimizer = _single_site_optimizer()
    transformed = optimizer.composition.prepare_frame(
        _single_site_frame(),
        fit_transformers=True,
    )

    assert transformed["A__total"].tolist() == pytest.approx([50.0, 60.0])
    assert optimizer.composition.replace_input_cols(
        ["A_La", "A_Sr", "temperature"]
    ) == ["A__ilr__1", "A__total", "temperature"]

    bounds = optimizer.composition.expanded_bounds(
        {"temperature": [850.0, 1000.0]},
        transformed,
    )
    assert bounds["A__total"] == [30.0, 70.0]


def test_variable_total_inverse_restores_requested_site_sum() -> None:
    optimizer = _single_site_optimizer()
    transformed = optimizer.composition.prepare_frame(
        _single_site_frame(),
        fit_transformers=True,
    )
    candidate = transformed.head(1).copy()
    candidate.loc[:, "A__ilr__1"] = 0.0
    candidate.loc[:, "A__total"] = 40.0
    candidate.loc[:, "temperature"] = 925.0

    restored = optimizer.inverse_compositions(candidate, repair=True)

    assert restored.loc[0, ["A_La", "A_Sr"]].sum() == pytest.approx(40.0)
    assert restored.loc[0, "A__total"] == pytest.approx(40.0)
    assert restored.loc[0, "temperature"] == pytest.approx(925.0)


def _two_site_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A_La": [30.0, 35.0],
            "A_Sr": [20.0, 25.0],
            "B_Fe": [30.0, 25.0],
            "B_Co": [20.0, 15.0],
            "property": [10.0, 12.0],
        }
    )


def test_two_variable_site_totals_can_be_coupled() -> None:
    optimizer = TabularBayesianOptimizer(
        input_cols=["A_La", "A_Sr", "B_Fe", "B_Co"],
        target_cols="property",
        composition_sites={
            "A": {
                "element_columns": {"La": "A_La", "Sr": "A_Sr"},
                "total_bounds": [30.0, 70.0],
            },
            "B": {
                "element_columns": {"Fe": "B_Fe", "Co": "B_Co"},
                "total_bounds": [30.0, 70.0],
            },
        },
        composition_total_constraints=[
            {"sites": ["A", "B"], "operator": "=", "total": 100.0}
        ],
    )
    transformed = optimizer.composition.prepare_frame(
        _two_site_frame(),
        fit_transformers=True,
    )

    _, opt_config = optimizer.candidates._prepare_configs(
        optimizer,
        {"name": "ei"},
        None,
        {},
    )
    assert opt_config.equality_constraints == [
        (["A__total", "B__total"], [1.0, 1.0], 100.0)
    ]

    candidate = transformed.head(1).copy()
    candidate.loc[:, "A__ilr__1"] = 0.0
    candidate.loc[:, "B__ilr__1"] = 0.0
    candidate.loc[:, "A__total"] = 40.0
    candidate.loc[:, "B__total"] = 60.0
    restored = optimizer.inverse_compositions(candidate, repair=True)
    assert restored.loc[0, ["A_La", "A_Sr"]].sum() == pytest.approx(40.0)
    assert restored.loc[0, ["B_Fe", "B_Co"]].sum() == pytest.approx(60.0)
    assert restored.loc[0, ["A__total", "B__total"]].sum() == pytest.approx(
        100.0
    )


def test_fixed_site_total_is_removed_from_coupled_constraint() -> None:
    optimizer = TabularBayesianOptimizer(
        input_cols=["A_La", "A_Sr", "B_Fe", "B_Co"],
        target_cols="property",
        composition_sites={
            "A": {
                "element_columns": {"La": "A_La", "Sr": "A_Sr"},
                "total_bounds": [30.0, 70.0],
            },
            "B": {
                "element_columns": {"Fe": "B_Fe", "Co": "B_Co"},
                "total": 50.0,
            },
        },
        composition_total_constraints=[
            {"sites": ["A", "B"], "operator": "=", "total": 100.0}
        ],
    )

    _, opt_config = optimizer.candidates._prepare_configs(
        optimizer,
        {"name": "ei"},
        None,
        {},
    )
    assert opt_config.equality_constraints == [
        (["A__total"], [1.0], 50.0)
    ]


def test_total_and_total_bounds_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="either 'total' or 'total_bounds'"):
        TabularBayesianOptimizer(
            composition_sites={
                "A": {
                    "element_columns": {"La": "A_La", "Sr": "A_Sr"},
                    "total": 50.0,
                    "total_bounds": [30.0, 70.0],
                }
            }
        )


def test_infeasible_coupled_total_constraint_is_rejected() -> None:
    with pytest.raises(ValueError, match="infeasible"):
        TabularBayesianOptimizer(
            composition_sites={
                "A": {
                    "element_columns": {"La": "A_La", "Sr": "A_Sr"},
                    "total_bounds": [30.0, 40.0],
                },
                "B": {
                    "element_columns": {"Fe": "B_Fe", "Co": "B_Co"},
                    "total_bounds": [30.0, 40.0],
                },
            },
            composition_total_constraints=[
                {"sites": ["A", "B"], "operator": "=", "total": 100.0}
            ],
        )


def test_variable_total_support_uses_canonical_public_optimizer() -> None:
    assert TabularBayesianOptimizer.__module__ == "bochan.tabular.optimizer.core"
