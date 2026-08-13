from __future__ import annotations

import pandas as pd
import pytest

from bochan.tabular import TabularBayesianOptimizer
from bochan.tabular.composition import ATOMIC_WEIGHTS


def _inverse(optimizer, frame, **coordinates):
    transformed = optimizer.composition.prepare_frame(frame, fit_transformers=True)
    candidate = transformed.head(1).copy()
    for name, value in coordinates.items():
        candidate.loc[:, name] = value
    return optimizer.inverse_compositions(candidate, repair=True)


def _ratio_constraint(lower: float, operator: str):
    return {
        "terms": [
            {"site": "A", "element": "Sr", "coefficient": 1.0},
            {"site": "A", "element": "La", "coefficient": -lower},
        ],
        "operator": operator,
    }


def test_same_site_atomic_ratio_is_repaired_after_ilr() -> None:
    frame = pd.DataFrame(
        {"A_La": [0.6, 0.5], "A_Sr": [0.4, 0.5], "property": [1.0, 2.0]}
    )
    optimizer = TabularBayesianOptimizer(
        input_cols=["A_La", "A_Sr"],
        target_cols="property",
        composition_sites={"A": {
            "element_columns": {"La": "A_La", "Sr": "A_Sr"},
            "representation": "ilr", "total": 1.0,
            "min_components": 2, "max_components": 2,
        }},
        composition_element_constraints=[_ratio_constraint(0.5, "=")],
        composition_constraint_rerank=False,
    )
    restored = _inverse(optimizer, frame, A__ilr__1=0.0)
    assert restored.loc[0, "A_Sr"] == pytest.approx(
        0.5 * restored.loc[0, "A_La"], abs=1e-7
    )
    assert restored.loc[0, ["A_La", "A_Sr"]].sum() == pytest.approx(1.0)


def test_same_site_atomic_ratio_range_is_repaired() -> None:
    frame = pd.DataFrame({
        "A_La": [0.6, 0.5], "A_Sr": [0.3, 0.4],
        "A_Ba": [0.1, 0.1], "property": [1.0, 2.0],
    })
    optimizer = TabularBayesianOptimizer(
        input_cols=["A_La", "A_Sr", "A_Ba"], target_cols="property",
        composition_sites={"A": {
            "element_columns": {"La": "A_La", "Sr": "A_Sr", "Ba": "A_Ba"},
            "representation": "ilr", "total": 1.0,
            "min_components": 2, "max_components": 3,
        }},
        composition_element_constraints=[
            _ratio_constraint(0.4, ">="), _ratio_constraint(0.6, "<=")
        ],
        composition_constraint_rerank=False,
    )
    restored = _inverse(optimizer, frame, A__ilr__1=2.0, A__ilr__2=-2.0)
    ratio = restored.loc[0, "A_Sr"] / restored.loc[0, "A_La"]
    assert 0.4 - 1e-7 <= ratio <= 0.6 + 1e-7


def test_cross_site_atomic_constraint_converts_weight_amounts() -> None:
    frame = pd.DataFrame({
        "A_La": [25.0, 30.0], "A_Sr": [25.0, 20.0],
        "B_Fe": [25.0, 20.0], "B_Co": [25.0, 30.0],
        "property": [1.0, 2.0],
    })
    optimizer = TabularBayesianOptimizer(
        input_cols=["A_La", "A_Sr", "B_Fe", "B_Co"], target_cols="property",
        composition_sites={
            "A": {"element_columns": {"La": "A_La", "Sr": "A_Sr"},
                  "input_basis": "weight_fraction", "representation": "ilr",
                  "total": 50.0, "min_components": 2, "max_components": 2},
            "B": {"element_columns": {"Fe": "B_Fe", "Co": "B_Co"},
                  "input_basis": "atomic_fraction", "representation": "ilr",
                  "total": 50.0, "min_components": 2, "max_components": 2},
        },
        composition_element_constraints=[{
            "terms": [
                {"site": "A", "element": "La", "coefficient": 1.0},
                {"site": "B", "element": "Fe", "coefficient": -0.5},
            ],
            "operator": "=", "basis": "atomic_amount",
        }],
        composition_constraint_rerank=False,
    )
    restored = _inverse(optimizer, frame, A__ilr__1=0.0, B__ilr__1=0.0)
    assert restored.loc[0, "A_La"] / ATOMIC_WEIGHTS["La"] == pytest.approx(
        0.5 * restored.loc[0, "B_Fe"], abs=1e-7
    )
    assert restored.loc[0, ["A_La", "A_Sr"]].sum() == pytest.approx(50.0)
    assert restored.loc[0, ["B_Fe", "B_Co"]].sum() == pytest.approx(50.0)


def test_compatible_steps_are_preserved_with_exact_ratio() -> None:
    frame = pd.DataFrame(
        {"A_La": [66.0, 64.0], "A_Sr": [33.0, 35.0], "property": [1.0, 2.0]}
    )
    optimizer = TabularBayesianOptimizer(
        input_cols=["A_La", "A_Sr"], target_cols="property",
        composition_sites={"A": {
            "element_columns": {"La": "A_La", "Sr": "A_Sr"},
            "representation": "ilr", "total": 99.0,
            "steps": {"La": 2.0, "Sr": 1.0},
            "min_components": 2, "max_components": 2,
        }},
        composition_element_constraints=[_ratio_constraint(0.5, "=")],
        composition_constraint_rerank=False,
    )
    restored = _inverse(optimizer, frame, A__ilr__1=0.0)
    assert restored.loc[0, "A_La"] == pytest.approx(66.0)
    assert restored.loc[0, "A_Sr"] == pytest.approx(33.0)


def test_incompatible_steps_and_ratio_are_rejected() -> None:
    with pytest.raises(ValueError, match="jointly infeasible"):
        TabularBayesianOptimizer(
            composition_sites={"A": {
                "element_columns": {"La": "A_La", "Sr": "A_Sr"},
                "total": 100.0, "steps": {"La": 2.0, "Sr": 1.0},
                "min_components": 2, "max_components": 2,
            }},
            composition_element_constraints=[_ratio_constraint(0.5, "=")],
        )


def test_fixed_fraction_constraint_is_forwarded_to_named_optimizer() -> None:
    frame = pd.DataFrame(
        {"A_La": [0.6, 0.5], "A_Sr": [0.4, 0.5], "property": [1.0, 2.0]}
    )
    optimizer = TabularBayesianOptimizer(
        input_cols=["A_La", "A_Sr"], target_cols="property",
        composition_sites={"A": {
            "element_columns": {"La": "A_La", "Sr": "A_Sr"},
            "representation": "fractions", "total": 1.0,
            "min_components": 2, "max_components": 2,
        }},
        composition_element_constraints=[_ratio_constraint(0.5, "=")],
        composition_constraint_rerank=False,
    )
    optimizer.composition.prepare_frame(frame, fit_transformers=True)
    _, opt_config = optimizer.candidates._prepare_configs(
        optimizer, {"name": "ei"}, None, {}
    )
    assert opt_config.equality_constraints == [
        (["A__fraction__Sr", "A__fraction__La"], [1.0, -0.5], 0.0)
    ]


def test_public_optimizer_exposes_element_constraint_support() -> None:
    assert TabularBayesianOptimizer.__module__ == "bochan.tabular.optimizer.core"
