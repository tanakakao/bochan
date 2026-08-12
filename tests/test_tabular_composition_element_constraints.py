import pandas as pd
import pytest

from bochan.tabular.composition import ATOMIC_WEIGHTS
from bochan.tabular.element_constraint_composition_optimizer import (
    TabularBayesianOptimizer,
)
from bochan.tabular.optimizer_api import (
    TabularBayesianOptimizer as _CoreTabularBayesianOptimizer,
)


def _fake_fit(self, data=None, y=None, **kwargs):
    return self


def test_same_site_atomic_ratio_is_repaired_after_ilr(monkeypatch) -> None:
    def fake_candidate(self, *args, **kwargs):
        return pd.DataFrame({"A__ilr__1": [0.0]}), 1.0

    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "fit", _fake_fit)
    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "candidate", fake_candidate)
    frame = pd.DataFrame(
        {
            "A_La": [0.6, 0.5],
            "A_Sr": [0.4, 0.5],
            "property": [1.0, 2.0],
        }
    )
    bo = TabularBayesianOptimizer(
        input_cols=["A_La", "A_Sr"],
        target_cols="property",
        composition_sites={
            "A": {
                "element_columns": {"La": "A_La", "Sr": "A_Sr"},
                "representation": "ilr",
                "total": 1.0,
                "min_components": 2,
                "max_components": 2,
            }
        },
        composition_element_constraints=[
            {
                "terms": [
                    {"site": "A", "element": "Sr", "coefficient": 1.0},
                    {"site": "A", "element": "La", "coefficient": -0.5},
                ],
                "operator": "=",
                "rhs": 0.0,
                "basis": "atomic_amount",
            }
        ],
        composition_constraint_rerank=False,
    )
    bo.fit(frame)

    candidates, _ = bo.candidate()
    assert candidates.loc[0, "A_Sr"] == pytest.approx(
        0.5 * candidates.loc[0, "A_La"], abs=1e-7
    )
    assert candidates.loc[0, ["A_La", "A_Sr"]].sum() == pytest.approx(1.0)


def test_same_site_atomic_ratio_range_is_repaired(monkeypatch) -> None:
    def fake_candidate(self, *args, **kwargs):
        return pd.DataFrame(
            {"A__ilr__1": [2.0], "A__ilr__2": [-2.0]}
        ), 1.0

    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "fit", _fake_fit)
    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "candidate", fake_candidate)
    frame = pd.DataFrame(
        {
            "A_La": [0.6, 0.5],
            "A_Sr": [0.3, 0.4],
            "A_Ba": [0.1, 0.1],
            "property": [1.0, 2.0],
        }
    )
    constraints = [
        {
            "terms": [
                {"site": "A", "element": "Sr", "coefficient": 1.0},
                {"site": "A", "element": "La", "coefficient": -0.4},
            ],
            "operator": ">=",
        },
        {
            "terms": [
                {"site": "A", "element": "Sr", "coefficient": 1.0},
                {"site": "A", "element": "La", "coefficient": -0.6},
            ],
            "operator": "<=",
        },
    ]
    bo = TabularBayesianOptimizer(
        input_cols=["A_La", "A_Sr", "A_Ba"],
        target_cols="property",
        composition_sites={
            "A": {
                "element_columns": {
                    "La": "A_La",
                    "Sr": "A_Sr",
                    "Ba": "A_Ba",
                },
                "representation": "ilr",
                "total": 1.0,
                "min_components": 2,
                "max_components": 3,
            }
        },
        composition_element_constraints=constraints,
        composition_constraint_rerank=False,
    )
    bo.fit(frame)

    candidates, _ = bo.candidate()
    ratio = candidates.loc[0, "A_Sr"] / candidates.loc[0, "A_La"]
    assert 0.4 - 1e-7 <= ratio <= 0.6 + 1e-7


def test_cross_site_atomic_constraint_converts_weight_amounts(monkeypatch) -> None:
    def fake_candidate(self, *args, **kwargs):
        return (
            pd.DataFrame(
                {
                    "A__ilr__1": [0.0],
                    "B__ilr__1": [0.0],
                }
            ),
            1.0,
        )

    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "fit", _fake_fit)
    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "candidate", fake_candidate)
    frame = pd.DataFrame(
        {
            "A_La": [25.0, 30.0],
            "A_Sr": [25.0, 20.0],
            "B_Fe": [25.0, 20.0],
            "B_Co": [25.0, 30.0],
            "property": [1.0, 2.0],
        }
    )
    bo = TabularBayesianOptimizer(
        input_cols=["A_La", "A_Sr", "B_Fe", "B_Co"],
        target_cols="property",
        composition_sites={
            "A": {
                "element_columns": {"La": "A_La", "Sr": "A_Sr"},
                "input_basis": "weight_fraction",
                "representation": "ilr",
                "total": 50.0,
                "min_components": 2,
                "max_components": 2,
            },
            "B": {
                "element_columns": {"Fe": "B_Fe", "Co": "B_Co"},
                "input_basis": "atomic_fraction",
                "representation": "ilr",
                "total": 50.0,
                "min_components": 2,
                "max_components": 2,
            },
        },
        composition_element_constraints=[
            {
                "terms": [
                    {"site": "A", "element": "La", "coefficient": 1.0},
                    {"site": "B", "element": "Fe", "coefficient": -0.5},
                ],
                "operator": "=",
                "basis": "atomic_amount",
            }
        ],
        composition_constraint_rerank=False,
    )
    bo.fit(frame)

    candidates, _ = bo.candidate()
    a_la_atomic = candidates.loc[0, "A_La"] / ATOMIC_WEIGHTS["La"]
    assert a_la_atomic == pytest.approx(
        0.5 * candidates.loc[0, "B_Fe"], abs=1e-7
    )
    assert candidates.loc[0, ["A_La", "A_Sr"]].sum() == pytest.approx(50.0)
    assert candidates.loc[0, ["B_Fe", "B_Co"]].sum() == pytest.approx(50.0)


def test_compatible_steps_are_preserved_with_exact_ratio(monkeypatch) -> None:
    def fake_candidate(self, *args, **kwargs):
        return pd.DataFrame({"A__ilr__1": [0.0]}), 1.0

    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "fit", _fake_fit)
    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "candidate", fake_candidate)
    frame = pd.DataFrame(
        {
            "A_La": [66.0, 64.0],
            "A_Sr": [33.0, 35.0],
            "property": [1.0, 2.0],
        }
    )
    bo = TabularBayesianOptimizer(
        input_cols=["A_La", "A_Sr"],
        target_cols="property",
        composition_sites={
            "A": {
                "element_columns": {"La": "A_La", "Sr": "A_Sr"},
                "representation": "ilr",
                "total": 99.0,
                "steps": {"La": 2.0, "Sr": 1.0},
                "min_components": 2,
                "max_components": 2,
            }
        },
        composition_element_constraints=[
            {
                "terms": [
                    {"site": "A", "element": "Sr", "coefficient": 1.0},
                    {"site": "A", "element": "La", "coefficient": -0.5},
                ],
                "operator": "=",
            }
        ],
        composition_constraint_rerank=False,
    )
    bo.fit(frame)

    candidates, _ = bo.candidate()
    assert candidates.loc[0, "A_La"] == pytest.approx(66.0)
    assert candidates.loc[0, "A_Sr"] == pytest.approx(33.0)


def test_incompatible_steps_and_ratio_are_rejected() -> None:
    with pytest.raises(ValueError, match="jointly infeasible"):
        TabularBayesianOptimizer(
            composition_sites={
                "A": {
                    "element_columns": {"La": "A_La", "Sr": "A_Sr"},
                    "total": 100.0,
                    "steps": {"La": 2.0, "Sr": 1.0},
                    "min_components": 2,
                    "max_components": 2,
                }
            },
            composition_element_constraints=[
                {
                    "terms": [
                        {"site": "A", "element": "Sr", "coefficient": 1.0},
                        {"site": "A", "element": "La", "coefficient": -0.5},
                    ],
                    "operator": "=",
                }
            ],
        )


def test_fixed_fraction_constraint_is_forwarded_to_named_optimizer(monkeypatch) -> None:
    captured = {}

    def fake_candidate(self, *args, **kwargs):
        captured["opt_config"] = kwargs["opt_config"]
        return (
            pd.DataFrame(
                {
                    "A__fraction__La": [2.0 / 3.0],
                    "A__fraction__Sr": [1.0 / 3.0],
                }
            ),
            1.0,
        )

    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "fit", _fake_fit)
    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "candidate", fake_candidate)
    frame = pd.DataFrame(
        {
            "A_La": [0.6, 0.5],
            "A_Sr": [0.4, 0.5],
            "property": [1.0, 2.0],
        }
    )
    bo = TabularBayesianOptimizer(
        input_cols=["A_La", "A_Sr"],
        target_cols="property",
        composition_sites={
            "A": {
                "element_columns": {"La": "A_La", "Sr": "A_Sr"},
                "representation": "fractions",
                "total": 1.0,
                "min_components": 2,
                "max_components": 2,
            }
        },
        composition_element_constraints=[
            {
                "terms": [
                    {"site": "A", "element": "Sr", "coefficient": 1.0},
                    {"site": "A", "element": "La", "coefficient": -0.5},
                ],
                "operator": "=",
            }
        ],
        composition_constraint_rerank=False,
    )
    bo.fit(frame)

    bo.candidate()
    assert captured["opt_config"]["constraints"] == [
        (
            ["A__fraction__Sr", "A__fraction__La"],
            [1.0, -0.5],
            "=",
            0.0,
        )
    ]


def test_public_optimizer_exposes_element_constraint_support() -> None:
    from bochan.tabular import TabularBayesianOptimizer as PublicOptimizer

    assert issubclass(PublicOptimizer, TabularBayesianOptimizer)
