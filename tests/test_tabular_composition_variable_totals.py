import pandas as pd
import pytest

from bochan.tabular.optimizer_api import (
    TabularBayesianOptimizer as _CoreTabularBayesianOptimizer,
)
from bochan.tabular.variable_total_composition_optimizer import (
    TabularBayesianOptimizer,
)


def _single_site_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A_La": [30.0, 42.0],
            "A_Sr": [20.0, 18.0],
            "temperature": [900.0, 950.0],
            "property": [10.0, 12.0],
        }
    )


def test_variable_total_is_added_as_model_feature(monkeypatch) -> None:
    captured = {}

    def fake_fit(self, data=None, y=None, **kwargs):
        captured["data"] = data
        captured["kwargs"] = kwargs
        return self

    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "fit", fake_fit)
    bo = TabularBayesianOptimizer(
        input_cols=["A_La", "A_Sr", "temperature"],
        target_cols="property",
        bounds={"temperature": [850.0, 1000.0]},
        composition_sites={
            "A": {
                "element_columns": {"La": "A_La", "Sr": "A_Sr"},
                "representation": "ilr",
                "total_bounds": [30.0, 70.0],
            }
        },
    )

    assert bo.fit(_single_site_frame()) is bo
    assert captured["data"]["A__total"].tolist() == pytest.approx([50.0, 60.0])
    assert captured["kwargs"]["input_cols"] == [
        "A__ilr__1",
        "A__total",
        "temperature",
    ]
    assert captured["kwargs"]["bounds"]["A__total"] == [30.0, 70.0]


def test_variable_total_candidate_restores_requested_site_sum(monkeypatch) -> None:
    def fake_fit(self, data=None, y=None, **kwargs):
        return self

    def fake_candidate(self, *args, **kwargs):
        return (
            pd.DataFrame(
                {
                    "A__ilr__1": [0.0],
                    "A__total": [40.0],
                    "temperature": [925.0],
                }
            ),
            1.0,
        )

    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "fit", fake_fit)
    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "candidate", fake_candidate)
    bo = TabularBayesianOptimizer(
        input_cols=["A_La", "A_Sr", "temperature"],
        target_cols="property",
        composition_sites={
            "A": {
                "element_columns": {"La": "A_La", "Sr": "A_Sr"},
                "representation": "ilr",
                "total_bounds": [30.0, 70.0],
                "min_components": 2,
                "max_components": 2,
            }
        },
    )
    bo.fit(_single_site_frame())

    candidates, _ = bo.candidate()
    assert candidates.loc[0, ["A_La", "A_Sr"]].sum() == pytest.approx(40.0)
    assert candidates.loc[0, "A__total"] == pytest.approx(40.0)
    assert candidates.loc[0, "temperature"] == pytest.approx(925.0)


def test_two_variable_site_totals_can_be_coupled(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "A_La": [30.0, 35.0],
            "A_Sr": [20.0, 25.0],
            "B_Fe": [30.0, 25.0],
            "B_Co": [20.0, 15.0],
            "property": [10.0, 12.0],
        }
    )
    captured = {}

    def fake_fit(self, data=None, y=None, **kwargs):
        return self

    def fake_candidate(self, *args, **kwargs):
        captured["opt_config"] = kwargs["opt_config"]
        return (
            pd.DataFrame(
                {
                    "A__ilr__1": [0.0],
                    "A__total": [40.0],
                    "B__ilr__1": [0.0],
                    "B__total": [60.0],
                }
            ),
            1.0,
        )

    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "fit", fake_fit)
    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "candidate", fake_candidate)
    bo = TabularBayesianOptimizer(
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
    bo.fit(frame)

    candidates, _ = bo.candidate()
    assert captured["opt_config"]["constraints"] == [
        (["A__total", "B__total"], [1.0, 1.0], "=", 100.0)
    ]
    assert candidates.loc[0, ["A_La", "A_Sr"]].sum() == pytest.approx(40.0)
    assert candidates.loc[0, ["B_Fe", "B_Co"]].sum() == pytest.approx(60.0)
    assert candidates.loc[0, ["A__total", "B__total"]].sum() == pytest.approx(
        100.0
    )


def test_fixed_site_total_is_removed_from_coupled_constraint(monkeypatch) -> None:
    captured = {}

    def fake_fit(self, data=None, y=None, **kwargs):
        return self

    def fake_candidate(self, *args, **kwargs):
        captured["opt_config"] = kwargs["opt_config"]
        return (
            pd.DataFrame(
                {
                    "A__ilr__1": [0.0],
                    "A__total": [50.0],
                    "B__ilr__1": [0.0],
                }
            ),
            1.0,
        )

    frame = pd.DataFrame(
        {
            "A_La": [30.0, 35.0],
            "A_Sr": [20.0, 25.0],
            "B_Fe": [30.0, 25.0],
            "B_Co": [20.0, 25.0],
            "property": [10.0, 12.0],
        }
    )
    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "fit", fake_fit)
    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "candidate", fake_candidate)
    bo = TabularBayesianOptimizer(
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
    bo.fit(frame)

    bo.candidate()
    assert captured["opt_config"]["constraints"] == [
        (["A__total"], [1.0], "=", 50.0)
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


def test_public_tabular_optimizer_exposes_variable_total_support() -> None:
    from bochan.tabular import TabularBayesianOptimizer as PublicOptimizer

    assert PublicOptimizer is TabularBayesianOptimizer
