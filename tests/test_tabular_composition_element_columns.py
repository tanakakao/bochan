import pandas as pd
import pytest

from bochan.tabular.element_column_composition_optimizer import (
    TabularBayesianOptimizer,
)
from bochan.tabular.optimizer_api import (
    TabularBayesianOptimizer as _CoreTabularBayesianOptimizer,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Fe": [50.0, 0.5],
            "Ti": [30.0, 0.3],
            "Al": [20.0, 0.2],
            "temperature": [900.0, 950.0],
            "property": [10.0, 12.0],
        }
    )


def test_element_columns_are_replaced_with_ilr_features(monkeypatch) -> None:
    captured = {}

    def fake_fit(self, data=None, y=None, **kwargs):
        captured["data"] = data
        captured["kwargs"] = kwargs
        return self

    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "fit", fake_fit)
    bo = TabularBayesianOptimizer(
        input_cols=["Fe", "Ti", "Al", "temperature"],
        target_cols="property",
        bounds={"temperature": [850.0, 1000.0]},
        composition_sites={
            "alloy": {
                "element_columns": {"Fe": "Fe", "Ti": "Ti", "Al": "Al"},
                "representation": "ilr",
            }
        },
    )

    assert bo.fit(_frame()) is bo
    assert all(column not in captured["data"] for column in ["Fe", "Ti", "Al"])
    assert captured["kwargs"]["input_cols"] == [
        "alloy__ilr__1",
        "alloy__ilr__2",
        "temperature",
    ]
    assert captured["kwargs"]["bounds"]["alloy__ilr__1"] == [-8.0, 8.0]
    assert captured["data"].loc[0, "alloy__ilr__1"] == pytest.approx(
        captured["data"].loc[1, "alloy__ilr__1"]
    )
    assert captured["data"].loc[0, "alloy__ilr__2"] == pytest.approx(
        captured["data"].loc[1, "alloy__ilr__2"]
    )


def test_element_column_candidates_restore_original_columns(monkeypatch) -> None:
    def fake_fit(self, data=None, y=None, **kwargs):
        return self

    def fake_candidate(self, *args, **kwargs):
        return (
            pd.DataFrame(
                {
                    "alloy__ilr__1": [0.0],
                    "alloy__ilr__2": [0.0],
                    "temperature": [925.0],
                }
            ),
            1.0,
        )

    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "fit", fake_fit)
    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "candidate", fake_candidate)
    bo = TabularBayesianOptimizer(
        input_cols=["Fe", "Ti", "Al", "temperature"],
        target_cols="property",
        bounds={"temperature": [850.0, 1000.0]},
        composition_sites={
            "alloy": {
                "element_columns": {"Fe": "Fe", "Ti": "Ti", "Al": "Al"},
                "representation": "ilr",
                "min_components": 2,
                "max_components": 2,
                "required_components": ["Fe"],
                "steps": {"Fe": 0.01, "Ti": 0.01, "Al": 0.01},
            }
        },
    )
    bo.fit(_frame())

    candidates, _ = bo.candidate()
    assert candidates.loc[0, ["Fe", "Ti", "Al"]].sum() == pytest.approx(1.0)
    assert (candidates.loc[0, ["Fe", "Ti", "Al"]] > 0).sum() == 2
    assert candidates.loc[0, "Fe"] > 0
    assert candidates.loc[0, "temperature"] == 925.0


def test_two_element_column_sites_are_repaired_independently(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "A_La": [0.6, 0.7],
            "A_Sr": [0.4, 0.3],
            "A_Ba": [0.0, 0.0],
            "B_Fe": [0.8, 0.6],
            "B_Co": [0.2, 0.0],
            "B_Mn": [0.0, 0.4],
            "temperature": [900.0, 950.0],
            "property": [10.0, 12.0],
        }
    )

    def fake_fit(self, data=None, y=None, **kwargs):
        return self

    def fake_candidate(self, *args, **kwargs):
        return (
            pd.DataFrame(
                {
                    "A__ilr__1": [0.0],
                    "A__ilr__2": [0.0],
                    "B__ilr__1": [0.0],
                    "B__ilr__2": [0.0],
                    "temperature": [925.0],
                }
            ),
            1.0,
        )

    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "fit", fake_fit)
    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "candidate", fake_candidate)
    bo = TabularBayesianOptimizer(
        input_cols=[
            "A_La",
            "A_Sr",
            "A_Ba",
            "B_Fe",
            "B_Co",
            "B_Mn",
            "temperature",
        ],
        target_cols="property",
        bounds={"temperature": [850.0, 1000.0]},
        composition_sites={
            "A": {
                "element_columns": {
                    "La": "A_La",
                    "Sr": "A_Sr",
                    "Ba": "A_Ba",
                },
                "min_components": 2,
                "max_components": 2,
                "required_components": ["La"],
                "steps": {"La": 0.01, "Sr": 0.01, "Ba": 0.01},
            },
            "B": {
                "element_columns": {
                    "Fe": "B_Fe",
                    "Co": "B_Co",
                    "Mn": "B_Mn",
                },
                "min_components": 2,
                "max_components": 2,
                "required_components": ["Fe"],
                "steps": {"Fe": 0.01, "Co": 0.01, "Mn": 0.01},
            },
        },
    )
    bo.fit(frame)

    candidates, _ = bo.candidate()
    assert candidates.loc[0, ["A_La", "A_Sr", "A_Ba"]].sum() == pytest.approx(1.0)
    assert candidates.loc[0, ["B_Fe", "B_Co", "B_Mn"]].sum() == pytest.approx(1.0)
    assert (candidates.loc[0, ["A_La", "A_Sr", "A_Ba"]] > 0).sum() == 2
    assert (candidates.loc[0, ["B_Fe", "B_Co", "B_Mn"]] > 0).sum() == 2
    assert candidates.loc[0, "A_La"] > 0
    assert candidates.loc[0, "B_Fe"] > 0


def test_formula_and_element_columns_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        TabularBayesianOptimizer(
            composition_sites={
                "alloy": {
                    "column": "formula",
                    "element_columns": {"Fe": "Fe", "Ti": "Ti"},
                    "elements": ["Fe", "Ti"],
                }
            }
        )


def test_total_100_restores_percentage_columns(monkeypatch) -> None:
    def fake_fit(self, data=None, y=None, **kwargs):
        return self

    def fake_candidate(self, *args, **kwargs):
        return (
            pd.DataFrame(
                {
                    "alloy__ilr__1": [0.0],
                    "alloy__ilr__2": [0.0],
                }
            ),
            1.0,
        )

    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "fit", fake_fit)
    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "candidate", fake_candidate)
    bo = TabularBayesianOptimizer(
        input_cols=["Fe", "Ti", "Al"],
        target_cols="property",
        composition_sites={
            "alloy": {
                "element_columns": {"Fe": "Fe", "Ti": "Ti", "Al": "Al"},
                "total": 100.0,
                "steps": {"Fe": 1.0, "Ti": 1.0, "Al": 1.0},
            }
        },
    )
    bo.fit(_frame())

    candidates, _ = bo.candidate()
    assert candidates.loc[0, ["Fe", "Ti", "Al"]].sum() == pytest.approx(100.0)


def test_public_tabular_optimizer_exposes_element_column_support() -> None:
    from bochan.tabular import TabularBayesianOptimizer as PublicOptimizer

    assert PublicOptimizer is TabularBayesianOptimizer
