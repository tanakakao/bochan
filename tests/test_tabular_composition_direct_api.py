import pandas as pd
import pytest

from bochan.composition import CompositionTransformer
from bochan.tabular.composition_optimizer import (
    TabularBayesianOptimizer,
    _TabularBayesianOptimizer,
)


def _frame():
    return pd.DataFrame(
        {
            "formula": ["Fe0.5Ni0.3Co0.2", "Fe0.4Ni0.4Co0.2"],
            "temperature": [900.0, 950.0],
            "property": [10.0, 12.0],
        }
    )


def test_composition_is_available_from_normal_api():
    transformer = CompositionTransformer(
        elements=["Fe", "Co", "Ni"],
        representation="ilr",
    )
    transformed = transformer.fit_transform(_frame()["formula"])
    assert list(transformed.columns) == [
        "composition__ilr__1",
        "composition__ilr__2",
    ]


def test_tabular_direct_arguments_transform_fit_data(monkeypatch):
    captured = {}

    def fake_fit(self, data=None, y=None, **kwargs):
        captured["data"] = data
        captured["kwargs"] = kwargs
        return self

    monkeypatch.setattr(_TabularBayesianOptimizer, "fit", fake_fit)
    bo = TabularBayesianOptimizer(
        task_type="regression",
        input_cols=["formula", "temperature"],
        target_cols="property",
        bounds={"temperature": [850.0, 1000.0]},
        composition_col="formula",
        composition_elements=["Fe", "Co", "Ni"],
        composition_representation="ilr",
        composition_steps={"Fe": 0.01, "Co": 0.01, "Ni": 0.01},
    )
    assert bo.fit(_frame()) is bo
    assert "formula" not in captured["data"].columns
    assert captured["kwargs"]["input_cols"] == [
        "formula__ilr__1",
        "formula__ilr__2",
        "temperature",
    ]
    assert captured["kwargs"]["bounds"]["formula__ilr__1"] == [-8.0, 8.0]


def test_candidate_is_restored_to_formula(monkeypatch):
    def fake_fit(self, data=None, y=None, **kwargs):
        return self

    def fake_candidate(self, *args, **kwargs):
        return (
            pd.DataFrame(
                {
                    "formula__ilr__1": [0.0],
                    "formula__ilr__2": [0.0],
                    "temperature": [925.0],
                }
            ),
            1.0,
        )

    monkeypatch.setattr(_TabularBayesianOptimizer, "fit", fake_fit)
    monkeypatch.setattr(_TabularBayesianOptimizer, "candidate", fake_candidate)
    bo = TabularBayesianOptimizer(
        input_cols=["formula", "temperature"],
        target_cols="property",
        bounds={"temperature": [850.0, 1000.0]},
        composition_col="formula",
        composition_elements=["Fe", "Co", "Ni"],
        composition_steps={"Fe": 0.01, "Co": 0.01, "Ni": 0.01},
    )
    bo.fit(_frame())
    candidates, _ = bo.candidate()
    assert candidates.loc[0, "formula"]
    fraction_columns = [column for column in candidates if "__fraction__" in column]
    assert candidates.loc[0, fraction_columns].sum() == pytest.approx(1.0)
    assert candidates.loc[0, "temperature"] == 925.0


def test_predict_accepts_raw_formula_dataframe(monkeypatch):
    captured = {}

    def fake_fit(self, data=None, y=None, **kwargs):
        return self

    def fake_predict(self, data, **kwargs):
        captured["data"] = data
        return data

    monkeypatch.setattr(_TabularBayesianOptimizer, "fit", fake_fit)
    monkeypatch.setattr(_TabularBayesianOptimizer, "predict", fake_predict)
    bo = TabularBayesianOptimizer(
        input_cols=["formula", "temperature"],
        target_cols="property",
        bounds={"temperature": [850.0, 1000.0]},
        composition_col="formula",
        composition_elements=["Fe", "Co", "Ni"],
    )
    bo.fit(_frame())
    bo.predict(_frame().drop(columns=["property"]))
    assert "formula" not in captured["data"].columns
    assert "formula__ilr__1" in captured["data"].columns
