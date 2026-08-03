import pandas as pd
import pytest

from bochan.tabular.multi_site_composition_optimizer import (
    TabularBayesianOptimizer,
    _CoreTabularBayesianOptimizer,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A_formula": ["La0.7Sr0.3", "La0.5Ba0.5"],
            "B_formula": ["Fe0.6Co0.4", "Fe0.5Mn0.5"],
            "temperature": [900.0, 950.0],
            "property": [10.0, 12.0],
        }
    )


def _sites() -> dict:
    return {
        "A": {
            "column": "A_formula",
            "elements": ["La", "Sr", "Ba"],
            "min_components": 2,
            "max_components": 2,
            "required_components": ["La"],
            "steps": {"La": 0.01, "Sr": 0.01, "Ba": 0.01},
        },
        "B": {
            "column": "B_formula",
            "elements": ["Fe", "Co", "Mn"],
            "min_components": 2,
            "max_components": 2,
            "required_components": ["Fe"],
            "steps": {"Fe": 0.01, "Co": 0.01, "Mn": 0.01},
        },
    }


def test_multi_site_fit_replaces_each_formula_column(monkeypatch) -> None:
    captured = {}

    def fake_fit(self, data=None, y=None, **kwargs):
        captured["data"] = data
        captured["kwargs"] = kwargs
        return self

    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "fit", fake_fit)
    optimizer = TabularBayesianOptimizer(
        input_cols=["A_formula", "B_formula", "temperature"],
        target_cols="property",
        bounds={"temperature": [850.0, 1000.0]},
        composition_sites=_sites(),
    )

    assert optimizer.fit(_frame()) is optimizer
    assert "A_formula" not in captured["data"].columns
    assert "B_formula" not in captured["data"].columns
    assert captured["kwargs"]["input_cols"] == [
        "A__ilr__1",
        "A__ilr__2",
        "B__ilr__1",
        "B__ilr__2",
        "temperature",
    ]
    assert captured["kwargs"]["bounds"]["A__ilr__1"] == [-8.0, 8.0]
    assert captured["kwargs"]["bounds"]["B__ilr__1"] == [-8.0, 8.0]


def test_multi_site_candidate_repairs_each_site_independently(monkeypatch) -> None:
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
    optimizer = TabularBayesianOptimizer(
        input_cols=["A_formula", "B_formula", "temperature"],
        target_cols="property",
        bounds={"temperature": [850.0, 1000.0]},
        composition_sites=_sites(),
    )
    optimizer.fit(_frame())

    candidates, _ = optimizer.candidate()

    assert candidates.loc[0, "A_formula"]
    assert candidates.loc[0, "B_formula"]
    a_columns = [column for column in candidates if column.startswith("A__fraction__")]
    b_columns = [column for column in candidates if column.startswith("B__fraction__")]
    assert candidates.loc[0, a_columns].sum() == pytest.approx(1.0)
    assert candidates.loc[0, b_columns].sum() == pytest.approx(1.0)
    assert sum(candidates.loc[0, a_columns] > 0) == 2
    assert sum(candidates.loc[0, b_columns] > 0) == 2
    assert candidates.loc[0, "A__fraction__La"] > 0
    assert candidates.loc[0, "B__fraction__Fe"] > 0
    assert candidates.loc[0, "temperature"] == 925.0


def test_multi_site_predict_accepts_raw_formula_columns(monkeypatch) -> None:
    captured = {}

    def fake_fit(self, data=None, y=None, **kwargs):
        return self

    def fake_predict(self, data, **kwargs):
        captured["data"] = data
        return data

    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "fit", fake_fit)
    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "predict", fake_predict)
    optimizer = TabularBayesianOptimizer(
        input_cols=["A_formula", "B_formula", "temperature"],
        target_cols="property",
        bounds={"temperature": [850.0, 1000.0]},
        composition_sites=_sites(),
    )
    optimizer.fit(_frame())
    optimizer.predict(_frame().drop(columns=["property"]))

    assert "A_formula" not in captured["data"].columns
    assert "B_formula" not in captured["data"].columns
    assert "A__ilr__1" in captured["data"].columns
    assert "B__ilr__1" in captured["data"].columns


def test_multi_site_rejects_legacy_composition_arguments() -> None:
    with pytest.raises(ValueError, match="not both"):
        TabularBayesianOptimizer(
            composition_col="formula",
            composition_sites=_sites(),
        )


def test_multi_site_requires_unique_columns() -> None:
    sites = _sites()
    sites["B"]["column"] = "A_formula"
    with pytest.raises(ValueError, match="unique"):
        TabularBayesianOptimizer(composition_sites=sites)
