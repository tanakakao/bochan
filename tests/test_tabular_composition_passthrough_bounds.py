import pandas as pd

from bochan.tabular import TabularBayesianOptimizer
from bochan.tabular.optimizer_api import (
    TabularBayesianOptimizer as _CoreTabularBayesianOptimizer,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "formula": ["Fe0.5Co0.3Ni0.2", "Fe0.4Co0.4Ni0.2"],
            "temperature": [900.0, 1000.0],
            "furnace": ["A", "B"],
            "property": [10.0, 12.0],
        }
    )


def test_legacy_composition_infers_passthrough_bounds_with_descriptors(
    monkeypatch,
) -> None:
    captured = {}

    def fake_fit(self, data=None, y=None, **kwargs):
        captured["data"] = data
        captured["kwargs"] = kwargs
        return self

    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "fit", fake_fit)
    bo = TabularBayesianOptimizer(
        input_cols=["formula", "temperature"],
        target_cols="property",
        composition_col="formula",
        composition_representation="ilr",
        composition_include_descriptors=True,
        composition_descriptor_properties=[
            "atomic_number",
            "atomic_weight",
            "electronegativity",
        ],
        composition_element_properties={
            "electronegativity": {
                "Fe": 1.83,
                "Co": 1.88,
                "Ni": 1.91,
            }
        },
    )

    assert bo.fit(_frame()) is bo
    assert captured["kwargs"]["bounds"]["temperature"] == [900.0, 1000.0]
    assert "formula__ilr__1" in captured["kwargs"]["bounds"]
    assert "formula__atomic_number__mean" in captured["kwargs"]["bounds"]


def test_explicit_passthrough_bound_is_preserved(monkeypatch) -> None:
    captured = {}

    def fake_fit(self, data=None, y=None, **kwargs):
        captured["kwargs"] = kwargs
        return self

    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "fit", fake_fit)
    bo = TabularBayesianOptimizer(
        input_cols=["formula", "temperature"],
        target_cols="property",
        bounds={"temperature": [850.0, 1050.0]},
        composition_col="formula",
        composition_representation="ilr",
    )

    bo.fit(_frame())
    assert captured["kwargs"]["bounds"]["temperature"] == [850.0, 1050.0]


def test_categorical_passthrough_bound_matches_label_encoding(monkeypatch) -> None:
    captured = {}

    def fake_fit(self, data=None, y=None, **kwargs):
        captured["kwargs"] = kwargs
        return self

    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "fit", fake_fit)
    bo = TabularBayesianOptimizer(
        input_cols=["formula", "temperature", "furnace"],
        target_cols="property",
        categorical_cols=["furnace"],
        composition_col="formula",
    )

    bo.fit(_frame())
    assert captured["kwargs"]["bounds"]["temperature"] == [900.0, 1000.0]
    assert captured["kwargs"]["bounds"]["furnace"] == [0.0, 1.0]


def test_multi_site_composition_infers_passthrough_bounds(monkeypatch) -> None:
    captured = {}
    frame = pd.DataFrame(
        {
            "A_formula": ["La0.6Sr0.4", "La0.7Sr0.3"],
            "temperature": [900.0, 1000.0],
            "property": [10.0, 12.0],
        }
    )

    def fake_fit(self, data=None, y=None, **kwargs):
        captured["kwargs"] = kwargs
        return self

    monkeypatch.setattr(_CoreTabularBayesianOptimizer, "fit", fake_fit)
    bo = TabularBayesianOptimizer(
        input_cols=["A_formula", "temperature"],
        target_cols="property",
        composition_sites={
            "A": {
                "column": "A_formula",
                "elements": ["La", "Sr"],
                "representation": "ilr",
            }
        },
    )

    bo.fit(frame)
    assert captured["kwargs"]["bounds"]["temperature"] == [900.0, 1000.0]
    assert "A__ilr__1" in captured["kwargs"]["bounds"]
