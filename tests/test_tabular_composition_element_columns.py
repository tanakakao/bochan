from __future__ import annotations

import pandas as pd
import pytest

from bochan.tabular import TabularBayesianOptimizer


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "La_col": [60.0, 50.0, 70.0],
            "Sr_col": [40.0, 50.0, 30.0],
            "temperature": [900.0, 1000.0, 1100.0],
            "property": [1.0, 2.0, 3.0],
        }
    )


def _optimizer(*, input_basis: str = "atomic_fraction") -> TabularBayesianOptimizer:
    return TabularBayesianOptimizer(
        input_cols=["La_col", "Sr_col", "temperature"],
        target_cols="property",
        composition_sites={
            "A": {
                "element_columns": {"La": "La_col", "Sr": "Sr_col"},
                "input_basis": input_basis,
                "representation": "fractions",
                "bounds": {"La": [0.2, 0.8], "Sr": [0.2, 0.8]},
                "min_components": 2,
                "max_components": 2,
            }
        },
    )


def test_element_columns_are_normalized_and_transformed_by_adapter() -> None:
    optimizer = _optimizer()
    transformed = optimizer.composition.prepare_frame(_frame(), fit_transformers=True)
    config = optimizer.composition.sites["A"]
    assert config["input_kind"] == "element_columns"
    assert config["element_columns"] == {"La": "La_col", "Sr": "Sr_col"}
    assert "La_col" not in transformed.columns
    assert "Sr_col" not in transformed.columns
    transformer = optimizer.composition.transformers["A"]
    assert set(transformer.feature_names_ or ()).issubset(transformed.columns)


def test_weight_basis_is_converted_before_fraction_coordinates() -> None:
    optimizer = _optimizer(input_basis="weight_fraction")
    transformed = optimizer.composition.prepare_frame(_frame(), fit_transformers=True)
    la_fraction = transformed.loc[0, "A__fraction__La"]
    sr_fraction = transformed.loc[0, "A__fraction__Sr"]
    expected_la = (60.0 / 138.90547) / ((60.0 / 138.90547) + (40.0 / 87.62))
    assert la_fraction == pytest.approx(expected_la)
    assert sr_fraction == pytest.approx(1.0 - expected_la)


def test_inverse_compositions_restores_native_element_columns() -> None:
    optimizer = _optimizer()
    transformed = optimizer.composition.prepare_frame(_frame(), fit_transformers=True)
    candidate = transformed.head(1).copy()
    candidate.loc[:, "A__fraction__La"] = 0.6
    candidate.loc[:, "A__fraction__Sr"] = 0.4
    restored = optimizer.inverse_compositions(candidate, repair=True)
    assert restored.loc[0, "La_col"] == pytest.approx(0.6)
    assert restored.loc[0, "Sr_col"] == pytest.approx(0.4)
    assert "__bochan_A_composition_formula__" not in restored.columns


def test_element_columns_are_removed_from_model_bounds_and_categories() -> None:
    optimizer = _optimizer()
    transformed = optimizer.composition.prepare_frame(_frame(), fit_transformers=True)
    bounds = optimizer.composition.expanded_bounds(
        {
            "La_col": [0.0, 100.0],
            "Sr_col": [0.0, 100.0],
            "temperature": [800.0, 1200.0],
        },
        transformed,
    )
    assert "La_col" not in bounds
    assert "Sr_col" not in bounds
    assert "temperature" in bounds
    assert optimizer.composition.resolve_categorical_cols(
        ["La_col", "Sr_col"],
        default_categorical_cols=["La_col", "Sr_col"],
    ) == []
