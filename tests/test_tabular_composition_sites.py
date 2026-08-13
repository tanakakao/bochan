from __future__ import annotations

import pandas as pd
import pytest

from bochan.tabular import TabularBayesianOptimizer


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A_site": ["La0.6Sr0.4", "La0.5Sr0.5", "La0.7Sr0.3"],
            "B_site": ["La0.5Sr0.3Ca0.2", "La0.4Sr0.4Ca0.2", "La0.6Sr0.2Ca0.2"],
            "temperature": [900.0, 1000.0, 1100.0],
            "property": [1.0, 2.0, 3.0],
        }
    )


def _optimizer() -> TabularBayesianOptimizer:
    return TabularBayesianOptimizer(
        input_cols=["A_site", "B_site", "temperature"],
        target_cols="property",
        composition_sites={
            "A": {
                "column": "A_site",
                "elements": ["La", "Sr"],
                "representation": "clr",
            },
            "B": {
                "column": "B_site",
                "elements": ["La", "Sr", "Ca"],
                "representation": "fractions",
                "bounds": {
                    "La": [0.3, 0.8],
                    "Sr": [0.1, 0.5],
                    "Ca": [0.1, 0.4],
                },
                "min_components": 3,
                "max_components": 3,
            },
        },
    )


def test_multi_site_adapter_replaces_formula_columns_and_expands_bounds() -> None:
    optimizer = _optimizer()
    transformed = optimizer.composition.prepare_frame(
        _frame(),
        fit_transformers=True,
    )
    assert "A_site" not in transformed.columns
    assert "B_site" not in transformed.columns
    assert "temperature" in transformed.columns
    assert set(optimizer.composition.transformers) == {"A", "B"}

    resolved_cols = optimizer.composition.replace_input_cols(
        ["A_site", "B_site", "temperature"]
    )
    expected = [
        *(optimizer.composition.transformers["A"].feature_names_ or ()),
        *(optimizer.composition.transformers["B"].feature_names_ or ()),
        "temperature",
    ]
    assert resolved_cols == expected

    bounds = optimizer.composition.expanded_bounds(
        {"temperature": [800.0, 1200.0]},
        transformed,
    )
    assert "temperature" in bounds
    for transformer in optimizer.composition.transformers.values():
        for name in transformer.feature_names_ or ():
            assert name in bounds


def test_inverse_compositions_repairs_each_site_independently() -> None:
    optimizer = _optimizer()
    transformed = optimizer.composition.prepare_frame(
        _frame(),
        fit_transformers=True,
    )
    candidate = transformed.head(1).copy()
    b_transformer = optimizer.composition.transformers["B"]
    b_names = list(b_transformer.feature_names_ or ())
    assert len(b_names) == 3
    candidate.loc[:, b_names] = [0.95, 0.03, 0.02]

    restored = optimizer.inverse_compositions(candidate, repair=True)
    assert restored.loc[0, "A_site"]
    assert restored.loc[0, "B_site"]
    fractions = restored.loc[
        0,
        [
            f"{b_transformer.prefix}__fraction__La",
            f"{b_transformer.prefix}__fraction__Sr",
            f"{b_transformer.prefix}__fraction__Ca",
        ],
    ].astype(float)
    assert fractions.sum() == pytest.approx(1.0)
    assert fractions.iloc[0] <= 0.8 + 1e-7
    assert fractions.iloc[1] >= 0.1 - 1e-7
    assert fractions.iloc[2] >= 0.1 - 1e-7


def test_transform_compositions_accepts_raw_formula_columns() -> None:
    optimizer = _optimizer()
    optimizer.composition.prepare_frame(_frame(), fit_transformers=True)
    transformed = optimizer.transform_compositions(_frame().iloc[:2])
    assert "A_site" not in transformed.columns
    assert "B_site" not in transformed.columns
    for transformer in optimizer.composition.transformers.values():
        assert set(transformer.feature_names_ or ()).issubset(transformed.columns)
