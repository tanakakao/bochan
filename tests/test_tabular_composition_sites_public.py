from __future__ import annotations

import pandas as pd

from bochan.tabular import TabularBayesianOptimizer


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A_site": ["La0.6Sr0.4", "La0.5Sr0.5"],
            "temperature": [1000.0, 1100.0],
            "property": [1.0, 2.0],
        }
    )


def _optimizer() -> TabularBayesianOptimizer:
    return TabularBayesianOptimizer(
        input_cols=["A_site", "temperature"],
        target_cols="property",
        composition_sites={
            "A": {
                "column": "A_site",
                "elements": ["La", "Sr"],
                "representation": "fractions",
                "bounds": {"La": [0.2, 0.8], "Sr": [0.2, 0.8]},
                "min_components": 2,
                "max_components": 2,
            }
        },
    )


def test_public_constructor_builds_composition_adapter() -> None:
    optimizer = _optimizer()
    assert optimizer.composition.enabled
    assert optimizer.composition.sites["A"]["column"] == "A_site"
    assert optimizer.composition.sites["A"]["elements"] == ("La", "Sr")


def test_composition_adapter_round_trip_uses_canonical_state() -> None:
    optimizer = _optimizer()
    transformed = optimizer.composition.prepare_frame(
        _frame(),
        fit_transformers=True,
    )
    transformer = optimizer.composition.transformers["A"]
    assert "A_site" not in transformed.columns
    assert set(transformer.feature_names_ or ()).issubset(transformed.columns)

    restored = optimizer.inverse_compositions(
        transformed.head(1),
        repair=True,
    )
    assert "A_site" in restored.columns
    assert restored.loc[0, "A_site"]


def test_public_optimizer_module_is_canonical_core() -> None:
    assert TabularBayesianOptimizer.__module__ == "bochan.tabular.optimizer.core"
