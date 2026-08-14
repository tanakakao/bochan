from __future__ import annotations

import inspect

import pandas as pd

from bochan.serving.webapp.targets import missing as web_missing
from bochan.tabular.config import TabularDataConfig
from bochan.tabular.data import prepare_dataframe_missing_values
from bochan.tabular.observation import data as observation_data


def test_prepare_dataframe_missing_values_imputes_features_and_preserves_targets() -> None:
    data = pd.DataFrame(
        {
            "x": [1.0, None, 3.0],
            "material": ["A", None, "A"],
            "target": [10.0, None, 30.0],
        }
    )
    config = TabularDataConfig(
        input_cols=["x", "material"],
        target_cols=[],
        categorical_cols=["material"],
        missing_strategy="impute",
        continuous_impute_strategy="mean",
        categorical_impute_strategy="mode",
        impute_targets=False,
    )

    prepared, impute_values, target_impute_values = (
        prepare_dataframe_missing_values(
            data,
            config,
            input_cols=["x", "material"],
            target_cols=[],
        )
    )

    assert prepared["x"].tolist() == [1.0, 2.0, 3.0]
    assert prepared["material"].tolist() == ["A", "A", "A"]
    assert prepared["target"].isna().tolist() == [False, True, False]
    assert impute_values == {"x": 2.0, "material": "A"}
    assert target_impute_values == {}


def test_prepare_dataframe_missing_values_drops_only_selected_columns() -> None:
    data = pd.DataFrame(
        {
            "x": [1.0, None, 3.0],
            "metadata": [None, "keep", None],
        }
    )
    config = TabularDataConfig(
        input_cols=["x"],
        target_cols=[],
        missing_strategy="drop",
    )

    prepared, _, _ = prepare_dataframe_missing_values(
        data,
        config,
        input_cols=["x"],
        target_cols=[],
    )

    assert prepared.index.tolist() == [0, 2]
    assert prepared["metadata"].isna().tolist() == [True, True]


def test_web_missing_policy_uses_public_tabular_preparation_api() -> None:
    source = inspect.getsource(web_missing)

    assert "prepare_dataframe_missing_values" in source
    assert "_apply_missing_value_strategy" not in source


def test_observation_adapter_uses_public_tabular_preparation_api() -> None:
    source = inspect.getsource(observation_data)

    assert "prepare_dataframe_missing_values" in source
    assert "_apply_missing_value_strategy" not in source
