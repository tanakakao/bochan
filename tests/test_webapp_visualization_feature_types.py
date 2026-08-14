from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from bochan.serving.webapp.services import visualization_sessions


def _session(
    data: pd.DataFrame,
    *,
    feature_columns: list[str],
    cat_dims: list[int] | None = None,
    category_maps: dict[str, object] | None = None,
) -> SimpleNamespace:
    dataset = SimpleNamespace(
        cat_dims=list(cat_dims or []),
        category_maps=dict(category_maps or {}),
    )
    return SimpleNamespace(
        tabular_optimizer=SimpleNamespace(dataset=dataset),
        data=data,
        feature_columns=list(feature_columns),
        target_columns=["property"],
        target_metadata={"property": {"internal_task": "regression"}},
        feature_constraints=[],
    )


def test_numeric_features_use_source_dtypes_when_model_dimensions_expand() -> None:
    data = pd.DataFrame(
        {
            "formula": ["Fe2O3", "Al2O3"],
            "temperature": [900.0, 1000.0],
            "enabled": [True, False],
            "grade": ["A", "B"],
            "property": [1.0, 2.0],
        }
    )
    session = _session(
        data,
        feature_columns=["formula", "temperature", "enabled", "grade"],
        # Simulate transformed coordinates whose categorical index no longer
        # matches the original source-column positions.
        cat_dims=[1],
        category_maps={"grade": {"A": 0, "B": 1}},
    )

    assert visualization_sessions._numeric_features(session) == ["temperature"]


def test_named_category_map_overrides_numeric_source_dtype() -> None:
    data = pd.DataFrame(
        {
            "code": [10, 20],
            "value": [0.1, 0.2],
            "property": [1.0, 2.0],
        }
    )
    session = _session(
        data,
        feature_columns=["code", "value"],
        category_maps={"code": {10: 0, 20: 1}},
    )

    assert visualization_sessions._numeric_features(session) == ["value"]


def test_missing_source_feature_keeps_positional_fallback() -> None:
    data = pd.DataFrame(
        {
            "temperature": [900.0, 1000.0],
            "property": [1.0, 2.0],
        }
    )
    session = _session(
        data,
        feature_columns=["temperature", "derived_numeric", "derived_category"],
        cat_dims=[2],
    )

    assert visualization_sessions._numeric_features(session) == [
        "temperature",
        "derived_numeric",
    ]


def test_visualization_options_handle_string_source_without_runtime_patch() -> None:
    data = pd.DataFrame(
        {
            "formula": ["Fe2O3", "Al2O3"],
            "temperature": [900.0, 1000.0],
            "property": [1.0, 2.0],
        }
    )
    session = _session(
        data,
        feature_columns=["formula", "temperature"],
        cat_dims=[],
    )

    options = visualization_sessions.visualization_options(session)

    assert options["numeric_features"] == ["temperature"]
    assert options["feature_controls"]["formula"]["kind"] == "categorical"
    assert options["feature_controls"]["temperature"]["kind"] == "numeric"


def test_feature_type_compatibility_module_is_removed() -> None:
    root = Path(visualization_sessions.__file__).resolve().parent

    assert not (root / "visualization_feature_types.py").exists()
    assert not (root / "runtime_adapters.py").exists()
    assert (
        visualization_sessions._numeric_features.__module__
        == "bochan.serving.webapp.services.visualization_sessions"
    )
