from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from bochan.serving.workbench import datasets, workflow_utils


def test_workbench_dataset_store_and_profile() -> None:
    frame = pd.DataFrame(
        {
            "temperature": [900.0, 950.0, 1000.0],
            "atmosphere": ["air", "n2", "air"],
        }
    )
    record = datasets.build_dataset_record(
        data=frame,
        name="sample.csv",
        source_type="csv",
    )
    store = datasets.DatasetStore()
    store.add(record)

    assert store.get(record.dataset_id) is record
    assert record.profile["n_rows"] == 3
    assert record.profile["n_columns"] == 2
    assert datasets.dataframe_preview(frame, limit=2) == [
        {"temperature": 900.0, "atmosphere": "air"},
        {"temperature": 950.0, "atmosphere": "n2"},
    ]


def test_workbench_feature_encoding_preserves_numeric_and_categorical_metadata() -> None:
    frame = pd.DataFrame(
        {
            "temperature": [900.0, 950.0, 1000.0],
            "atmosphere": ["air", "n2", "air"],
        }
    )
    search_space = [
        SimpleNamespace(
            name="temperature",
            type="numeric",
            lower=850.0,
            upper=1050.0,
            step=25.0,
            fixed=False,
            fixed_value=None,
            categories=None,
        ),
        SimpleNamespace(
            name="atmosphere",
            type="categorical",
            lower=None,
            upper=None,
            step=None,
            fixed=False,
            fixed_value=None,
            categories=["air", "n2"],
        ),
    ]

    encoded = workflow_utils._encode_features(
        data=frame,
        feature_columns=["temperature", "atmosphere"],
        search_space=search_space,
    )

    assert encoded["bounds"] == [[850.0, 0.0], [1050.0, 1.0]]
    assert encoded["numeric_indices"] == [0]
    assert encoded["cat_dims"] == [1]
    assert encoded["steps"] == {0: 25.0}
    assert encoded["category_maps"] == {"atmosphere": {"air": 0, "n2": 1}}


def test_workbench_dataset_loader_rejects_non_web_sources() -> None:
    with pytest.raises(ValueError, match="Unsupported source_type"):
        datasets.load_dataframe_from_payload(source_type="sqlite")  # type: ignore[arg-type]


def test_webapp_import_uses_workbench_dataset_services() -> None:
    pytest.importorskip("fastapi")

    app_module = importlib.import_module("bochan.serving.webapp.app")

    assert app_module.DatasetStore is datasets.DatasetStore
    assert app_module.build_dataset_record is datasets.build_dataset_record
    assert app_module.dataframe_preview is datasets.dataframe_preview
    assert app_module.load_dataframe_from_payload is datasets.load_dataframe_from_payload


def _desktop_import_offenders(root: Path) -> list[str]:
    return [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "bochan.desktop" in path.read_text(encoding="utf-8")
    ]


def test_bochan_package_has_no_desktop_imports() -> None:
    source_root = Path(__file__).parents[1] / "src" / "bochan"

    assert _desktop_import_offenders(source_root) == []


def test_web_tests_have_no_desktop_imports() -> None:
    tests_root = Path(__file__).parent

    assert _desktop_import_offenders(tests_root) == []
