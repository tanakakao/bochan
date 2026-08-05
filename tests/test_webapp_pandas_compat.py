from __future__ import annotations

import importlib

import pandas as pd

# Import the core optimizer before the Web package. This reproduces the cached
# ``dataframe_to_tensors`` reference that previously bypassed the Web wrapper.
optimizer_module = importlib.import_module("bochan.tabular.optimizer")

from bochan.serving.webapp import app as _app  # noqa: E402,F401
from bochan.tabular import converter  # noqa: E402
from bochan.tabular.config import TabularDataConfig  # noqa: E402


def test_web_compat_updates_cached_optimizer_converter_alias() -> None:
    frame = pd.DataFrame(
        {
            "composition_coordinate": [0.1, 0.2, 0.3],
            "phase": pd.Series(["alpha", "beta", "beta"], dtype="string"),
        }
    )
    config = TabularDataConfig(
        input_cols=["composition_coordinate", "phase"],
        categorical_cols=["phase"],
        category_maps={"phase": {"alpha": 0, "beta": 1}},
        encode_categories=True,
    )

    assert optimizer_module.dataframe_to_tensors is converter.dataframe_to_tensors

    dataset = optimizer_module.dataframe_to_tensors(frame, config)

    assert dataset.X[:, 1].tolist() == [0.0, 1.0, 1.0]
    assert dataset.category_maps == {"phase": {"alpha": 0, "beta": 1}}
