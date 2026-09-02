from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd
import pytest
import torch

from bochan.api import BayesianOptimizer, ModelConfig, MultiOutputConfig
from bochan.api.modeling.build import build_model
from bochan.serving.fastapi.schemas.requests import FitModelRequest, TellRequest
from bochan.serving.fastapi.schemas.tabular import TabularFitModelRequest
from bochan.tabular import TabularDataConfig
from bochan.tabular.data import dataframe_to_tensors


class CaptureModel:
    def __init__(self, train_X: Any, train_Y: Any, train_Yvar: Any | None = None, **_: Any) -> None:
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_Yvar = train_Yvar


class CaptureWrapper:
    def __init__(self, submodels: list[Any]) -> None:
        self.models = submodels


def _capture_wrapper(*, submodels: list[Any], **_: Any) -> CaptureWrapper:
    return CaptureWrapper(submodels)


def _capture_config() -> ModelConfig:
    return ModelConfig(
        task_type="regression",
        model_type="capture",
        model_factory=CaptureModel,
    )


def test_dataframe_target_variance_columns_are_not_features() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y1": [1.0, 1.5, 2.0],
            "y2": [2.0, 2.5, 3.0],
            "y1_var": [0.01, 0.02, 0.03],
            "y2_var": [0.04, 0.05, 0.06],
        }
    )
    dataset = dataframe_to_tensors(
        frame,
        TabularDataConfig(
            target_cols=["y1", "y2"],
            target_variance_cols=["y1_var", "y2_var"],
        ),
    )

    assert dataset.feature_names == ["x"]
    assert dataset.Yvar is not None
    assert dataset.Yvar.shape == dataset.Y.shape == torch.Size([3, 2])
    torch.testing.assert_close(
        dataset.Yvar,
        torch.tensor(
            [[0.01, 0.04], [0.02, 0.05], [0.03, 0.06]],
            dtype=torch.double,
        ),
    )


@pytest.mark.parametrize(
    ("variance_values", "match"),
    [
        ([0.01, 0.0, 0.03], "strictly positive"),
        ([0.01, float("inf"), 0.03], "finite"),
    ],
)
def test_dataframe_target_variance_values_are_validated(
    variance_values: list[float],
    match: str,
) -> None:
    frame = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0], "y_var": variance_values}
    )
    with pytest.raises(ValueError, match=match):
        dataframe_to_tensors(
            frame,
            TabularDataConfig(
                input_cols=["x"],
                target_cols="y",
                target_variance_cols="y_var",
            ),
        )


def test_dataframe_target_variance_column_contract_is_validated() -> None:
    frame = pd.DataFrame(
        {"x": [0.0, 1.0], "y1": [1.0, 2.0], "y2": [2.0, 3.0], "v": [0.01, 0.02]}
    )
    with pytest.raises(ValueError, match="exactly one variance column"):
        dataframe_to_tensors(
            frame,
            TabularDataConfig(
                input_cols=["x"],
                target_cols=["y1", "y2"],
                target_variance_cols=["v"],
            ),
        )
    with pytest.raises(ValueError, match="must not be included in input_cols"):
        dataframe_to_tensors(
            frame,
            TabularDataConfig(
                input_cols=["x", "v"],
                target_cols="y1",
                target_variance_cols="v",
            ),
        )


def test_build_model_forwards_scalar_and_wide_train_yvar() -> None:
    X = torch.rand(4, 2, dtype=torch.double)
    Y = torch.rand(4, 2, dtype=torch.double)
    Yvar = torch.full_like(Y, 0.01)
    model = build_model(X, Y, _capture_config(), train_Yvar=Yvar).model

    assert model.train_Yvar is Yvar
    assert model.train_Y.shape == torch.Size([4, 2])


def test_independent_multi_output_slices_train_yvar_per_output() -> None:
    X = torch.rand(4, 2, dtype=torch.double)
    Y = torch.rand(4, 2, dtype=torch.double)
    Yvar = torch.tensor(
        [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06], [0.07, 0.08]],
        dtype=torch.double,
    )
    output = _capture_config()
    config = replace(
        output,
        task_type="multi_objective",
        multi_output_config=MultiOutputConfig(
            output_configs=[output, output],
            wrapper_factory=_capture_wrapper,
        ),
    )
    wrapper = build_model(X, Y, config, train_Yvar=Yvar).model

    assert len(wrapper.models) == 2
    torch.testing.assert_close(wrapper.models[0].train_Yvar, Yvar[:, :1])
    torch.testing.assert_close(wrapper.models[1].train_Yvar, Yvar[:, 1:2])


def test_bayesian_optimizer_update_requires_consistent_known_noise() -> None:
    optimizer = BayesianOptimizer(_capture_config())
    optimizer.train_X = torch.zeros(2, 1)
    optimizer.train_Y = torch.zeros(2, 1)
    optimizer.train_Yvar = torch.full((2, 1), 0.01)

    with pytest.raises(ValueError, match="new_Yvar is required"):
        optimizer.update_data(torch.ones(1, 1), torch.ones(1, 1))

    optimizer.update_data(
        torch.ones(1, 1),
        torch.ones(1, 1),
        torch.full((1, 1), 0.02),
    )
    assert optimizer.train_Yvar.shape == torch.Size([3, 1])
    torch.testing.assert_close(optimizer.train_Yvar[-1], torch.tensor([0.02]))


def test_public_optimizer_tell_appends_known_variance() -> None:
    optimizer = BayesianOptimizer(_capture_config())
    optimizer.train_X = torch.zeros(2, 1)
    optimizer.train_Y = torch.zeros(2, 1)
    optimizer.train_Yvar = torch.full((2, 1), 0.01)
    optimizer.observations = None

    optimizer.tell(
        torch.ones(1, 1),
        torch.ones(1, 1),
        torch.full((1, 1), 0.02),
        refit=False,
    )

    assert optimizer.train_Yvar.shape == torch.Size([3, 1])
    torch.testing.assert_close(optimizer.train_Yvar[-1], torch.tensor([0.02]))


def test_fastapi_tensor_schemas_accept_known_variance() -> None:
    request = FitModelRequest.model_validate(
        {
            "model_config": {"model_type": "base", "task_type": "regression"},
            "train_X": [[0.0], [1.0]],
            "train_Y": [[1.0], [2.0]],
            "train_Yvar": [[0.01], [0.02]],
        }
    )
    assert request.train_Yvar == [[0.01], [0.02]]
    tell = TellRequest(new_X=[[2.0]], new_Y=[[3.0]], new_Yvar=[[0.03]])
    assert tell.new_Yvar == [[0.03]]


def test_tabular_fastapi_schema_validates_variance_columns() -> None:
    payload = {
        "data": [
            {"x": 0.0, "y1": 1.0, "y2": 2.0, "v1": 0.01, "v2": 0.02},
            {"x": 1.0, "y1": 1.5, "y2": 2.5, "v1": 0.02, "v2": 0.03},
        ],
        "model_config": {"model_type": "deepkernel", "task_type": "multi_objective"},
        "input_cols": ["x"],
        "target_cols": ["y1", "y2"],
        "target_variance_cols": ["v1", "v2"],
    }
    request = TabularFitModelRequest.model_validate(payload)
    assert request.target_variance_cols == ["v1", "v2"]

    bad = dict(payload)
    bad["target_variance_cols"] = ["v1"]
    with pytest.raises(ValueError, match="exactly one variance column"):
        TabularFitModelRequest.model_validate(bad)
