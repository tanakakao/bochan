from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from bochan.serving.webapp.app import VisualizationRequestSchema
from bochan.serving.webapp.reuse_dataset import store_for_model_reuse
from bochan.serving.webapp.visualization_sessions import (
    VisualizationSession,
    _pareto_plot,
    _target_relation,
    register_visualization_session,
)
from bochan.visualization import show_pareto_plot, show_target_relation_plot


class _ExplodingDatasetStore:
    def get(self, dataset_id: str):
        raise AssertionError(f"transient DatasetStore was accessed: {dataset_id}")


def test_model_reuse_uses_retained_session_data_without_dataset_store() -> None:
    data = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [1.0, 2.0],
        }
    )
    session = VisualizationSession(
        optimizer=SimpleNamespace(model=SimpleNamespace()),
        tabular_optimizer=SimpleNamespace(dataset=SimpleNamespace(cat_dims=[])),
        data=data,
        encoded_targets=pd.DataFrame({"y": [1.0, 2.0]}),
        feature_columns=["x"],
        target_columns=["y"],
        target_metadata={"y": {"internal_task": "regression"}},
        hybrid_model=False,
    )
    session.result = {
        "dataset_id": "artifact-dataset",
        "dataset_name": "saved-model",
    }
    register_visualization_session("reuse-source-data", session)

    request = SimpleNamespace(dataset_id="missing-transient-dataset")
    workflow_store = store_for_model_reuse(
        _ExplodingDatasetStore(),
        request,
        "reuse-source-data",
    )
    record = workflow_store.get(request.dataset_id)

    assert record.dataset_id == "missing-transient-dataset"
    assert record.name == "saved-model"
    assert record.profile == {"n_rows": 2, "n_columns": 2}
    pd.testing.assert_frame_equal(record.data, data)
    assert record.data is not data


def test_normal_workflow_keeps_original_dataset_store() -> None:
    store = SimpleNamespace(get=lambda dataset_id: dataset_id)

    assert store_for_model_reuse(store, SimpleNamespace(), None) is store


def test_visualization_request_accepts_pareto_front_toggle() -> None:
    request = VisualizationRequestSchema(
        kind="pareto",
        target_x="strength",
        target_y="conductivity",
        show_pareto_front=True,
    )

    assert request.show_pareto_front is True


def test_regression_target_relation_delegates_pareto_settings(
    monkeypatch,
) -> None:
    expected = SimpleNamespace(name="pareto-figure")
    captured: dict[str, object] = {}

    def fake_show_pareto_plot(
        y: pd.DataFrame,
        target1: str,
        target2: str,
        df_cand=None,
        *,
        directions=None,
        show_pareto_front=False,
        cycle=None,
        range_padding=0.05,
    ):
        captured.update(
            {
                "y": y.copy(),
                "target1": target1,
                "target2": target2,
                "df_cand": df_cand,
                "directions": directions,
                "show_pareto_front": show_pareto_front,
                "cycle": cycle,
                "range_padding": range_padding,
            }
        )
        return expected

    monkeypatch.setattr(
        "bochan.visualization.plots.show_pareto_plot",
        fake_show_pareto_plot,
    )
    data = pd.DataFrame(
        {
            "strength": [1.0, 2.0, 3.0],
            "conductivity": [5.0, 4.0, 3.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "strength_mean": [2.5],
            "strength_std": [0.1],
            "conductivity_mean": [3.5],
            "conductivity_std": [0.2],
        }
    )
    directions = {
        "strength": "maximize",
        "conductivity": "minimize",
    }

    figure = show_target_relation_plot(
        data,
        "strength",
        "conductivity",
        task_types={
            "strength": "regression",
            "conductivity": "regression",
        },
        df_cand=candidates,
        directions=directions,
        show_pareto_front=True,
    )

    assert figure is expected
    assert captured["target1"] == "strength"
    assert captured["target2"] == "conductivity"
    assert captured["df_cand"] is candidates
    assert captured["directions"] is directions
    assert captured["show_pareto_front"] is True
    assert captured["cycle"] is None
    pd.testing.assert_frame_equal(captured["y"], data)


def test_show_pareto_plot_draws_current_data_front_for_mixed_directions() -> None:
    data = pd.DataFrame(
        {
            "strength": [1.0, 2.0, 3.0, 4.0, 4.0, 2.0, 2.0],
            "loss": [5.0, 4.0, 6.0, 3.0, 4.0, 2.0, 2.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "strength_mean": [100.0],
            "strength_std": [1.0],
            "loss_mean": [0.0],
            "loss_std": [0.1],
        }
    )

    figure = show_pareto_plot(
        data,
        "strength",
        "loss",
        df_cand=candidates,
        directions={"strength": "maximize", "loss": "minimize"},
        show_pareto_front=True,
    )
    traces = {trace.name: trace for trace in figure.data}

    assert set(traces) == {"候補点", "入力データ", "パレートフロント"}
    assert list(traces["パレートフロント"].x) == [2.0, 4.0]
    assert list(traces["パレートフロント"].y) == [2.0, 3.0]
    assert 100.0 not in list(traces["パレートフロント"].x)


def test_show_pareto_plot_rejects_unknown_direction() -> None:
    with pytest.raises(ValueError, match="maximize or minimize"):
        show_pareto_plot(
            pd.DataFrame({"x": [1.0], "y": [2.0]}),
            "x",
            "y",
            directions={"x": "up"},
            show_pareto_front=True,
        )


def _pareto_session() -> VisualizationSession:
    session = VisualizationSession(
        optimizer=SimpleNamespace(model=SimpleNamespace()),
        tabular_optimizer=SimpleNamespace(dataset=SimpleNamespace(cat_dims=[])),
        data=pd.DataFrame(
            {
                "strength": [0.0, 300.0, 600.0],
                "conductivity": [10.0, 14.0, 20.0],
            }
        ),
        encoded_targets=pd.DataFrame(),
        feature_columns=["x"],
        target_columns=["strength", "conductivity"],
        target_metadata={
            "strength": {"internal_task": "regression"},
            "conductivity": {"internal_task": "regression"},
        },
        hybrid_model=False,
        rows=[
            {
                "values": {"x": 0.5},
                "predictions": {
                    "strength": {"mean": 420.0, "std": 1200.0},
                    "conductivity": {"mean": 17.5, "std": 80.0},
                },
                "acq_value": 1.2,
            }
        ],
    )
    session.result = {
        "directions": {
            "strength": "maximize",
            "conductivity": "minimize",
        }
    }
    return session


def test_web_pareto_uses_displayed_data_range_and_independent_zoom_axes() -> None:
    figure = _pareto_plot(_pareto_session(), "strength", "conductivity")
    traces = {trace.name: trace for trace in figure.data}

    assert set(traces) == {"候補点", "入力データ"}
    assert list(traces["候補点"].x) == [420.0]
    assert list(traces["候補点"].y) == [17.5]
    assert list(traces["候補点"].error_x.array) == [1200.0]
    assert list(traces["候補点"].error_y.array) == [80.0]
    assert list(figure.layout.xaxis.range) == pytest.approx([-30.0, 630.0])
    assert list(figure.layout.yaxis.range) == pytest.approx([9.5, 20.5])
    assert figure.layout.xaxis.autorange is False
    assert figure.layout.yaxis.autorange is False
    assert figure.layout.xaxis.fixedrange is False
    assert figure.layout.yaxis.fixedrange is False
    assert figure.layout.xaxis.scaleanchor is None
    assert figure.layout.yaxis.scaleanchor is None
    assert figure.layout.yaxis.scaleratio is None
    assert figure.layout.dragmode == "zoom"


def test_web_pareto_uses_effective_directions_for_current_data_front() -> None:
    figure = _pareto_plot(
        _pareto_session(),
        "strength",
        "conductivity",
        show_pareto_front=True,
    )
    traces = {trace.name: trace for trace in figure.data}

    assert list(traces["パレートフロント"].x) == [0.0, 300.0, 600.0]
    assert list(traces["パレートフロント"].y) == [10.0, 14.0, 20.0]


def test_target_relation_web_path_uses_same_pareto_front_and_candidates() -> None:
    figure = _target_relation(
        _pareto_session(),
        "strength",
        "conductivity",
        show_pareto_front=True,
    )
    traces = {trace.name: trace for trace in figure.data}

    assert set(traces) == {"候補点", "入力データ", "パレートフロント"}
    assert list(figure.layout.xaxis.range) == pytest.approx([-30.0, 630.0])
    assert list(figure.layout.yaxis.range) == pytest.approx([9.5, 20.5])
    assert figure.layout.yaxis.scaleanchor is None
