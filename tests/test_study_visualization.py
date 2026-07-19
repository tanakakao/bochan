import pytest
import torch

from bochan.api import BochanStudy
from bochan.visualization.study import (
    show_optimization_history_study,
    show_pareto_front_study,
    study_history_dataframe,
    study_pareto_dataframe,
)


def test_study_history_dataframe_contains_best_so_far():
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    study = BochanStudy(bounds=bounds)
    study.add_observations(
        torch.tensor([[0.1], [0.2], [0.3], [0.4]], dtype=torch.double),
        torch.tensor([1.0, 0.5, 1.5, 1.2], dtype=torch.double),
        metadata=[
            {"cycle": 0},
            {"cycle": 0},
            {"cycle": 1},
            {"cycle": 1},
        ],
    )

    df = study_history_dataframe(study, target_name="score")

    assert df["score"].tolist() == pytest.approx([1.0, 0.5, 1.5, 1.2])
    assert df["best_value"].tolist() == pytest.approx([1.0, 1.0, 1.5, 1.5])
    assert df["is_best"].tolist() == [True, False, True, False]
    assert df["cycle"].tolist() == [0, 0, 1, 1]


def test_study_pareto_dataframe_flags_non_dominated_trials():
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    study = BochanStudy(
        bounds=bounds,
        model_config={"task_type": "multi_objective", "model_type": "base"},
    )
    study.add_observations(
        torch.tensor([[0.1], [0.2], [0.3], [0.4]], dtype=torch.double),
        torch.tensor(
            [[1.0, 1.0], [2.0, 3.0], [3.0, 2.0], [2.5, 2.5]],
            dtype=torch.double,
        ),
    )

    df = study_pareto_dataframe(
        study,
        directions=["maximize", "minimize"],
        target_cols=["strength", "cost"],
    )

    assert df.columns.tolist() == ["trial_id", "strength", "cost", "is_pareto"]
    assert df.loc[df["is_pareto"], "trial_id"].tolist() == [0, 2]


def test_study_optimization_history_plot_contains_observed_and_best_traces():
    pytest.importorskip("plotly")
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    study = BochanStudy(bounds=bounds)
    study.add_observations(
        torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.double),
        torch.tensor([1.0, 0.5, 1.5], dtype=torch.double),
    )

    fig = show_optimization_history_study(study, target_name="score")

    assert [trace.name for trace in fig.data] == [
        "observed",
        "best so far",
        "new best",
    ]
    assert list(fig.data[1].y) == pytest.approx([1.0, 1.0, 1.5])


def test_study_pareto_plot_highlights_front():
    pytest.importorskip("plotly")
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    study = BochanStudy(
        bounds=bounds,
        model_config={"task_type": "multi_objective", "model_type": "base"},
    )
    study.add_observations(
        torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.double),
        torch.tensor([[1.0, 1.0], [2.0, 3.0], [3.0, 2.0]], dtype=torch.double),
    )

    fig = show_pareto_front_study(
        study,
        directions=["maximize", "minimize"],
        target_cols=["strength", "cost"],
    )

    assert [trace.name for trace in fig.data] == ["completed", "Pareto front"]
    assert set(fig.data[1].customdata[:, 0]) == {0, 2}
