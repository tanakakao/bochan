from __future__ import annotations

from types import SimpleNamespace

import pytest

pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")
pytest.importorskip("plotly")


class _RecordingRegressionModel:
    def __init__(self) -> None:
        self.seen_rows: list[int] = []

    def posterior(self, X, **kwargs):
        del kwargs
        self.seen_rows.append(int(X.shape[0]))
        mean = X[:, :1]
        return SimpleNamespace(
            mean=mean,
            variance=torch.zeros_like(mean),
        )


class _FailIfCalledModel:
    def posterior(self, X, **kwargs):  # pragma: no cover - failure path only
        del X, kwargs
        raise AssertionError("classification-only visualization must not request posterior predictions")


def test_web_visualization_subset_caps_large_training_data() -> None:
    from bochan.serving.webapp.target_results import _visualization_subset

    train_x = torch.arange(2500, dtype=torch.double).reshape(-1, 1)
    targets = pd.DataFrame({"property": train_x[:, 0].numpy()})

    sampled_x, sampled_targets, original_count = _visualization_subset(
        train_x,
        targets,
        max_points=2000,
    )

    assert original_count == 2500
    assert sampled_x.shape == (2000, 1)
    assert len(sampled_targets) == 2000
    assert sampled_x[0, 0].item() == pytest.approx(0.0)
    assert sampled_x[-1, 0].item() == pytest.approx(2499.0)


def test_web_yy_plot_predicts_only_sampled_rows() -> None:
    from bochan.serving.webapp.target_results import _build_visualizations

    model = _RecordingRegressionModel()
    optimizer = SimpleNamespace(model=model)
    train_x = torch.arange(10, dtype=torch.double).reshape(-1, 1)
    targets = pd.DataFrame({"property": train_x[:, 0].numpy()})

    figures, warnings = _build_visualizations(
        optimizer=optimizer,
        train_x=train_x,
        original_targets=targets,
        target_columns=["property"],
        target_metadata={"property": {"internal_task": "regression"}},
        hybrid_model=True,
        max_points=4,
    )

    assert warnings == []
    assert model.seen_rows == [4]
    assert len(figures) == 1
    assert "10 点から 4 点" in figures[0]["description"]


def test_classification_only_web_visualization_skips_prediction() -> None:
    from bochan.serving.webapp.target_results import _build_visualizations

    optimizer = SimpleNamespace(model=_FailIfCalledModel())
    train_x = torch.arange(8, dtype=torch.double).reshape(-1, 1)
    targets = pd.DataFrame({"label": [0, 1, 0, 1, 0, 1, 0, 1]})

    figures, warnings = _build_visualizations(
        optimizer=optimizer,
        train_x=train_x,
        original_targets=targets,
        target_columns=["label"],
        target_metadata={"label": {"internal_task": "binary"}},
        hybrid_model=True,
    )

    assert figures == []
    assert len(warnings) == 1
    assert "専用可視化は未接続" in warnings[0]
