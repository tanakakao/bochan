from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")
pytest.importorskip("botorch")
pytest.importorskip("fastapi")

from bochan.desktop.services import DatasetStore, build_dataset_record  # noqa: E402
from bochan.serving.webapp.app import RegressionRunRequest  # noqa: E402
from bochan.serving.webapp.workflows import run_regression_web_workflow  # noqa: E402


def _store_with_three_class_data() -> tuple[DatasetStore, str]:
    x = torch.linspace(0.0, 1.0, 15, dtype=torch.double).numpy()
    y = (x >= 1.0 / 3.0).astype(int) + (x >= 2.0 / 3.0).astype(int)
    data = pd.DataFrame({"x": x, "y": y})
    record = build_dataset_record(
        data=data,
        name="three_class_perturbation.csv",
        source_type="csv",
    )
    store = DatasetStore()
    store.add(record)
    return store, record.dataset_id


def _request(
    dataset_id: str,
    *,
    task_type: str,
) -> RegressionRunRequest:
    setting = {
        "target": "y",
        "task_type": task_type,
        "optimize": True,
        "direction": "maximize",
        "goal": "none",
        "value": None,
    }
    if task_type == "classification":
        setting["target_classes"] = [2]
    else:
        setting["class_order"] = [0, 1, 2]

    return RegressionRunRequest(
        dataset_id=dataset_id,
        feature_columns=["x"],
        target_column="y",
        target_columns=["y"],
        model_type="base",
        model_kwargs={"web_target_settings": [setting]},
        fit_maxiter=8,
        normalize=True,
        outcome_transform=True,
        input_perturbation=True,
        n_w=8,
        perturbation_std=0.15,
        search_space=[
            {
                "name": "x",
                "type": "numeric",
                "lower": 0.0,
                "upper": 1.0,
                "fixed": False,
            }
        ],
        acquisition={
            "name": "variance",
            "beta": 2.0,
            "acqf_kwargs": {
                "web_family": "active_learning",
                "web_risk_type": "none",
                "web_risk_alpha": 0.2,
            },
        },
        optimizer={
            "name": "optimize_acqf",
            "q": 2,
            "num_restarts": 2,
            "raw_samples": 32,
            "sequential": True,
        },
    )


@pytest.mark.parametrize("task_type", ["classification", "ordinal"])
def test_web_three_class_active_learning_runs_with_input_perturbation(
    task_type: str,
) -> None:
    """Multiclass and ordinal Web active learning must keep candidates finite."""

    torch.manual_seed(0)
    store, dataset_id = _store_with_three_class_data()
    result = run_regression_web_workflow(
        _request(dataset_id, task_type=task_type),
        store,
    )

    assert result["candidates"]
    assert len(result["candidates"]) == 2
    assert result["metadata"]["input_perturbation_risk_type"] == "none"
