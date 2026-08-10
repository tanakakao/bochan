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


def _target_setting(
    *,
    task_type: str,
    family: str,
    direction: str = "maximize",
    target_classes: list[int] | None = None,
) -> dict[str, object]:
    lse = family == "level_set_estimation"
    setting: dict[str, object] = {
        "target": "y",
        "task_type": task_type,
        "optimize": True,
        "direction": direction,
        "goal": "above" if lse else "none",
        "value": None,
    }
    if task_type == "classification":
        setting["target_classes"] = list(target_classes or [2])
        if lse:
            setting["value"] = 0.5
    else:
        setting["class_order"] = [0, 1, 2]
        if lse:
            setting["value"] = 1
    return setting


def _request(
    dataset_id: str,
    *,
    task_type: str,
    family: str,
    acquisition_name: str | None = None,
    direction: str = "maximize",
    target_classes: list[int] | None = None,
) -> RegressionRunRequest:
    if acquisition_name is None:
        if family == "level_set_estimation":
            acquisition_name = "straddle"
        elif family == "active_learning":
            acquisition_name = "variance"
        else:
            acquisition_name = "EI"
    acqf_kwargs: dict[str, object] = {
        "web_family": family,
        "web_risk_type": "none",
        "web_risk_alpha": 0.2,
    }
    if family == "level_set_estimation":
        acqf_kwargs["web_level_set_parameter"] = 1.96

    return RegressionRunRequest(
        dataset_id=dataset_id,
        feature_columns=["x"],
        target_column="y",
        target_columns=["y"],
        model_type="base",
        model_kwargs={
            "web_target_settings": [
                _target_setting(
                    task_type=task_type,
                    family=family,
                    direction=direction,
                    target_classes=target_classes,
                )
            ]
        },
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
            "name": acquisition_name,
            "beta": 2.0,
            "acqf_kwargs": acqf_kwargs,
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
        _request(
            dataset_id,
            task_type=task_type,
            family="active_learning",
        ),
        store,
    )

    assert result["candidates"]
    assert len(result["candidates"]) == 2
    assert result["metadata"]["input_perturbation_risk_type"] == "none"


@pytest.mark.parametrize("task_type", ["classification", "ordinal"])
def test_web_three_class_lse_runs_with_input_perturbation(task_type: str) -> None:
    """Hybrid classification/ordinal Web LSE must preserve nominal q semantics."""

    torch.manual_seed(0)
    store, dataset_id = _store_with_three_class_data()
    result = run_regression_web_workflow(
        _request(
            dataset_id,
            task_type=task_type,
            family="level_set_estimation",
        ),
        store,
    )

    assert result["candidates"]
    assert len(result["candidates"]) == 2
    assert result["metadata"]["input_perturbation_risk_type"] == "none"


@pytest.mark.parametrize("acquisition_name", ["EI", "NEI", "PI", "UCB"])
def test_web_three_class_bo_uses_hybrid_objective_acquisition(
    acquisition_name: str,
) -> None:
    """Single-output multiclass BO must not unwrap Hybrid into raw class BO."""

    torch.manual_seed(0)
    store, dataset_id = _store_with_three_class_data()
    result = run_regression_web_workflow(
        _request(
            dataset_id,
            task_type="classification",
            family="bayesian_optimization",
            acquisition_name=acquisition_name,
        ),
        store,
    )

    assert result["candidates"]
    assert len(result["candidates"]) == 2
    assert result["metadata"]["input_perturbation_risk_type"] == "none"


def test_web_multiclass_ei_preserves_multiple_class_utility_and_direction() -> None:
    """Web multiclass EI must keep selected-class utility and minimize direction."""

    torch.manual_seed(0)
    store, dataset_id = _store_with_three_class_data()
    result = run_regression_web_workflow(
        _request(
            dataset_id,
            task_type="classification",
            family="bayesian_optimization",
            acquisition_name="EI",
            direction="minimize",
            target_classes=[0, 2],
        ),
        store,
    )

    assert result["candidates"]
    assert len(result["candidates"]) == 2
    assert result["metadata"]["input_perturbation_risk_type"] == "none"
