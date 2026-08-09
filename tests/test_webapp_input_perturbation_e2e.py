from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")
pytest.importorskip("botorch")
pytest.importorskip("fastapi")

from bochan.desktop.services import DatasetStore, build_dataset_record  # noqa: E402
from bochan.serving.webapp.app import RegressionRunRequest  # noqa: E402
from bochan.serving.webapp.workflows import run_regression_web_workflow  # noqa: E402


def _store_with_regression_data() -> tuple[DatasetStore, str]:
    x = torch.linspace(0.0, 1.0, 10, dtype=torch.double).numpy()
    data = pd.DataFrame({
        "x": x,
        "y": (x - 0.35) ** 2 + 0.15 * x,
    })
    record = build_dataset_record(data=data, name="perturbation.csv", source_type="csv")
    store = DatasetStore()
    store.add(record)
    return store, record.dataset_id


def _request(dataset_id: str, *, risk_type: str = "none", cross_validation: bool = False) -> RegressionRunRequest:
    return RegressionRunRequest(
        dataset_id=dataset_id,
        feature_columns=["x"],
        target_column="y",
        target_columns=["y"],
        model_type="base",
        model_kwargs={
            "web_target_settings": [
                {
                    "target": "y",
                    "task_type": "regression",
                    "optimize": True,
                    "direction": "maximize",
                    "goal": "none",
                    "value": None,
                }
            ]
        },
        fit_maxiter=8,
        normalize=True,
        outcome_transform=True,
        input_perturbation=True,
        n_w=16,
        perturbation_std=0.1,
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
            "name": "EI",
            "beta": 2.0,
            "acqf_kwargs": {
                "web_family": "bayesian_optimization",
                "web_risk_type": risk_type,
                "web_risk_alpha": 0.2,
            },
        },
        optimizer={
            "name": "optimize_acqf",
            "q": 3,
            "num_restarts": 2,
            "raw_samples": 32,
            "sequential": True,
        },
        cross_validation=cross_validation,
        cv_config={"splitter": "kfold", "n_splits": 2} if cross_validation else None,
    )


@pytest.mark.parametrize("risk_type", ["none", "var", "cvar"])
def test_web_regression_runs_end_to_end_with_input_perturbation(risk_type: str) -> None:
    """Browser-default BO must support every Web InputPerturbation risk mode."""

    torch.manual_seed(0)
    store, dataset_id = _store_with_regression_data()
    result = run_regression_web_workflow(_request(dataset_id, risk_type=risk_type), store)

    assert result["candidates"]
    assert len(result["candidates"]) == 3
    assert result["metadata"]["input_perturbation_risk_type"] == risk_type
    assert result["metadata"]["input_perturbation_risk_enabled"] is (risk_type != "none")


def test_web_cross_validation_runs_with_input_perturbation() -> None:
    """Cross-validation must score nominal rows, not expanded perturbation rows."""

    torch.manual_seed(0)
    store, dataset_id = _store_with_regression_data()
    result = run_regression_web_workflow(_request(dataset_id, cross_validation=True), store)

    assert result["candidates"]
    assert len(result["candidates"]) == 3
    assert result["metadata"]["input_perturbation_risk_type"] == "none"
