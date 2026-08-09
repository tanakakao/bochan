from __future__ import annotations

from time import perf_counter
from types import SimpleNamespace

import pytest

pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")
pytest.importorskip("botorch")
pytest.importorskip("fastapi")


def test_web_ngboost_runtime_defaults_follow_fit_maxiter() -> None:
    from bochan.serving.webapp.model_runtime import apply_web_model_runtime_defaults

    resolved = apply_web_model_runtime_defaults(
        {},
        model_type="ngboost_ensemble",
        fit_maxiter=128,
    )

    assert resolved == {
        "ensemble_size": 3,
        "n_estimators": 128,
        "verbose": False,
    }


def test_web_ngboost_runtime_defaults_preserve_explicit_parameters() -> None:
    from bochan.serving.webapp.model_runtime import apply_web_model_runtime_defaults

    resolved = apply_web_model_runtime_defaults(
        {
            "ensemble_size": 2,
            "n_estimators": 17,
            "verbose": True,
            "learning_rate": 0.03,
        },
        model_type="ngboost_ensemble",
        fit_maxiter=128,
    )

    assert resolved["ensemble_size"] == 2
    assert resolved["n_estimators"] == 17
    assert resolved["verbose"] is True
    assert resolved["learning_rate"] == pytest.approx(0.03)


def test_web_target_settings_apply_ngboost_runtime_defaults() -> None:
    from bochan.serving.webapp.app import RegressionRunRequest
    from bochan.serving.webapp.target_settings import _resolve_target_settings

    request = RegressionRunRequest(
        dataset_id="unused",
        feature_columns=["x"],
        target_column="y",
        target_columns=["y"],
        model_type="ngboost_ensemble",
        fit_maxiter=23,
    )

    _, model_kwargs = _resolve_target_settings(
        request,
        target_columns=["y"],
        directions={"y": "maximize"},
    )

    assert model_kwargs["ensemble_size"] == 3
    assert model_kwargs["n_estimators"] == 23
    assert model_kwargs["verbose"] is False


def _risk_request(*, model_type: str, input_perturbation: bool) -> SimpleNamespace:
    return SimpleNamespace(
        model_type=model_type,
        input_perturbation=input_perturbation,
        n_w=16,
        acquisition={
            "acqf_kwargs": {
                "web_family": "bayesian_optimization",
                "web_risk_type": "none",
                "web_risk_alpha": 0.2,
            }
        },
    )


def test_web_evolutionary_search_uses_interactive_budget() -> None:
    from bochan.serving.webapp.search_settings import resolve_search_method

    optimizer, kwargs, nsgaii = resolve_search_method(
        "ga",
        multi_objective=False,
    )

    assert optimizer == "evo"
    assert nsgaii is False
    assert kwargs == {
        "method": "ga",
        "options": {
            "pop_size": 32,
            "num_generations": 40,
        },
    }

    _, generic_evo_kwargs, _ = resolve_search_method(
        "evo",
        multi_objective=False,
    )
    assert generic_evo_kwargs["options"] == {
        "pop_size": 32,
        "num_generations": 40,
    }


def test_web_ngboost_uses_smaller_ga_budget() -> None:
    from bochan.serving.webapp.risk_settings import web_risk_run
    from bochan.serving.webapp.search_settings import resolve_search_method

    with web_risk_run(
        _risk_request(model_type="ngboost_ensemble", input_perturbation=False)
    ):
        _, kwargs, _ = resolve_search_method("ga", multi_objective=False)

    assert kwargs["options"] == {
        "pop_size": 24,
        "num_generations": 24,
    }


def test_web_perturbed_ngboost_uses_tight_ga_budget() -> None:
    from bochan.serving.webapp.risk_settings import web_risk_run
    from bochan.serving.webapp.search_settings import resolve_search_method

    with web_risk_run(
        _risk_request(model_type="ngboost_ensemble", input_perturbation=True)
    ):
        _, kwargs, _ = resolve_search_method("ga", multi_objective=False)

    assert kwargs["options"] == {
        "pop_size": 16,
        "num_generations": 20,
    }


def test_web_perturbed_random_forest_keeps_general_ga_budget() -> None:
    from bochan.serving.webapp.risk_settings import web_risk_run
    from bochan.serving.webapp.search_settings import resolve_search_method

    with web_risk_run(
        _risk_request(model_type="random_forest", input_perturbation=True)
    ):
        _, kwargs, _ = resolve_search_method("ga", multi_objective=False)

    assert kwargs["options"] == {
        "pop_size": 32,
        "num_generations": 40,
    }


def _ngboost_store() -> tuple[object, str]:
    from bochan.desktop.services import DatasetStore, build_dataset_record

    x = torch.linspace(0.0, 1.0, 16, dtype=torch.double).numpy()
    data = pd.DataFrame(
        {
            "x": x,
            "property": 1.0 - (x - 0.65) ** 2,
        }
    )
    record = build_dataset_record(
        data=data,
        name="ngboost-web-runtime.csv",
        source_type="csv",
    )
    store = DatasetStore()
    store.add(record)
    return store, record.dataset_id


def _ngboost_request(
    dataset_id: str,
    *,
    input_perturbation: bool = False,
    q: int = 1,
    sequential: bool = True,
    acquisition_family: str = "bayesian_optimization",
):
    from bochan.serving.webapp.app import RegressionRunRequest

    risk_type = "cvar" if input_perturbation else "none"
    return RegressionRunRequest(
        dataset_id=dataset_id,
        feature_columns=["x"],
        target_column="property",
        target_columns=["property"],
        directions={"property": "maximize"},
        model_type="ngboost_ensemble",
        model_kwargs={
            "random_state": 0,
            "web_target_settings": [
                {
                    "target": "property",
                    "task_type": "regression",
                    "optimize": True,
                    "direction": "maximize",
                    "goal": "none",
                    "value": None,
                }
            ],
        },
        # The regression tests intentionally use a tiny round count. The unit
        # tests above fix the production default mapping (128 -> 128 rounds).
        fit_maxiter=6,
        normalize=True,
        outcome_transform=True,
        input_perturbation=input_perturbation,
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
                "web_family": acquisition_family,
                "web_risk_type": risk_type,
                "web_risk_alpha": 0.2,
            },
        },
        optimizer={
            "name": "ga",
            "q": q,
            "num_restarts": 1,
            "raw_samples": 16,
            "sequential": sequential,
        },
        cross_validation=False,
        feature_importance={"enabled": False},
    )


def test_web_ngboost_q_batch_uses_joint_execution_copy() -> None:
    from bochan.serving.webapp.candidate_runtime import (
        apply_web_candidate_runtime_defaults,
        uses_ngboost_joint_batch,
    )

    request = _ngboost_request("unused", q=3, sequential=True)
    resolved = apply_web_candidate_runtime_defaults(request)

    assert uses_ngboost_joint_batch(request) is True
    assert request.optimizer.sequential is True
    assert resolved is not request
    assert resolved.optimizer is not request.optimizer
    assert resolved.optimizer.q == 3
    assert resolved.optimizer.sequential is False


def test_web_ngboost_joint_execution_is_narrowly_scoped() -> None:
    from bochan.serving.webapp.candidate_runtime import (
        apply_web_candidate_runtime_defaults,
    )

    q1 = _ngboost_request("unused", q=1, sequential=True)
    already_joint = _ngboost_request("unused", q=3, sequential=False)
    lse = _ngboost_request(
        "unused",
        q=3,
        sequential=True,
        acquisition_family="level_set_estimation",
    )
    random_forest = _ngboost_request("unused", q=3, sequential=True).model_copy(
        update={"model_type": "random_forest"}
    )

    assert apply_web_candidate_runtime_defaults(q1) is q1
    assert apply_web_candidate_runtime_defaults(already_joint) is already_joint
    assert apply_web_candidate_runtime_defaults(lse) is lse
    assert apply_web_candidate_runtime_defaults(random_forest) is random_forest


def _run_ngboost_request(request):
    from bochan.serving.webapp.workflows import run_regression_web_workflow

    torch.manual_seed(0)
    store, dataset_id = _ngboost_store()
    request = request(dataset_id)
    started = perf_counter()
    result = run_regression_web_workflow(request, store)
    elapsed = perf_counter() - started
    return result, elapsed


def test_real_web_ngboost_fit_and_suggestion_complete() -> None:
    pytest.importorskip("ngboost")

    result, elapsed = _run_ngboost_request(
        lambda dataset_id: _ngboost_request(dataset_id, input_perturbation=False)
    )

    assert result["model_type"] == "ngboost_ensemble"
    assert result["task_type"] == "regression"
    assert len(result["candidates"]) == 1
    assert result["metadata"]["optimizer"] == "evo"
    assert result["metadata"]["search_method"] == "ga"
    assert result["metadata"]["timings_ms"]["fit"] >= 0.0
    assert result["metadata"]["timings_ms"]["candidate"] >= 0.0
    # This is only a runaway guard, not a performance benchmark. The old Web
    # defaults could take many minutes; the focused 6-round smoke should finish
    # comfortably within this generous CI limit.
    assert elapsed < 120.0


def test_real_web_ngboost_input_perturbation_suggestion_complete() -> None:
    pytest.importorskip("ngboost")

    result, elapsed = _run_ngboost_request(
        lambda dataset_id: _ngboost_request(dataset_id, input_perturbation=True)
    )

    assert len(result["candidates"]) == 1
    assert result["metadata"]["input_perturbation_risk_type"] == "cvar"
    assert result["metadata"]["input_perturbation_risk_enabled"] is True
    assert result["metadata"]["timings_ms"]["candidate"] >= 0.0
    assert elapsed < 120.0


def test_real_web_ngboost_joint_q_batch_with_input_perturbation() -> None:
    pytest.importorskip("ngboost")

    result, elapsed = _run_ngboost_request(
        lambda dataset_id: _ngboost_request(
            dataset_id,
            input_perturbation=True,
            q=3,
            sequential=True,
        )
    )

    uniqueness = result["metadata"]["candidate_uniqueness"]
    assert len(result["candidates"]) == 3
    assert uniqueness["requested_q"] == 3
    assert uniqueness["sequential"] is False
    assert uniqueness["unique_count"] == 3
    assert result["batch_acq_value"] is not None
    assert result["metadata"]["input_perturbation_risk_type"] == "cvar"
    assert result["metadata"]["input_perturbation_risk_enabled"] is True
    assert result["metadata"]["timings_ms"]["candidate"] >= 0.0
    assert elapsed < 120.0
