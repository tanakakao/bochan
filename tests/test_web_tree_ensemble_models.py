from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")
pytest.importorskip("botorch")
pytest.importorskip("fastapi")

from botorch.posteriors.ensemble import EnsemblePosterior  # noqa: E402
from bochan.api.model_registry import DEFAULT_MODEL_REGISTRY  # noqa: E402
from bochan.desktop.services import DatasetStore, build_dataset_record  # noqa: E402
from bochan.models.hybrid.task_aware_posterior import (  # noqa: E402
    HybridPosteriorComponent,
    TaskAwareHybridPosterior,
)
from bochan.serving.webapp.app import RegressionRunRequest  # noqa: E402
from bochan.serving.webapp.workflows import run_regression_web_workflow  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
TREE_ENSEMBLE_MODEL_TYPES = (
    "random_forest",
    "lightgbm_ensemble",
    "ngboost_ensemble",
)
WEB_OUTPUT_TASK_TYPES = (
    "regression",
    "binary",
    "multiclass",
    "ordinal",
)


def test_web_tree_ensemble_models_exist_for_normal_and_mixed_output_tasks() -> None:
    registry = DEFAULT_MODEL_REGISTRY.raw()

    for input_type in ("normal", "mixed"):
        for task_type in WEB_OUTPUT_TASK_TYPES:
            task_registry = registry[input_type][task_type]
            for model_type in TREE_ENSEMBLE_MODEL_TYPES:
                assert model_type in task_registry, (input_type, task_type, model_type)


def test_web_model_options_expose_tree_ensemble_family() -> None:
    source = (ROOT / "web" / "src" / "modelOptions.ts").read_text(encoding="utf-8")

    assert '"tree_ensemble"' in source
    assert 'label: "ツリー・アンサンブル"' in source
    assert '{ value: "random_forest", label: "Random Forest", family: "tree_ensemble" }' in source
    assert '{ value: "lightgbm_ensemble", label: "LightGBM", family: "tree_ensemble" }' in source
    assert '{ value: "ngboost_ensemble", label: "NGBoost", family: "tree_ensemble" }' in source


def test_web_tree_ensemble_search_excludes_gradient_and_falls_back_to_ga() -> None:
    source = (ROOT / "web" / "src" / "pages" / "OptimizePage.tsx").read_text(encoding="utf-8")

    assert 'const treeEnsembleModel = isTreeEnsembleModelType(modelType);' in source
    assert '(!treeEnsembleModel || option.family !== "gradient")' in source
    assert 'const fallback: SearchMethod = treeEnsembleModel ? "ga" : "normal";' in source
    assert 'option.value !== "nsgaii"' in source


def test_web_extra_installs_optional_tree_ensemble_dependencies() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)

    web = project["project"]["optional-dependencies"]["web"]
    assert "scikit-learn>=1.3" in web
    assert "lightgbm>=4.7,<5" in web
    assert "ngboost>=0.5.11,<0.6" in web


def test_task_aware_hybrid_posterior_preserves_finite_ensemble_function_draws() -> None:
    values = torch.tensor(
        [
            [[0.0], [1.0], [2.0]],
            [[10.0], [11.0], [12.0]],
        ],
        dtype=torch.double,
    )
    ensemble = EnsemblePosterior(values=values)
    component = HybridPosteriorComponent(
        mean=ensemble.mean.squeeze(-1),
        variance=ensemble.variance.squeeze(-1),
        posterior=ensemble,
        name="property",
    )
    posterior = TaskAwareHybridPosterior(
        mean=component.mean.unsqueeze(-1),
        variance=component.variance.unsqueeze(-1),
        components=[component],
    )

    assert posterior.base_sample_shape == torch.Size([1])
    samples = posterior.rsample_from_base_samples(
        sample_shape=torch.Size([2]),
        base_samples=torch.tensor([[-8.0], [8.0]], dtype=torch.double),
    )

    assert samples.shape == torch.Size([2, 3, 1])
    assert torch.equal(samples[0, :, 0], values[0, :, 0])
    assert torch.equal(samples[1, :, 0], values[1, :, 0])


def _random_forest_store() -> tuple[DatasetStore, str]:
    x = torch.linspace(0.0, 1.0, 12, dtype=torch.double).numpy()
    data = pd.DataFrame(
        {
            "x": x,
            "strength": 1.0 - (x - 0.25) ** 2,
            "ductility": 1.0 - (x - 0.75) ** 2,
        }
    )
    record = build_dataset_record(
        data=data,
        name="tree-ensemble.csv",
        source_type="csv",
    )
    store = DatasetStore()
    store.add(record)
    return store, record.dataset_id


def _random_forest_request(
    dataset_id: str,
    *,
    multi_objective: bool,
) -> RegressionRunRequest:
    targets = ["strength", "ductility"] if multi_objective else ["strength"]
    target_settings = [
        {
            "target": target,
            "task_type": "regression",
            "optimize": True,
            "direction": "maximize",
            "goal": "none",
            "value": None,
        }
        for target in targets
    ]
    return RegressionRunRequest(
        dataset_id=dataset_id,
        feature_columns=["x"],
        target_column=targets[0],
        target_columns=targets,
        directions={target: "maximize" for target in targets},
        model_type="random_forest",
        model_kwargs={
            "n_estimators": 16,
            "random_state": 0,
            "web_target_settings": target_settings,
        },
        fit_maxiter=4,
        normalize=True,
        outcome_transform=True,
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
            "name": "EHVI" if multi_objective else "EI",
            "beta": 2.0,
            "acqf_kwargs": {"web_family": "bayesian_optimization"},
        },
        optimizer={
            "name": "ga",
            "q": 1,
            "num_restarts": 1,
            "raw_samples": 16,
            "sequential": True,
        },
    )


def test_web_random_forest_single_objective_runs_with_evolutionary_search() -> None:
    torch.manual_seed(0)
    store, dataset_id = _random_forest_store()

    result = run_regression_web_workflow(
        _random_forest_request(dataset_id, multi_objective=False),
        store,
    )

    assert result["model_type"] == "random_forest"
    assert result["task_type"] == "regression"
    assert len(result["candidates"]) == 1
    assert result["metadata"]["optimizer"] == "evo"
    assert result["metadata"]["search_method"] == "ga"


def test_web_random_forest_multiobjective_runs_with_independent_surrogates() -> None:
    torch.manual_seed(0)
    store, dataset_id = _random_forest_store()

    result = run_regression_web_workflow(
        _random_forest_request(dataset_id, multi_objective=True),
        store,
    )

    assert result["model_type"] == "random_forest"
    assert result["task_type"] == "multi_objective"
    assert result["target_columns"] == ["strength", "ductility"]
    assert len(result["candidates"]) == 1
    assert result["metadata"]["optimizer"] == "evo"
    assert result["metadata"]["search_method"] == "ga"
