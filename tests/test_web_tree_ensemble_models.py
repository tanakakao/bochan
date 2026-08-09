from __future__ import annotations

from pathlib import Path
import tomllib

from bochan.api.model_registry import DEFAULT_MODEL_REGISTRY


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
