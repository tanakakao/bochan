from __future__ import annotations

import importlib.util
from pathlib import Path

from bochan.api.model_registry import DEFAULT_MODEL_REGISTRY


OLD_PUBLIC_PACKAGES = (
    "bochan.models.regression.boosting",
    "bochan.models.classification.external",
    "bochan.models.classification.neural",
)

EXTERNAL_MODEL_TYPES = (
    "lightgbm",
    "lightgbm_ensemble",
    "ngboost",
    "ngboost_ensemble",
    "random_forest",
)


def test_obsolete_model_packages_are_removed() -> None:
    for module_name in OLD_PUBLIC_PACKAGES:
        assert importlib.util.find_spec(module_name) is None


def test_repository_has_no_obsolete_model_imports() -> None:
    paths = list(Path("src/bochan").rglob("*.py"))
    paths.extend(
        path
        for path in Path("tests").rglob("*.py")
        if path.name != "test_model_package_layout.py"
    )
    paths.extend(Path(".github/workflows").glob("*.yml"))

    offenders: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for module_name in OLD_PUBLIC_PACKAGES:
            if module_name in text:
                offenders.append(f"{path}: {module_name}")
    assert offenders == []


def test_external_regression_registry_uses_external_package() -> None:
    tree = DEFAULT_MODEL_REGISTRY.raw()
    for input_type in ("normal", "mixed"):
        regression = tree[input_type]["regression"]
        for model_type in EXTERNAL_MODEL_TYPES:
            module_name, _ = regression[model_type]
            assert module_name == "bochan.models.regression.external"


def test_classification_registry_is_split_by_task() -> None:
    tree = DEFAULT_MODEL_REGISTRY.raw()
    for input_type in ("normal", "mixed"):
        binary = tree[input_type]["binary"]
        multiclass = tree[input_type]["multiclass"]

        for model_type in EXTERNAL_MODEL_TYPES:
            binary_module, _ = binary[model_type]
            multiclass_module, _ = multiclass[model_type]
            assert binary_module == "bochan.models.classification.binary.external"
            assert multiclass_module == "bochan.models.classification.multiclass.external"

        binary_deep_module, _ = binary["deep_ensemble"]
        multiclass_deep_module, _ = multiclass["deep_ensemble"]
        assert binary_deep_module == "bochan.models.classification.binary.neural"
        assert multiclass_deep_module == "bochan.models.classification.multiclass.neural"


def test_ordinal_registry_contains_external_and_neural_models() -> None:
    tree = DEFAULT_MODEL_REGISTRY.raw()
    for input_type in ("normal", "mixed"):
        ordinal = tree[input_type]["ordinal"]
        for model_type in EXTERNAL_MODEL_TYPES:
            module_name, _ = ordinal[model_type]
            assert module_name == "bochan.models.ordinal.external"
        deep_module, _ = ordinal["deep_ensemble"]
        assert deep_module == "bochan.models.ordinal.neural"
