from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "bochan"
MODELS_ROOT = SRC_ROOT / "models"


def _tracked_files() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip("tracked-file check requires a Git checkout")
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def test_distributional_regression_families_use_canonical_layout() -> None:
    assert not (MODELS_ROOT / "regression" / "non_gaussian").exists()
    assert (MODELS_ROOT / "regression" / "beta" / "base" / "models.py").is_file()
    assert (MODELS_ROOT / "regression" / "gamma" / "base" / "models.py").is_file()
    assert (
        MODELS_ROOT
        / "regression"
        / "count"
        / "poisson"
        / "base"
        / "models.py"
    ).is_file()
    assert (
        MODELS_ROOT
        / "regression"
        / "count"
        / "negative_binomial"
        / "base"
        / "models.py"
    ).is_file()


def test_model_likelihoods_are_family_owned() -> None:
    assert not (SRC_ROOT / "likelihoods").exists()
    assert (MODELS_ROOT / "ordinal" / "likelihood.py").is_file()
    assert (MODELS_ROOT / "regression" / "gaussian" / "likelihood.py").is_file()

    tree = ast.parse(
        (MODELS_ROOT / "regression" / "gaussian" / "likelihood.py").read_text(
            encoding="utf-8"
        )
    )
    function_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert "build_single_task_likelihood" in function_names
    assert "build_multitask_likelihood" in function_names
    assert "singletasklikelihood" not in function_names
    assert "multitasklikelihood" not in function_names


def test_family_specific_components_are_not_global_components() -> None:
    global_components = MODELS_ROOT / "components"
    for filename in (
        "beta.py",
        "gamma.py",
        "poisson.py",
        "negative_binomial.py",
        "multiclass.py",
    ):
        assert not (global_components / filename).exists()

    assert (MODELS_ROOT / "regression" / "beta" / "_components.py").is_file()
    assert (MODELS_ROOT / "regression" / "gamma" / "_components.py").is_file()
    assert (
        MODELS_ROOT / "regression" / "count" / "poisson" / "_components.py"
    ).is_file()
    assert (
        MODELS_ROOT
        / "regression"
        / "count"
        / "negative_binomial"
        / "_components.py"
    ).is_file()
    assert (MODELS_ROOT / "classification" / "multiclass" / "_components.py").is_file()


def test_acquisition_probability_helper_is_not_named_likelihood() -> None:
    binary_root = SRC_ROOT / "acquisition" / "binary"
    assert not (binary_root / "_likelihood.py").exists()
    assert (binary_root / "_probability.py").is_file()


def test_removed_model_paths_are_not_referenced() -> None:
    forbidden = (
        "bochan.models.regression.non_gaussian",
        "bochan.likelihoods",
        "bochan.models.components.beta",
        "bochan.models.components.gamma",
        "bochan.models.components.poisson",
        "bochan.models.components.negative_binomial",
        "bochan.models.components.multiclass",
        "bochan.acquisition.binary._likelihood",
    )
    offenders: list[str] = []
    for root in (SRC_ROOT, REPO_ROOT / "tests"):
        for path in root.rglob("*.py"):
            if path == Path(__file__):
                continue
            source = path.read_text(encoding="utf-8")
            if any(token in source for token in forbidden):
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders



def test_non_gaussian_posterior_sampling_is_not_monkey_patched() -> None:
    sampling_path = MODELS_ROOT / "components" / "sampling.py"
    source = sampling_path.read_text(encoding="utf-8")
    for class_name in (
        "BetaPosterior",
        "GammaPosterior",
        "PoissonPosterior",
        "NegativeBinomialPosterior",
    ):
        assert f"{class_name}.rsample_from_base_samples =" not in source

    posterior_files = (
        MODELS_ROOT / "regression" / "beta" / "_components.py",
        MODELS_ROOT / "regression" / "gamma" / "_components.py",
        MODELS_ROOT / "regression" / "count" / "poisson" / "_components.py",
        MODELS_ROOT
        / "regression"
        / "count"
        / "negative_binomial"
        / "_components.py",
    )
    for path in posterior_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        methods = {
            node.name
            for class_node in tree.body
            if isinstance(class_node, ast.ClassDef) and class_node.name.endswith("Posterior")
            for node in class_node.body
            if isinstance(node, ast.FunctionDef)
        }
        assert "rsample_from_base_samples" in methods

def test_generated_python_artifacts_are_not_tracked() -> None:
    tracked = _tracked_files()
    offenders = [
        path
        for path in tracked
        if path.startswith("src/bochan.egg-info/")
        or "/__pycache__/" in path
        or path.endswith(".pyc")
    ]
    assert not offenders
