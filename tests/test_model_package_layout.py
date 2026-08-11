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


def test_workflows_do_not_reference_removed_model_paths() -> None:
    forbidden = (
        "src/bochan/models/regression/non_gaussian/",
        "src/bochan/likelihoods/",
        "src/bochan/models/components/beta.py",
        "src/bochan/models/components/gamma.py",
        "src/bochan/models/components/poisson.py",
        "src/bochan/models/components/negative_binomial.py",
        "src/bochan/models/components/multiclass.py",
        "src/bochan/acquisition/binary/_likelihood.py",
    )
    offenders: list[str] = []
    workflows_root = REPO_ROOT / ".github" / "workflows"
    for path in workflows_root.glob("*.yml"):
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


def test_model_kernels_and_posteriors_are_family_owned() -> None:
    assert not (SRC_ROOT / "kernels").exists()
    assert not (SRC_ROOT / "posteriors").exists()

    assert (
        MODELS_ROOT / "classification" / "binary" / "base" / "kernel.py"
    ).is_file()
    assert (
        MODELS_ROOT / "classification" / "binary" / "base" / "posterior.py"
    ).is_file()
    assert (
        MODELS_ROOT / "classification" / "common" / "posterior.py"
    ).is_file()
    assert (MODELS_ROOT / "ordinal" / "base" / "kernel.py").is_file()
    assert (MODELS_ROOT / "ordinal" / "posterior.py").is_file()
    assert (
        MODELS_ROOT / "classification" / "multiclass" / "base" / "posterior.py"
    ).is_file()


def test_ordinal_base_has_one_canonical_model_and_kernel_implementation() -> None:
    ordinal_base = MODELS_ROOT / "ordinal" / "base"
    assert not (ordinal_base / "models_core.py").exists()

    models_source = (ordinal_base / "models.py").read_text(encoding="utf-8")
    kernel_source = (ordinal_base / "kernel.py").read_text(encoding="utf-8")

    assert "_OldOrdinalGPModel" not in models_source
    assert "def build_mixed_ordinal_kernel(" not in models_source
    assert kernel_source.count("def build_mixed_ordinal_kernel(") == 1
    assert models_source.count("class OrdinalGPModel(") == 1


def test_removed_kernel_and_posterior_paths_are_not_referenced() -> None:
    forbidden = (
        "bochan.kernels",
        "bochan.posteriors",
        "src/bochan/kernels/",
        "src/bochan/posteriors/",
        "classification.multiclass.base.posteriors",
        "classification/multiclass/base/posteriors.py",
        "models.ordinal.base.models_core",
        "models/ordinal/base/models_core.py",
    )
    offenders: list[str] = []
    roots = (
        SRC_ROOT,
        REPO_ROOT / "tests",
        REPO_ROOT / "docs",
        REPO_ROOT / ".github",
    )
    helper_names = {
        "model-component-layout-refactor.yml",
        "model-component-layout-refactor-pr.yml",
        "model_component_layout_refactor.py",
    }
    for root in roots:
        if not root.exists():
            continue
        for candidate in root.rglob("*"):
            if candidate.name in helper_names:
                continue
            if (
                not candidate.is_file()
                or candidate.suffix not in {".py", ".md", ".yml", ".yaml"}
            ):
                continue
            if candidate == Path(__file__):
                continue
            candidate_source = candidate.read_text(encoding="utf-8")
            if any(token in candidate_source for token in forbidden):
                offenders.append(str(candidate.relative_to(REPO_ROOT)))
    assert not offenders


def test_binary_kernel_has_task_specific_builder_name() -> None:
    kernel_path = (
        MODELS_ROOT / "classification" / "binary" / "base" / "kernel.py"
    )
    source = kernel_path.read_text(encoding="utf-8")
    assert "def build_binary_mixed_kernel(" in source
    assert "def categorical_kernel(" not in source
