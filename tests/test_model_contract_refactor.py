from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = REPO_ROOT / "src" / "bochan" / "models"


LEGACY_PATCH_MODULES = (
    MODELS_ROOT / "projected_input_perturbation.py",
    MODELS_ROOT / "classification" / "binary" / "high_dim" / "input_perturbation.py",
    MODELS_ROOT
    / "classification"
    / "multiclass"
    / "robust"
    / "heteroscedastic_alignment.py",
    MODELS_ROOT / "ordinal" / "robust" / "_num_classes.py",
)

LEGACY_PATCH_INSTALLERS = {
    "configure_projected_model_classes",
    "configure_projected_binary_perturbation",
    "apply_hybrid_class_probability_shapes",
    "apply_task_aware_hybrid_posterior",
    "attach_prediction_methods",
    "enable_num_classes_inference",
    "_install_kronecker_input_transform_support",
    "apply_heteroscedastic_alignment",
    "apply_multiclass_posteriors",
}

LEGACY_PUBLIC_MODEL_NAMES = {
    "DeepGPModel",
    "DeepMixedGPModel",
    "DeepKernelGPModel",
    "DeepKernelMixedGPModel",
    "DeepKernelDeepGPModel",
    "DeepKernelDeepMixedGPModel",
    "SaasSingleTaskGP",
    "SaasMixedSingleTaskGP",
    "PCASingleTaskGP",
    "PCAMixedSingleTaskGP",
    "REMBOSingleTaskGP",
    "REMBOMixedSingleTaskGP",
    "VAESingleTaskGP",
    "VAEMixedSingleTaskGP",
    "PerturbationSupportedKroneckerMultiTaskGP",
    "MixedKroneckerMultiTaskGP",
    "KroneckerMultiTaskMixedGP",
    "BinaryClassificationDeepGPModel",
    "BinaryClassificationMixedDeepGPModel",
    "MulticlassDeepGPModel",
    "MulticlassMixedDeepGPModel",
    "OrdinalDeepGPModel",
    "OrdinalMixedDeepGPModel",
}

LEGACY_SHARED_ARGUMENTS = {
    "list_hidden_dims",
    "inducing_points_num",
}


def _iter_model_python_files():
    return MODELS_ROOT.rglob("*.py")


def test_runtime_patch_modules_are_removed() -> None:
    assert not [path for path in LEGACY_PATCH_MODULES if path.exists()]


def test_runtime_patch_installers_are_removed() -> None:
    offenders: list[tuple[str, str]] = []
    for path in _iter_model_python_files():
        text = path.read_text(encoding="utf-8")
        for name in LEGACY_PATCH_INSTALLERS:
            if name in text:
                offenders.append((str(path.relative_to(REPO_ROOT)), name))
    assert not offenders


def test_legacy_public_model_names_are_not_defined_or_exported() -> None:
    offenders: list[tuple[str, str]] = []
    for path in _iter_model_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in LEGACY_PUBLIC_MODEL_NAMES:
                    offenders.append((str(path.relative_to(REPO_ROOT)), node.name))
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                if node.id in LEGACY_PUBLIC_MODEL_NAMES:
                    offenders.append((str(path.relative_to(REPO_ROOT)), node.id))
    assert not offenders


def test_legacy_shared_constructor_arguments_are_removed() -> None:
    offenders: list[tuple[str, str, str]] = []
    for path in _iter_model_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            arg_names = {
                arg.arg
                for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            }
            for name in LEGACY_SHARED_ARGUMENTS & arg_names:
                offenders.append((str(path.relative_to(REPO_ROOT)), node.name, name))
    assert not offenders
