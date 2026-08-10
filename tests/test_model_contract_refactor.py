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
    MODELS_ROOT / "ordinal" / "high_dim" / "saas_fixed.py",
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

LEGACY_METHOD_NAMES = {
    "posterior_latent",
    "posterior_f",
}

LEGACY_PROPERTIES_BY_CLASS = {
    "_BaseProjectedModel": {"raw_train_X", "train_X", "train_Y"},
    "VAEGaussianGPModel": {"raw_train_X", "train_X", "train_Y"},
    "_BaseOrdinalGPModel": {"train_X", "train_Y", "inducing_points_original"},
    "MultiOutputOrdinalModel": {"train_X", "raw_train_X", "train_Y"},
}

SAAS_CANONICAL_CLASSES = {
    "SaasOrdinalGPModel",
    "SaasOrdinalMixedGPModel",
}
SAAS_LEGACY_ARGUMENTS = {"num_inducing_points", "ordinal_likelihood"}


def _iter_model_python_files():
    return MODELS_ROOT.rglob("*.py")


def _function_arg_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    return {
        arg.arg
        for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    }


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
            for name in LEGACY_SHARED_ARGUMENTS & _function_arg_names(node):
                offenders.append((str(path.relative_to(REPO_ROOT)), node.name, name))
    assert not offenders


def test_legacy_latent_posterior_method_aliases_are_removed() -> None:
    offenders: list[tuple[str, str]] = []
    for path in _iter_model_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in LEGACY_METHOD_NAMES:
                    offenders.append((str(path.relative_to(REPO_ROOT)), node.name))
    assert not offenders


def test_known_training_data_compatibility_properties_are_removed() -> None:
    offenders: list[tuple[str, str, str]] = []
    for path in _iter_model_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for class_node in [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]:
            forbidden = LEGACY_PROPERTIES_BY_CLASS.get(class_node.name, set())
            if not forbidden:
                continue
            for node in class_node.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in forbidden:
                    offenders.append(
                        (str(path.relative_to(REPO_ROOT)), class_node.name, node.name)
                    )
    assert not offenders


def test_ordinal_saas_uses_canonical_constructor_arguments() -> None:
    saas_path = MODELS_ROOT / "ordinal" / "high_dim" / "saas.py"
    tree = ast.parse(saas_path.read_text(encoding="utf-8"), filename=str(saas_path))
    offenders: list[tuple[str, str]] = []
    for class_node in [node for node in tree.body if isinstance(node, ast.ClassDef)]:
        if class_node.name not in SAAS_CANONICAL_CLASSES:
            continue
        init = next(
            (
                node
                for node in class_node.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "__init__"
            ),
            None,
        )
        assert init is not None
        for name in SAAS_LEGACY_ARGUMENTS & _function_arg_names(init):
            offenders.append((class_node.name, name))
    assert not offenders


def test_multiclass_high_dim_package_exports_canonical_models() -> None:
    from bochan.models.classification.multiclass import high_dim

    expected = {
        "SaasMulticlassClassificationGPModel",
        "SaasMulticlassClassificationMixedGPModel",
        "PCAMulticlassClassificationGPModel",
        "PCAMulticlassClassificationMixedGPModel",
        "REMBOMulticlassClassificationGPModel",
        "REMBOMulticlassClassificationMixedGPModel",
    }
    assert expected <= set(high_dim.__all__)
    for name in expected:
        assert getattr(high_dim, name).__name__ == name


def test_hybrid_package_exports_canonical_contract() -> None:
    import bochan.models.hybrid as hybrid

    expected = {
        "HybridMultiOutputModel",
        "HybridPosterior",
        "TaskAwareHybridPosterior",
        "OutputIndex",
        "OutputSpec",
        "PosteriorMode",
        "TaskType",
    }
    assert expected <= set(hybrid.__all__)
    for name in expected:
        assert hasattr(hybrid, name), name
