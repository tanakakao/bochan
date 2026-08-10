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


def test_robust_train_data_mixin_has_no_compatibility_aliases() -> None:
    robust_path = MODELS_ROOT / "components" / "robust.py"
    tree = ast.parse(robust_path.read_text(encoding="utf-8"), filename=str(robust_path))
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrainDataMixin"
    )
    method_names = {
        node.name
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    forbidden = {
        "train_input_raw",
        "train_input",
        "transformed_train_input",
        "transformed_train_inputs",
        "raw_train_X",
        "train_X_original",
        "train_X",
        "train_Y",
    }
    assert not (method_names & forbidden)
    assert "TrainInputsAliasMixin" not in robust_path.read_text(encoding="utf-8")


def test_no_mechanical_latent_posterior_identifier_corruption() -> None:
    forbidden = (
        "latent_" "posterioror",
        "latent_" "posterioramily",
        "latent_" "posteriorn",
    )
    offenders: list[tuple[str, str]] = []
    for root_name in ("src", "tests"):
        root = REPO_ROOT / root_name
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in source:
                    offenders.append((str(path.relative_to(REPO_ROOT)), token))
    assert not offenders


def test_model_methods_do_not_repeat_property_decorator() -> None:
    offenders: list[tuple[str, str]] = []
    for path in MODELS_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            property_count = sum(
                isinstance(decorator, ast.Name) and decorator.id == "property"
                for decorator in node.decorator_list
            )
            if property_count > 1:
                offenders.append((str(path.relative_to(REPO_ROOT)), node.name))
    assert not offenders


def test_projected_model_constructors_use_latent_dim_only() -> None:
    offenders: list[tuple[str, str, str]] = []
    checked = 0
    for path in MODELS_ROOT.rglob("*decomposition.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for class_node in [node for node in tree.body if isinstance(node, ast.ClassDef)]:
            if not class_node.name.startswith(("PCA", "REMBO")):
                continue
            if class_node.name.endswith(("Transformer", "Config")):
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
            if init is None:
                continue
            checked += 1
            args = _function_arg_names(init)
            forwards_constructor_args = init.args.vararg is not None or init.args.kwarg is not None
            if "n_components" in args:
                offenders.append((str(path.relative_to(REPO_ROOT)), class_node.name, "n_components"))
            if "latent_dim" not in args and not forwards_constructor_args:
                offenders.append((str(path.relative_to(REPO_ROOT)), class_node.name, "missing latent_dim"))
    assert checked > 0
    assert not offenders


def test_projected_preprojection_transform_is_a_method() -> None:
    projected_path = MODELS_ROOT / "components" / "projected.py"
    tree = ast.parse(projected_path.read_text(encoding="utf-8"), filename=str(projected_path))
    base = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_BaseProjectedModel")
    method = next(
        node
        for node in base.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_to_preprojection_space"
    )
    assert not any(
        isinstance(decorator, ast.Name) and decorator.id == "property"
        for decorator in method.decorator_list
    )


def test_model_constructors_use_num_inducing() -> None:
    offenders: list[tuple[str, str]] = []
    checked = 0
    for path in MODELS_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for class_node in [node for node in tree.body if isinstance(node, ast.ClassDef)]:
            init = next(
                (
                    node
                    for node in class_node.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "__init__"
                ),
                None,
            )
            if init is None:
                continue
            args = _function_arg_names(init)
            if "num_inducing" in args:
                checked += 1
            if "num_inducing_points" in args:
                offenders.append((str(path.relative_to(REPO_ROOT)), class_node.name))
    assert checked > 0
    assert not offenders


def test_binary_conditioning_annotation_uses_canonical_name() -> None:
    path = MODELS_ROOT / "classification" / "binary" / "base" / "models.py"
    source = path.read_text(encoding="utf-8")
    assert '"GPClassificationModel"' not in source


def test_binary_base_package_exports_implementation_classes_directly() -> None:
    import bochan.models.classification.binary.base as binary_base
    from bochan.models.classification.binary.base import models as binary_models

    assert binary_base.BinaryClassificationGPModel is binary_models.BinaryClassificationGPModel
    assert (
        binary_base.BinaryClassificationMixedGPModel
        is binary_models.BinaryClassificationMixedGPModel
    )


def test_binary_base_has_no_stale_compatibility_class_names() -> None:
    path = MODELS_ROOT / "classification" / "binary" / "base" / "models.py"
    source = path.read_text(encoding="utf-8")
    assert '"GPClassificationModel"' not in source
    assert '"GPClassificationMixedModel"' not in source
