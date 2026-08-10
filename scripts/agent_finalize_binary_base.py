from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "src" / "bochan" / "models"

# One-shot migration: fold package-level binary defaults into the owning models.


def finalize_binary_defaults_and_names() -> None:
    path = MODELS / "classification" / "binary" / "base" / "models.py"
    text = path.read_text(encoding="utf-8")
    old = "        num_inducing: int = 20,"
    count = text.count(old)
    if count != 2:
        raise RuntimeError(f"expected exactly 2 binary base inducing defaults, got {count}")
    text = text.replace(old, "        num_inducing: int = 128,")
    text = text.replace(
        '-> "GPClassificationMixedModel":',
        '-> "BinaryClassificationMixedGPModel":',
    )
    path.write_text(text, encoding="utf-8")


def extend_contract_guard() -> None:
    path = ROOT / "tests" / "test_model_contract_refactor.py"
    text = path.read_text(encoding="utf-8")
    additions: list[str] = []
    if "test_binary_base_package_exports_implementation_classes_directly" not in text:
        additions.append(r'''

def test_binary_base_package_exports_implementation_classes_directly() -> None:
    import bochan.models.classification.binary.base as binary_base
    from bochan.models.classification.binary.base import models as binary_models

    assert binary_base.BinaryClassificationGPModel is binary_models.BinaryClassificationGPModel
    assert (
        binary_base.BinaryClassificationMixedGPModel
        is binary_models.BinaryClassificationMixedGPModel
    )
''')
    if "test_binary_base_has_no_stale_compatibility_class_names" not in text:
        additions.append(r'''

def test_binary_base_has_no_stale_compatibility_class_names() -> None:
    path = MODELS_ROOT / "classification" / "binary" / "base" / "models.py"
    source = path.read_text(encoding="utf-8")
    assert '"GPClassificationModel"' not in source
    assert '"GPClassificationMixedModel"' not in source
''')
    if additions:
        path.write_text(
            text.rstrip() + "\n" + "".join(additions).rstrip() + "\n",
            encoding="utf-8",
        )


def validate() -> None:
    model_path = MODELS / "classification" / "binary" / "base" / "models.py"
    source = model_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(model_path))
    defaults: dict[str, int] = {}
    for class_node in [node for node in tree.body if isinstance(node, ast.ClassDef)]:
        if class_node.name not in {
            "BinaryClassificationGPModel",
            "BinaryClassificationMixedGPModel",
        }:
            continue
        init = next(
            node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "__init__"
        )
        kwonly_defaults = {
            arg.arg: default
            for arg, default in zip(init.args.kwonlyargs, init.args.kw_defaults)
            if default is not None
        }
        positional_defaults = {}
        if init.args.defaults:
            positional_args = [*init.args.posonlyargs, *init.args.args]
            for arg, default in zip(
                positional_args[-len(init.args.defaults):],
                init.args.defaults,
            ):
                positional_defaults[arg.arg] = default
        default = kwonly_defaults.get("num_inducing", positional_defaults.get("num_inducing"))
        if not isinstance(default, ast.Constant) or default.value != 128:
            raise RuntimeError(
                f"{class_node.name}.num_inducing must default to 128, got {ast.dump(default) if default else None}"
            )
        defaults[class_node.name] = 128
    if len(defaults) != 2:
        raise RuntimeError(f"missing binary base models: {sorted(defaults)}")

    for legacy_name in ('"GPClassificationModel"', '"GPClassificationMixedModel"'):
        if legacy_name in source:
            raise RuntimeError(f"stale binary compatibility class name remains: {legacy_name}")

    init_path = MODELS / "classification" / "binary" / "base" / "__init__.py"
    init_tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    shadowing = {
        node.name
        for node in init_tree.body
        if isinstance(node, ast.ClassDef)
        and node.name in {
            "BinaryClassificationGPModel",
            "BinaryClassificationMixedGPModel",
        }
    }
    if shadowing:
        raise RuntimeError(f"binary package still defines wrapper classes: {sorted(shadowing)}")


if __name__ == "__main__":
    finalize_binary_defaults_and_names()
    extend_contract_guard()
    validate()
