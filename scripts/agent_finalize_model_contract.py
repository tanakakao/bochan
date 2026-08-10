from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "src" / "bochan" / "models"
TEXT_SUFFIXES = {".py", ".md", ".rst", ".txt"}


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def remove_class_methods(path: Path, class_name: str, method_names: set[str]) -> None:
    text = read(path)
    target: ast.ClassDef | None = None
    for node in ast.parse(text).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            target = node
            break
    if target is None:
        raise RuntimeError(f"class {class_name} not found in {path}")

    spans: list[tuple[int, int]] = []
    for node in target.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in method_names:
            start = node.lineno - 1
            for decorator in node.decorator_list:
                start = min(start, decorator.lineno - 1)
            spans.append((start, node.end_lineno))

    lines = text.splitlines(keepends=True)
    for start, end in sorted(spans, reverse=True):
        while end < len(lines) and not lines[end].strip():
            end += 1
        del lines[start:end]
    write(path, "".join(lines))


def formalize_robust_train_data_mixin() -> None:
    robust = MODELS / "components" / "robust.py"
    text = read(robust)
    text = text.replace('"TrainInputsAliasMixin",', '"TrainDataMixin",')
    text = text.replace("# train input alias mixin", "# train data synchronization mixin")
    text = text.replace("class TrainInputsAliasMixin:", "class TrainDataMixin:")
    text = text.replace(
        "train_inputs 系の共通 alias と set_train_data 実装。",
        "train_inputs / train_targets と latent model を同期する set_train_data 実装。",
    )
    write(robust, text)
    remove_class_methods(
        robust,
        "TrainDataMixin",
        {
            "train_input_raw",
            "train_input",
            "transformed_train_input",
            "transformed_train_inputs",
            "raw_train_X",
            "train_X_original",
            "train_X",
            "train_Y",
        },
    )

    for path in MODELS.rglob("*.py"):
        text = read(path)
        new = text.replace("TrainInputsAliasMixin", "TrainDataMixin")
        if new != text:
            write(path, new)


def fix_heteroscedastic_ordinal_num_classes() -> None:
    path = MODELS / "ordinal" / "robust" / "heteroscedastic.py"
    text = read(path)
    old = '"num_classes": int(num_classes),'
    count = text.count(old)
    if count != 2:
        raise RuntimeError(f"expected 2 heteroscedastic constructor num_classes entries, got {count}")
    write(path, text.replace(old, '"num_classes": int(self.num_classes),'))


def fix_rrp_ordinal_num_classes() -> None:
    path = MODELS / "ordinal" / "robust" / "relevance_pursuit.py"
    text = read(path)
    import_anchor = "    _OrdinalLatentGP,\n    _normalize_dims,\n"
    import_replacement = (
        "    _OrdinalLatentGP,\n"
        "    _infer_num_classes_from_train_Y,\n"
        "    _normalize_dims,\n"
    )
    if "_infer_num_classes_from_train_Y" not in text:
        if import_anchor not in text:
            raise RuntimeError("ordinal RRP base import anchor not found")
        text = text.replace(import_anchor, import_replacement, 1)

    old = "            num_classes = int(train_Y.max().item()) + 1"
    count = text.count(old)
    if count != 2:
        raise RuntimeError(f"expected 2 RRP num_classes inference sites, got {count}")
    text = text.replace(old, "            num_classes = _infer_num_classes_from_train_Y(train_Y)")
    write(path, text)


def normalize_remaining_ordinal_rrp_names() -> None:
    old = "OutlierRelevancePursuitOrdinal"
    new = "RobustRelevancePursuitOrdinal"
    for root_name in ("src", "tests", "docs"):
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
                continue
            text = read(path)
            replacement = text.replace(old, new)
            if replacement != text:
                write(path, replacement)


def clean_latent_posterior_messages() -> None:
    path = ROOT / "src/bochan/acquisition/binary/levelset_estimation/hetero_single_output.py"
    text = read(path)
    triple = (
        '            "  - model.latent_posterior(X)\\n"\n'
        '            "  - model.latent_posterior(X)\\n"\n'
        '            "  - model.latent_posterior(X)\\n"\n'
    )
    if triple in text:
        text = text.replace(
            triple,
            '            "  - model.latent_posterior(X)\\n"\n',
        )
    write(path, text)


def extend_contract_guard() -> None:
    path = ROOT / "tests/test_model_contract_refactor.py"
    text = read(path)
    if "test_robust_train_data_mixin_has_no_compatibility_aliases" in text:
        return
    addition = r'''


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
'''
    write(path, text.rstrip() + addition.rstrip() + "\n")


def validate() -> None:
    robust = MODELS / "components" / "robust.py"
    robust_text = read(robust)
    if "TrainInputsAliasMixin" in robust_text:
        raise RuntimeError("TrainInputsAliasMixin remains")

    hetero = read(MODELS / "ordinal" / "robust" / "heteroscedastic.py")
    if '"num_classes": int(num_classes),' in hetero:
        raise RuntimeError("heteroscedastic ordinal still stores unresolved num_classes")

    relevance = read(MODELS / "ordinal" / "robust" / "relevance_pursuit.py")
    if "OutlierRelevancePursuitOrdinal" in relevance:
        raise RuntimeError("legacy ordinal RRP name remains")
    if "num_classes = int(train_Y.max().item()) + 1" in relevance:
        raise RuntimeError("ordinal RRP still bypasses canonical num_classes inference")

    forbidden = (
        "latent_" "posterioror",
        "latent_" "posterioramily",
        "latent_" "posteriorn",
    )
    for root_name in ("src", "tests"):
        root = ROOT / root_name
        for path in root.rglob("*.py"):
            text = read(path)
            for token in forbidden:
                if token in text:
                    raise RuntimeError(f"{path.relative_to(ROOT)} contains {token}")


if __name__ == "__main__":
    formalize_robust_train_data_mixin()
    fix_heteroscedastic_ordinal_num_classes()
    fix_rrp_ordinal_num_classes()
    normalize_remaining_ordinal_rrp_names()
    clean_latent_posterior_messages()
    extend_contract_guard()
    validate()
