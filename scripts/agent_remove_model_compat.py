from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "src" / "bochan" / "models"
TEXT_SUFFIXES = {".py", ".md", ".rst", ".txt"}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _method_spans(text: str, *, names: set[str] | None = None, backward_only: bool = False):
    spans: list[tuple[int, int]] = []
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if names is not None and node.name not in names:
            continue
        if backward_only:
            doc = ast.get_docstring(node) or ""
            if "Backward-supported" not in doc and "backward-supported" not in doc:
                continue
        spans.append((node.lineno - 1, node.end_lineno))
    return sorted(spans, reverse=True)


def _remove_methods(path: Path, *, names: set[str] | None = None, backward_only: bool = False) -> None:
    text = _read(path)
    lines = text.splitlines(keepends=True)
    changed = False
    for start, end in _method_spans(text, names=names, backward_only=backward_only):
        while end < len(lines) and not lines[end].strip():
            end += 1
        del lines[start:end]
        changed = True
    if changed:
        _write(path, "".join(lines))


def _replace_method(path: Path, class_name: str, method_name: str, source: str) -> None:
    text = _read(path)
    tree = ast.parse(text)
    target = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                    target = item
                    break
    if target is None:
        raise RuntimeError(f"missing {class_name}.{method_name} in {path}")
    lines = text.splitlines(keepends=True)
    lines[target.lineno - 1 : target.end_lineno] = [source.rstrip() + "\n"]
    _write(path, "".join(lines))


def _remove_binary_latent_aliases() -> None:
    for path in MODELS.rglob("*.py"):
        _remove_methods(path, names={"posterior_latent", "posterior_f"})

    roots = [ROOT / "src", ROOT / "tests", ROOT / "docs"]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
                continue
            if path == ROOT / "tests" / "test_model_contract_refactor.py":
                continue
            text = _read(path)
            new = text.replace("posterior_latent", "latent_posterior")
            new = new.replace("posterior_f", "latent_posterior")
            new = new.replace(
                '("latent_posterior", "latent_posterior", "latent_posterior", "posterior")',
                '("latent_posterior", "posterior")',
            )
            new = new.replace(
                '("latent_posterior", "latent_posterior", "posterior")',
                '("latent_posterior", "posterior")',
            )
            if new != text:
                _write(path, new)


def _remove_explicit_model_alias_properties() -> None:
    for path in MODELS.rglob("*.py"):
        _remove_methods(path, backward_only=True)

    explicit: dict[str, set[str]] = {
        "src/bochan/models/components/projected.py": {"raw_train_X", "train_X", "train_Y"},
        "src/bochan/models/regression/gaussian/high_dim/vae.py": {"raw_train_X", "train_X", "train_Y"},
        "src/bochan/models/ordinal/base/models_core.py": {"train_X", "train_Y", "inducing_points_original"},
        "src/bochan/models/ordinal/base/multioutput.py": {"train_X", "raw_train_X", "train_Y"},
    }
    for rel, names in explicit.items():
        path = ROOT / rel
        if path.exists():
            _remove_methods(path, names=names)

    multi = ROOT / "src/bochan/models/ordinal/base/multioutput.py"
    _replace_method(
        multi,
        "MultiOutputOrdinalModel",
        "_get_submodel_train_input_raw",
        '''    @staticmethod
    def _get_submodel_train_input_raw(model: Model) -> Tensor:
        if hasattr(model, "train_inputs_raw"):
            train_inputs_raw = model.train_inputs_raw
            if not isinstance(train_inputs_raw, tuple):
                raise TypeError(
                    "model.train_inputs_raw must be a tuple. "
                    f"Got {type(train_inputs_raw).__name__}."
                )
            return train_inputs_raw[0]
        if hasattr(model, "train_input_raw"):
            return model.train_input_raw
        if hasattr(model, "train_inputs"):
            train_inputs = model.train_inputs
            if not isinstance(train_inputs, tuple):
                raise TypeError(
                    "model.train_inputs must be a tuple. "
                    f"Got {type(train_inputs).__name__}."
                )
            return train_inputs[0]
        raise AttributeError(
            "Submodel must expose train_inputs_raw, train_input_raw, or train_inputs."
        )
''',
    )
    _replace_method(
        multi,
        "MultiOutputOrdinalModel",
        "_get_submodel_train_targets",
        '''    @staticmethod
    def _get_submodel_train_targets(model: Model) -> Tensor:
        if not hasattr(model, "train_targets"):
            raise AttributeError("Submodel must expose train_targets.")
        return model.train_targets
''',
    )


def _migrate_ordinal_saas_call_sites() -> None:
    roots = [ROOT / "src", ROOT / "tests", ROOT / "docs"]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
                continue
            text = _read(path)
            if "SaasOrdinal" not in text:
                continue
            new = text.replace("num_inducing_points=", "num_inducing=")
            new = new.replace("ordinal_likelihood=", "likelihood=")
            new = new.replace("``ordinal_likelihood``", "``likelihood``")
            if new != text:
                _write(path, new)


def _validate() -> None:
    offenders: list[str] = []
    for path in MODELS.rglob("*.py"):
        text = _read(path)
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
                "posterior_latent",
                "posterior_f",
            }:
                offenders.append(f"{path.relative_to(ROOT)}: legacy method {node.name}")
        if "Backward-supported alias" in text or "Backward-supported raw" in text:
            offenders.append(f"{path.relative_to(ROOT)}: backward alias documentation remains")
        if "saas_fixed" in text:
            offenders.append(f"{path.relative_to(ROOT)}: saas_fixed reference remains")
        if "_pop_inducing_points_num_alias" in text:
            offenders.append(f"{path.relative_to(ROOT)}: inducing alias resolver remains")

    fixed = ROOT / "src/bochan/models/ordinal/high_dim/saas_fixed.py"
    if fixed.exists():
        offenders.append("src/bochan/models/ordinal/high_dim/saas_fixed.py still exists")

    if offenders:
        raise RuntimeError("compatibility definitions remain:\n" + "\n".join(offenders))


def main() -> None:
    _remove_binary_latent_aliases()
    _remove_explicit_model_alias_properties()
    _migrate_ordinal_saas_call_sites()
    _validate()


if __name__ == "__main__":
    main()
