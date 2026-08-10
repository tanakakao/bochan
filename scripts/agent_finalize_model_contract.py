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


def fix_robust_ordinal_num_classes() -> None:
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
            raise RuntimeError("robust ordinal base import anchor not found")
        text = text.replace(import_anchor, import_replacement, 1)

    old = "            num_classes = int(train_Y.max().item()) + 1"
    count = text.count(old)
    if count != 4:
        raise RuntimeError(f"expected 4 robust ordinal num_classes inference sites, got {count}")
    text = text.replace(old, "            num_classes = _infer_num_classes_from_train_Y(train_Y)")
    write(path, text)


def fix_projected_mll_contract() -> None:
    path = MODELS / "components" / "projected.py"
    text = read(path)
    if "from gpytorch.models import ExactGP" not in text:
        text = text.replace(
            "from gpytorch.mlls import ExactMarginalLogLikelihood\n",
            "from gpytorch.mlls import ExactMarginalLogLikelihood\nfrom gpytorch.models import ExactGP\n",
            1,
        )

    start = text.index("    def make_mll(self, **kwargs: Any):\n")
    end = text.index("    @property\n    def train_input_raw", start)
    replacement = '''    def make_mll(self, **kwargs: Any):
        """内部 ``base_model`` の種類に応じた MLL を構築する。

        Exact Gaussian GP では ``beta`` は適用対象ではないため無視する。
        Variational / task-specific model は、その model が公開する ``make_mll``
        または ordinal MLL builder に固有引数を渡す。TypeError による互換
        fallback は行わず、model family ごとの contract を明示する。
        """
        if isinstance(self.base_model, ExactGP):
            mll_kwargs = dict(kwargs)
            mll_kwargs.pop("beta", None)
            if mll_kwargs:
                names = ", ".join(sorted(mll_kwargs))
                raise TypeError(
                    "Exact projected models received unsupported MLL keyword arguments: "
                    f"{names}."
                )
            return ExactMarginalLogLikelihood(
                self.base_model.likelihood,
                self.base_model,
            )

        base_make_mll = getattr(self.base_model, "make_mll", None)
        if callable(base_make_mll):
            return base_make_mll(**kwargs)

        if hasattr(self, "ordinal_likelihood"):
            from bochan.fit import make_ordinal_mll

            return make_ordinal_mll(self, **kwargs)

        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(
                f"{type(self.base_model).__name__} received unsupported MLL keyword "
                f"arguments: {names}."
            )
        raise TypeError(
            f"{type(self.base_model).__name__} does not define an MLL construction contract."
        )

'''
    write(path, text[:start] + replacement + text[end:])


def dedupe_repeated_property_decorators() -> None:
    """Alias method removal で残った連続 ``@property`` を1つに正規化する。"""
    for path in MODELS.rglob("*.py"):
        lines = read(path).splitlines(keepends=True)
        out: list[str] = []
        for line in lines:
            if line.strip() == "@property" and out and out[-1] == line:
                continue
            out.append(line)
        new = "".join(out)
        if new != "".join(lines):
            write(path, new)


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
    additions: list[str] = []
    if "test_robust_train_data_mixin_has_no_compatibility_aliases" not in text:
        additions.append(r'''

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
''')
    if "test_no_mechanical_latent_posterior_identifier_corruption" not in text:
        additions.append(r'''

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
''')
    if "test_model_methods_do_not_repeat_property_decorator" not in text:
        additions.append(r'''

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
''')
    if additions:
        write(path, text.rstrip() + "\n" + "".join(additions).rstrip() + "\n")


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
        raise RuntimeError("robust ordinal still bypasses canonical num_classes inference")

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

    for path in MODELS.rglob("*.py"):
        tree = ast.parse(read(path), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                count = sum(
                    isinstance(decorator, ast.Name) and decorator.id == "property"
                    for decorator in node.decorator_list
                )
                if count > 1:
                    raise RuntimeError(
                        f"{path.relative_to(ROOT)}::{node.name} has repeated @property"
                    )


if __name__ == "__main__":
    formalize_robust_train_data_mixin()
    fix_heteroscedastic_ordinal_num_classes()
    fix_robust_ordinal_num_classes()
    fix_projected_mll_contract()
    dedupe_repeated_property_decorators()
    normalize_remaining_ordinal_rrp_names()
    clean_latent_posterior_messages()
    extend_contract_guard()
    validate()
