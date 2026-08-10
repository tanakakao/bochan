from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "src" / "bochan" / "models"

PROJECTED_FILES = [
    MODELS / "regression" / "gaussian" / "high_dim" / "decomposition.py",
    MODELS / "classification" / "binary" / "high_dim" / "decomposition.py",
    MODELS / "classification" / "multiclass" / "high_dim" / "decomposition.py",
    MODELS / "ordinal" / "high_dim" / "decomposition.py",
    MODELS / "regression" / "non_gaussian" / "beta" / "high_dim" / "beta_decomposition.py",
    MODELS / "regression" / "non_gaussian" / "gamma" / "high_dim" / "gamma_decomposition.py",
    MODELS / "regression" / "non_gaussian" / "poisson" / "high_dim" / "poisson_decomposition.py",
    MODELS / "regression" / "non_gaussian" / "negative_binomial" / "high_dim" / "negative_binomial_decomposition.py",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _offsets(source: str) -> list[int]:
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _absolute(offsets: list[int], lineno: int, col_offset: int) -> int:
    return offsets[lineno - 1] + col_offset


def _is_projected_model_class(node: ast.ClassDef) -> bool:
    return (
        node.name.startswith(("PCA", "REMBO"))
        and not node.name.endswith(("Transformer", "Config"))
    )


def fix_orphan_projected_property() -> None:
    path = MODELS / "components" / "projected.py"
    text = read(path)
    bad = "    @property\n    def _to_preprojection_space(self, X: Tensor) -> Tensor:\n"
    good = "    def _to_preprojection_space(self, X: Tensor) -> Tensor:\n"
    if bad in text:
        text = text.replace(bad, good, 1)
    if good not in text:
        raise RuntimeError("_to_preprojection_space method not found")
    write(path, text)


def canonicalize_projected_utils() -> None:
    path = MODELS / "components" / "projected_utils.py"
    text = read(path)
    old = '''def _resolve_latent_dim(
    *,
    latent_dim: Optional[int],
    n_components: Optional[int],
    default: int,
) -> int:
    """latent_dim / n_components の後方互換を解決する。"""
    if latent_dim is not None and n_components is not None and latent_dim != n_components:
        raise ValueError(
            f"latent_dim and n_components are both specified but inconsistent: "
            f"latent_dim={latent_dim}, n_components={n_components}."
        )
    value = n_components if n_components is not None else latent_dim
    return int(default if value is None else value)
'''
    new = '''def _resolve_latent_dim(
    *,
    latent_dim: Optional[int],
    default: int,
) -> int:
    """Projected model の latent dimension を正の整数として解決する。"""
    value = int(default if latent_dim is None else latent_dim)
    if value <= 0:
        raise ValueError(f"latent_dim must be a positive integer, got {value}.")
    return value
'''
    if old in text:
        text = text.replace(old, new, 1)
    elif new not in text:
        raise RuntimeError("projected latent-dimension resolver block not found")
    write(path, text)


def canonicalize_model_constructor_dimension(path: Path) -> None:
    """PCA/REMBO model の public dimension argument を latent_dim に統一する。"""
    source = read(path)
    tree = ast.parse(source, filename=str(path))
    offsets = _offsets(source)
    lines = source.splitlines(keepends=True)
    edits: list[tuple[int, int, str]] = []

    for class_node in [node for node in tree.body if isinstance(node, ast.ClassDef)]:
        if not _is_projected_model_class(class_node):
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

        args = [*init.args.posonlyargs, *init.args.args, *init.args.kwonlyargs]
        arg_names = {arg.arg for arg in args}
        legacy_arg = next((arg for arg in args if arg.arg == "n_components"), None)
        if legacy_arg is None:
            continue

        if "latent_dim" in arg_names:
            # Each constructor argument is intentionally one-per-line in these wrappers.
            line_start = offsets[legacy_arg.lineno - 1]
            line_end = offsets[legacy_arg.lineno]
            line = source[line_start:line_end]
            if "n_components" not in line:
                raise RuntimeError(f"could not isolate n_components argument line: {path}:{class_node.name}")
            edits.append((line_start, line_end, ""))
        else:
            start = _absolute(offsets, legacy_arg.lineno, legacy_arg.col_offset)
            edits.append((start, start + len("n_components"), "latent_dim"))

        # Rename uses of the removed / renamed local variable, but never keyword names.
        for node in ast.walk(init):
            if isinstance(node, ast.Name) and node.id == "n_components":
                start = _absolute(offsets, node.lineno, node.col_offset)
                edits.append((start, start + len("n_components"), "latent_dim"))

    # Avoid applying edits that fall inside a whole argument line scheduled for deletion.
    deletions = [(start, end) for start, end, replacement in edits if replacement == ""]
    filtered: list[tuple[int, int, str]] = []
    for edit in edits:
        start, end, replacement = edit
        if replacement and any(d_start <= start and end <= d_end for d_start, d_end in deletions):
            continue
        filtered.append(edit)

    for start, end, replacement in sorted(set(filtered), reverse=True):
        source = source[:start] + replacement + source[end:]

    # The shared resolver no longer accepts the compatibility keyword.
    source = re.sub(
        r"_resolve_latent_dim\(\s*latent_dim=latent_dim,\s*n_components=(?:latent_dim|n_components),\s*default=(?P<default>\d+),?\s*\)",
        lambda match: f"_resolve_latent_dim(latent_dim=latent_dim, default={match.group('default')})",
        source,
        flags=re.DOTALL,
    )

    # Rebuild paths must use the same public constructor keyword.
    source = source.replace("n_components=self.latent_dim,", "latent_dim=self.latent_dim,")
    source = source.replace("n_components=self.projected_dim,", "latent_dim=self.projected_dim,")

    # Remove model-level compatibility state; transformer config keeps n_components.
    source = re.sub(r"^\s*self\.n_components\s*=\s*self\.projected_dim\s*\n", "", source, flags=re.MULTILINE)

    # Remove stale compatibility documentation without changing internal config terminology.
    source = re.sub(r"^\s*n_components:.*(?:後方互換|compatibility).*\n", "", source, flags=re.MULTILINE | re.IGNORECASE)
    source = source.replace("外部 API は ``n_components`` に統一する。", "外部 API は ``latent_dim`` に統一する。")
    source = source.replace(
        "旧 API の ``latent_dim`` は ``__init__`` 引数から削除する。",
        "PCAConfig / REMBOConfig 内部では ``n_components`` を使う。",
    )
    source = source.replace("n_components の指定", "latent_dim の指定")

    write(path, source)


def simplify_multiclass_resolver() -> None:
    path = MODELS / "classification" / "multiclass" / "high_dim" / "decomposition.py"
    text = read(path)
    old = '''def _resolve_projected_dim(
    *,
    n_components: Optional[int],
    latent_dim: int,
    input_dim: int,
    name: str,
) -> int:
    """
    PCA / REMBO の射影次元を決める。

    `n_components` が明示されていない場合は、デフォルト `latent_dim=8` が
    入力次元を超えても使えるように `input_dim` へ丸める。
    `n_components` が明示されている場合は、指定ミスを検出するために例外にする。
    """
    input_dim = int(input_dim)
    if input_dim <= 0:
        raise ValueError(f"{name}: input dimension must be positive.")

    if n_components is None:
        resolved = min(int(latent_dim), input_dim)
    else:
        resolved = int(n_components)

    if resolved <= 0:
        raise ValueError(f"{name}: n_components / latent_dim must be positive. Got {resolved}.")
    if resolved > input_dim:
        raise ValueError(
            f"{name}: n_components must be <= input dimension. "
            f"Got n_components={resolved}, input_dim={input_dim}."
        )
    return resolved
'''
    new = '''def _resolve_projected_dim(
    *,
    latent_dim: int,
    input_dim: int,
    name: str,
) -> int:
    """PCA / REMBO の射影次元を ``latent_dim`` から解決する。"""
    input_dim = int(input_dim)
    if input_dim <= 0:
        raise ValueError(f"{name}: input dimension must be positive.")

    resolved = min(int(latent_dim), input_dim)
    if resolved <= 0:
        raise ValueError(f"{name}: latent_dim must be positive. Got {resolved}.")
    return resolved
'''
    if old in text:
        text = text.replace(old, new, 1)
    elif new not in text:
        raise RuntimeError("multiclass projected-dimension resolver block not found")

    text = re.sub(
        r"(_resolve_projected_dim\(\s*)n_components=latent_dim,\s*",
        r"\1",
        text,
        flags=re.DOTALL,
    )
    write(path, text)


def projected_model_names() -> set[str]:
    names: set[str] = set()
    for path in PROJECTED_FILES:
        tree = ast.parse(read(path), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and _is_projected_model_class(node):
                names.add(node.name)
    return names


def migrate_python_call_sites() -> None:
    """bochan PCA/REMBO model callsだけ n_components= -> latent_dim= に移行する。"""
    names = projected_model_names()
    for root_name in ("src", "tests"):
        root = ROOT / root_name
        for path in root.rglob("*.py"):
            source = read(path)
            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError:
                continue
            offsets = _offsets(source)
            edits: list[tuple[int, int, str]] = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                else:
                    func_name = None
                if func_name not in names:
                    continue
                for keyword in node.keywords:
                    if keyword.arg != "n_components":
                        continue
                    start = _absolute(offsets, keyword.lineno, keyword.col_offset)
                    if source[start : start + len("n_components")] == "n_components":
                        edits.append((start, start + len("n_components"), "latent_dim"))
            for start, end, replacement in sorted(set(edits), reverse=True):
                source = source[:start] + replacement + source[end:]
            if edits:
                write(path, source)


def extend_contract_test() -> None:
    path = ROOT / "tests" / "test_model_contract_refactor.py"
    text = read(path)
    additions: list[str] = []

    if "test_projected_model_constructors_use_latent_dim_only" not in text:
        additions.append(r'''

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
''')

    if "test_projected_preprojection_transform_is_a_method" not in text:
        additions.append(r'''

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
''')

    if additions:
        write(path, text.rstrip() + "\n" + "".join(additions).rstrip() + "\n")


def validate() -> None:
    for path in PROJECTED_FILES:
        source = read(path)
        tree = ast.parse(source, filename=str(path))
        for class_node in [node for node in tree.body if isinstance(node, ast.ClassDef)]:
            if not _is_projected_model_class(class_node):
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
            args = {
                arg.arg
                for arg in [*init.args.posonlyargs, *init.args.args, *init.args.kwonlyargs]
            }
            forwards_constructor_args = init.args.vararg is not None or init.args.kwarg is not None
            if "n_components" in args or ("latent_dim" not in args and not forwards_constructor_args):
                raise RuntimeError(
                    f"non-canonical projected signature: {path}:{class_node.name}: {sorted(args)}"
                )

        if re.search(
            r"_resolve_latent_dim\([^)]*n_components\s*=",
            source,
            flags=re.DOTALL,
        ):
            raise RuntimeError(f"stale n_components resolver call remains: {path}")

    projected = ast.parse(read(MODELS / "components" / "projected.py"))
    base = next(
        node
        for node in projected.body
        if isinstance(node, ast.ClassDef) and node.name == "_BaseProjectedModel"
    )
    method = next(
        node
        for node in base.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_to_preprojection_space"
    )
    if any(isinstance(decorator, ast.Name) and decorator.id == "property" for decorator in method.decorator_list):
        raise RuntimeError("_to_preprojection_space remains a property")


if __name__ == "__main__":
    fix_orphan_projected_property()
    canonicalize_projected_utils()
    for projected_file in PROJECTED_FILES:
        canonicalize_model_constructor_dimension(projected_file)
    simplify_multiclass_resolver()
    migrate_python_call_sites()
    extend_contract_test()
    validate()
