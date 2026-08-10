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


def fix_orphan_projected_property() -> None:
    path = MODELS / "components" / "projected.py"
    text = read(path)
    bad = "    @property\n    def _to_preprojection_space(self, X: Tensor) -> Tensor:\n"
    if bad not in text:
        raise RuntimeError("expected orphan @property before _to_preprojection_space")
    write(path, text.replace(bad, "    def _to_preprojection_space(self, X: Tensor) -> Tensor:\n", 1))


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
    if old not in text:
        raise RuntimeError("projected_utils compatibility resolver block not found")
    write(path, text.replace(old, new, 1))


def remove_dual_dimension_parameters() -> None:
    for path in PROJECTED_FILES:
        text = read(path)

        if path.parts[-3:] == ("ordinal", "high_dim", "decomposition.py"):
            text = text.replace(
                "n_components: Optional[int] = 2,",
                "latent_dim: Optional[int] = 2,",
            )
            text = text.replace("n_components=n_components,", "n_components=latent_dim,")
            text = text.replace("``n_components``", "``latent_dim``")
            text = text.replace("n_components の指定", "latent_dim の指定")
        else:
            text = re.sub(
                r"\n(?P<indent>\s*)n_components: (?:Optional\[int\]|int \| None) = None,",
                "",
                text,
            )
            text = text.replace(
                "n_components if n_components is not None else latent_dim",
                "latent_dim",
            )
            text = text.replace(
                "n_components=n_components,\n            latent_dim=latent_dim,",
                "latent_dim=latent_dim,",
            )
            text = text.replace(
                "latent_dim=latent_dim,\n            n_components=n_components,",
                "latent_dim=latent_dim,",
            )
            text = text.replace("            n_components=n_components,\n", "")

        # Rebuild calls must use the same canonical public argument.
        text = text.replace(
            "            n_components=self.latent_dim,\n",
            "            latent_dim=self.latent_dim,\n",
        )
        text = text.replace(
            "            n_components=self.projected_dim,\n",
            "            latent_dim=self.projected_dim,\n",
        )
        write(path, text)


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
    if old not in text:
        raise RuntimeError("multiclass projected resolver block not found")
    text = text.replace(old, new, 1)
    text = text.replace("            n_components=n_components,\n", "")
    write(path, text)


def migrate_python_call_sites() -> None:
    """PCA/REMBO model callsだけ n_components= -> latent_dim= に移行する。"""
    projected_names: set[str] = set()
    for path in PROJECTED_FILES:
        tree = ast.parse(read(path), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name.startswith(("PCA", "REMBO")):
                projected_names.add(node.name)

    for root_name in ("src", "tests"):
        root = ROOT / root_name
        for path in root.rglob("*.py"):
            source = read(path)
            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError:
                continue
            replacements: list[tuple[int, int, str]] = []
            lines = source.splitlines(keepends=True)
            offsets = [0]
            for line in lines:
                offsets.append(offsets[-1] + len(line))

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                if func_name not in projected_names:
                    continue
                for kw in node.keywords:
                    if kw.arg != "n_components" or not hasattr(kw, "lineno"):
                        continue
                    start = offsets[kw.lineno - 1] + kw.col_offset
                    segment = source[start : offsets[kw.end_lineno]]
                    match = re.match(r"n_components(?=\s*=)", segment)
                    if match:
                        replacements.append((start, start + len("n_components"), "latent_dim"))

            if replacements:
                for start, end, value in sorted(replacements, reverse=True):
                    source = source[:start] + value + source[end:]
                write(path, source)


def extend_contract_test() -> None:
    path = ROOT / "tests" / "test_model_contract_refactor.py"
    text = read(path)
    if "test_projected_model_constructors_use_latent_dim_only" in text:
        return
    addition = r'''


def test_projected_model_constructors_use_latent_dim_only() -> None:
    offenders: list[tuple[str, str, str]] = []
    checked = 0
    for path in MODELS_ROOT.rglob("*decomposition.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for class_node in [node for node in tree.body if isinstance(node, ast.ClassDef)]:
            if not class_node.name.startswith(("PCA", "REMBO")):
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
            if "n_components" in args:
                offenders.append((str(path.relative_to(REPO_ROOT)), class_node.name, "n_components"))
            if "latent_dim" not in args:
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
'''
    write(path, text.rstrip() + addition + "\n")


def validate() -> None:
    for path in PROJECTED_FILES:
        tree = ast.parse(read(path), filename=str(path))
        for class_node in [node for node in tree.body if isinstance(node, ast.ClassDef)]:
            if not class_node.name.startswith(("PCA", "REMBO")):
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
            if "n_components" in args or "latent_dim" not in args:
                raise RuntimeError(f"non-canonical projected signature: {path}:{class_node.name}: {sorted(args)}")

    projected = ast.parse(read(MODELS / "components" / "projected.py"))
    base = next(node for node in projected.body if isinstance(node, ast.ClassDef) and node.name == "_BaseProjectedModel")
    method = next(node for node in base.body if isinstance(node, ast.FunctionDef) and node.name == "_to_preprojection_space")
    if any(isinstance(d, ast.Name) and d.id == "property" for d in method.decorator_list):
        raise RuntimeError("_to_preprojection_space remains a property")


if __name__ == "__main__":
    fix_orphan_projected_property()
    canonicalize_projected_utils()
    remove_dual_dimension_parameters()
    simplify_multiclass_resolver()
    migrate_python_call_sites()
    extend_contract_test()
    validate()
