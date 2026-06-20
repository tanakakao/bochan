from __future__ import annotations

import ast
import copy
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGET = "_binary_values_to_probability_for_ipv"
FILES = [
    ROOT / "src/bochan/acquisition/binary/active_learning/single_output.py",
    ROOT / "src/bochan/acquisition/binary/active_learning/integrated_posterior_variance.py",
]


def _offsets(text: str) -> list[int]:
    starts = [0]
    starts.extend(match.end() for match in re.finditer("\n", text))
    return starts


def _span(node: ast.AST, starts: list[int]) -> tuple[int, int]:
    return (
        starts[node.lineno - 1] + node.col_offset,
        starts[node.end_lineno - 1] + node.end_col_offset,
    )


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    result: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            result[child] = parent
    return result


def _enclosing_function(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = parents.get(current)
    return None


def _model_expression(function: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.expr:
    args = {
        arg.arg
        for arg in [
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        ]
    }
    if "self" in args:
        return ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr="model", ctx=ast.Load())
    if "model" in args:
        return ast.Name(id="model", ctx=ast.Load())
    raise RuntimeError(f"Cannot resolve model in caller {function.name!r}.")


def _add_model_parameter() -> None:
    path = FILES[0]
    text = path.read_text(encoding="utf-8")
    old = f"def {TARGET}(\n    values: Tensor,"
    new = f"def {TARGET}(\n    model: Model,\n    values: Tensor,"
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"Could not find {TARGET} signature in {path}.")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _add_model_to_calls(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    parents = _parents(tree)
    starts = _offsets(text)
    replacements: list[tuple[int, int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != TARGET:
            continue
        if node.args and isinstance(node.args[0], (ast.Name, ast.Attribute)):
            first = ast.unparse(node.args[0])
            if first in {"model", "self.model"}:
                continue

        function = _enclosing_function(node, parents)
        if function is None:
            raise RuntimeError(f"Call to {TARGET} outside a function in {path}.")

        rewritten = copy.deepcopy(node)
        rewritten.args.insert(0, _model_expression(function))
        start, end = _span(node, starts)
        replacements.append((start, end, ast.unparse(rewritten)))

    for start, end, replacement in sorted(replacements, reverse=True):
        text = text[:start] + replacement + text[end:]
    path.write_text(text, encoding="utf-8")


def main() -> None:
    _add_model_parameter()
    for path in FILES:
        _add_model_to_calls(path)


if __name__ == "__main__":
    main()
