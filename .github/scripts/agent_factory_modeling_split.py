from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

ROOT = Path.cwd()
API = ROOT / "src/bochan/api"
FACTORY = API / "factory.py"

source = FACTORY.read_text(encoding="utf-8")
tree = ast.parse(source)
functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}

build_start = functions["infer_input_type"].lineno
fit_start = functions["_get_num_data"].lineno
callable_start = functions["_has_var_keyword"].lineno
acquisition_start = functions["_looks_like_ehvi"].lineno

lines = source.splitlines(keepends=True)
header = "".join(lines[: build_start - 1])
build_block = "".join(lines[build_start - 1 : fit_start - 1])
fit_block = "".join(lines[fit_start - 1 : callable_start - 1])
callable_block = "".join(lines[callable_start - 1 : acquisition_start - 1])
remaining_block = "".join(lines[acquisition_start - 1 :])

# _resolve_output_fit_configs is fit policy, even though it historically lived
# among model-construction helpers.
fit_config_node = functions["_resolve_output_fit_configs"]
fit_config_text = "".join(lines[fit_config_node.lineno - 1 : fit_config_node.end_lineno])
start_offset = sum(len(line) for line in lines[build_start - 1 : fit_config_node.lineno - 1])
end_offset = start_offset + len(fit_config_text)
build_block = build_block[:start_offset] + build_block[end_offset:]
fit_block = fit_config_text.rstrip() + "\n\n\n" + fit_block.lstrip()

modeling = API / "modeling"
modeling.mkdir(exist_ok=True)
(modeling / "__init__.py").write_text('"""Model construction and fitting for the public API."""\n', encoding="utf-8")
(modeling / "build.py").write_text(
    '''"""Model construction for the public API."""\n\nfrom __future__ import annotations\n\nfrom collections.abc import Callable, Mapping, Sequence\nfrom dataclasses import replace\nfrom typing import Any\n\nfrom ..configs import (\n    FitConfig,\n    InputType,\n    ModelBundle,\n    ModelConfig,\n    MultiOutputConfig,\n    OutputConfig,\n)\n\n\n'''
    + build_block.lstrip(),
    encoding="utf-8",
)
(modeling / "fit.py").write_text(
    '''"""Model fitting for the public API."""\n\nfrom __future__ import annotations\n\nfrom collections.abc import Callable, Sequence\nfrom typing import Any\n\nfrom ..configs import FitConfig, ModelBundle, MultiOutputConfig\nfrom ..support.callables import _filter_kwargs_for_callable\n\n\n'''
    + fit_block.lstrip(),
    encoding="utf-8",
)

support = API / "support"
support.mkdir(exist_ok=True)
(support / "__init__.py").write_text('"""Shared implementation support for the public API."""\n', encoding="utf-8")
(support / "callables.py").write_text(
    '''"""Callable signature helpers shared by API responsibility modules."""\n\nfrom __future__ import annotations\n\nimport inspect\nfrom collections.abc import Callable\nfrom typing import Any\n\n\n'''
    + callable_block.lstrip(),
    encoding="utf-8",
)

# Keep only acquisition/objective and low-level optimizer construction in the
# legacy-named file for this first split. Later PRs will move these blocks too.
factory_header = header
factory_header = factory_header.replace("import inspect\n", "")
factory_header += "\nfrom .support.callables import _filter_kwargs_for_callable\n"
FACTORY.write_text(factory_header.rstrip() + "\n\n\n" + remaining_block.lstrip(), encoding="utf-8")

build_names = {
    node.name
    for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    and build_start <= node.lineno < fit_start
}
build_names.discard("_resolve_output_fit_configs")
fit_names = {
    "_resolve_output_fit_configs",
    *(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and fit_start <= node.lineno < callable_start
    ),
}
callable_names = {"_has_var_keyword", "_filter_kwargs_for_callable"}


def render_alias(alias: ast.alias) -> str:
    return alias.name if alias.asname is None else f"{alias.name} as {alias.asname}"


def module_name(node: ast.ImportFrom, suffix: str) -> str:
    if node.level:
        return "." * node.level + suffix
    return f"bochan.api.{suffix}"


def rewrite_factory_imports(path: Path) -> bool:
    if path in {FACTORY, modeling / "build.py", modeling / "fit.py", support / "callables.py"}:
        return False
    text = path.read_text(encoding="utf-8")
    try:
        parsed = ast.parse(text)
    except SyntaxError:
        return False
    replacements: list[tuple[int, int, str]] = []
    text_lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in text_lines:
        offsets.append(offsets[-1] + len(line))

    for node in ast.walk(parsed):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        is_factory = (
            (node.module == "factory" and node.level > 0)
            or (node.module == "bochan.api.factory" and node.level == 0)
        )
        if not is_factory:
            continue
        groups: dict[str, list[ast.alias]] = defaultdict(list)
        for alias in node.names:
            if alias.name in build_names:
                groups["modeling.build"].append(alias)
            elif alias.name in fit_names:
                groups["modeling.fit"].append(alias)
            elif alias.name in callable_names:
                groups["support.callables"].append(alias)
            else:
                groups["factory"].append(alias)
        rendered = []
        for suffix in ("modeling.build", "modeling.fit", "support.callables", "factory"):
            aliases = groups.get(suffix)
            if aliases:
                rendered.append(
                    f"from {module_name(node, suffix)} import "
                    + ", ".join(render_alias(alias) for alias in aliases)
                )
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        replacements.append((start, end, "\n".join(rendered)))

    if not replacements:
        return False
    for start, end, replacement in sorted(replacements, reverse=True):
        text = text[:start] + replacement + text[end:]
    path.write_text(text, encoding="utf-8")
    return True


changed = []
for root in (ROOT / "src", ROOT / "tests"):
    for path in root.rglob("*.py"):
        if rewrite_factory_imports(path):
            changed.append(path)

# Public API keeps its stable top-level names but sources them from canonical owners.
# Direct imports of moved symbols from bochan.api.factory must be gone.
for root in (ROOT / "src", ROOT / "tests"):
    for path in root.rglob("*.py"):
        try:
            parsed = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(parsed):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            is_factory = (
                (node.module == "factory" and node.level > 0)
                or (node.module == "bochan.api.factory" and node.level == 0)
            )
            if not is_factory:
                continue
            stale = {alias.name for alias in node.names} & (build_names | fit_names | callable_names)
            if stale:
                raise RuntimeError(f"moved factory symbols remain in {path}: {sorted(stale)}")

remaining_defs = {
    node.name
    for node in ast.parse(FACTORY.read_text(encoding="utf-8")).body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
}
stale_defs = remaining_defs & (build_names | fit_names | callable_names)
if stale_defs:
    raise RuntimeError(f"factory still owns moved definitions: {sorted(stale_defs)}")

print("rewritten factory imports:")
for path in changed:
    print(path)
