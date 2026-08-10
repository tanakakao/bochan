from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "src" / "bochan" / "models"

LEGACY = "num_inducing_points"
CANONICAL = "num_inducing"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def offsets(source: str) -> list[int]:
    values = [0]
    for line in source.splitlines(keepends=True):
        values.append(values[-1] + len(line))
    return values


def absolute(values: list[int], lineno: int, col_offset: int) -> int:
    return values[lineno - 1] + col_offset


def constructor(node: ast.ClassDef):
    return next(
        (
            child
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            and child.name == "__init__"
        ),
        None,
    )


def function_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.arg]:
    return [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]


def collect_renamed_classes() -> tuple[set[str], dict[str, set[str]]]:
    renamed: set[str] = set()
    bases: dict[str, set[str]] = {}
    for path in MODELS.rglob("*.py"):
        tree = ast.parse(read(path), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            base_names: set[str] = set()
            for base in node.bases:
                if isinstance(base, ast.Name):
                    base_names.add(base.id)
                elif isinstance(base, ast.Attribute):
                    base_names.add(base.attr)
            bases[node.name] = base_names
            init = constructor(node)
            if init is None:
                continue
            if LEGACY in {arg.arg for arg in function_args(init)}:
                renamed.add(node.name)
    return renamed, bases


def descendants_of(renamed: set[str], bases: dict[str, set[str]]) -> set[str]:
    descendants = set(renamed)
    changed = True
    while changed:
        changed = False
        for name, base_names in bases.items():
            if name in descendants:
                continue
            if base_names & descendants:
                descendants.add(name)
                changed = True
    return descendants


def canonicalize_model_constructors(renamed: set[str]) -> None:
    for path in MODELS.rglob("*.py"):
        source = read(path)
        tree = ast.parse(source, filename=str(path))
        line_offsets = offsets(source)
        edits: list[tuple[int, int, str]] = []

        for class_node in tree.body:
            if not isinstance(class_node, ast.ClassDef) or class_node.name not in renamed:
                continue
            init = constructor(class_node)
            if init is None:
                continue
            args = function_args(init)
            names = {arg.arg for arg in args}
            legacy_arg = next(arg for arg in args if arg.arg == LEGACY)

            if CANONICAL in names:
                line_start = line_offsets[legacy_arg.lineno - 1]
                line_end = line_offsets[legacy_arg.lineno]
                line = source[line_start:line_end]
                stripped = line.strip()
                if not stripped.startswith(f"{LEGACY}:") and not stripped.startswith(f"{LEGACY} ="):
                    raise RuntimeError(
                        f"cannot safely remove compatibility argument line: {path}:{class_node.name}: {stripped}"
                    )
                edits.append((line_start, line_end, ""))
            else:
                start = absolute(line_offsets, legacy_arg.lineno, legacy_arg.col_offset)
                edits.append((start, start + len(LEGACY), CANONICAL))

            # Rename only variable uses of the constructor argument. Keyword names
            # on helper / GPyTorch calls are deliberately not AST Name nodes.
            for node in ast.walk(init):
                if isinstance(node, ast.Name) and node.id == LEGACY:
                    start = absolute(line_offsets, node.lineno, node.col_offset)
                    edits.append((start, start + len(LEGACY), CANONICAL))

            # Model state follows the same canonical name throughout the class.
            for node in ast.walk(class_node):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == LEGACY
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"
                ):
                    start = absolute(line_offsets, node.end_lineno, node.end_col_offset) - len(LEGACY)
                    edits.append((start, start + len(LEGACY), CANONICAL))

        deletions = [(start, end) for start, end, replacement in edits if replacement == ""]
        filtered: list[tuple[int, int, str]] = []
        for start, end, replacement in edits:
            if replacement and any(d_start <= start and end <= d_end for d_start, d_end in deletions):
                continue
            filtered.append((start, end, replacement))

        if filtered:
            for start, end, replacement in sorted(set(filtered), reverse=True):
                source = source[:start] + replacement + source[end:]
            write(path, source)


def call_target_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def is_super_init(call: ast.Call) -> bool:
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "__init__"
        and isinstance(func.value, ast.Call)
        and isinstance(func.value.func, ast.Name)
        and func.value.func.id == "super"
    )


def is_self_class_call(call: ast.Call) -> bool:
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "__class__"
        and isinstance(func.value, ast.Name)
        and func.value.id == "self"
    )


def migrate_python_call_sites(renamed: set[str], descendants: set[str]) -> None:
    """Renamed bochan constructorsだけ keyword を canonical 名へ移す。"""
    for root_name in ("src", "tests"):
        root = ROOT / root_name
        for path in root.rglob("*.py"):
            source = read(path)
            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError:
                continue
            line_offsets = offsets(source)
            edits: list[tuple[int, int, str]] = []

            for class_node in [node for node in tree.body if isinstance(node, ast.ClassDef)]:
                class_allows_super = class_node.name in descendants
                class_is_renamed = class_node.name in renamed
                for node in ast.walk(class_node):
                    if not isinstance(node, ast.Call):
                        continue
                    target = call_target_name(node)
                    should_migrate = target in renamed
                    should_migrate = should_migrate or (class_allows_super and is_super_init(node))
                    should_migrate = should_migrate or (class_is_renamed and is_self_class_call(node))
                    if not should_migrate:
                        continue
                    for keyword in node.keywords:
                        if keyword.arg != LEGACY:
                            continue
                        start = absolute(line_offsets, keyword.lineno, keyword.col_offset)
                        if source[start : start + len(LEGACY)] == LEGACY:
                            edits.append((start, start + len(LEGACY), CANONICAL))

            # Also migrate top-level direct constructor calls.
            for node in tree.body:
                if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                    continue
                call = node.value
                if call_target_name(call) not in renamed:
                    continue
                for keyword in call.keywords:
                    if keyword.arg == LEGACY:
                        start = absolute(line_offsets, keyword.lineno, keyword.col_offset)
                        edits.append((start, start + len(LEGACY), CANONICAL))

            if edits:
                for start, end, replacement in sorted(set(edits), reverse=True):
                    source = source[:start] + replacement + source[end:]
                write(path, source)


def fix_stale_binary_annotation() -> None:
    path = MODELS / "classification" / "binary" / "base" / "models.py"
    text = read(path)
    text = text.replace('-> "GPClassificationModel":', '-> "BinaryClassificationGPModel":')
    write(path, text)


def extend_contract_test() -> None:
    path = ROOT / "tests" / "test_model_contract_refactor.py"
    text = read(path)
    additions: list[str] = []

    if "test_model_constructors_use_num_inducing" not in text:
        additions.append(r'''

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
''')

    if "test_binary_conditioning_annotation_uses_canonical_name" not in text:
        additions.append(r'''

def test_binary_conditioning_annotation_uses_canonical_name() -> None:
    path = MODELS_ROOT / "classification" / "binary" / "base" / "models.py"
    source = path.read_text(encoding="utf-8")
    assert '"GPClassificationModel"' not in source
''')

    if additions:
        write(path, text.rstrip() + "\n" + "".join(additions).rstrip() + "\n")


def validate(renamed: set[str]) -> None:
    offenders: list[str] = []
    stale_state: list[str] = []
    for path in MODELS.rglob("*.py"):
        tree = ast.parse(read(path), filename=str(path))
        for class_node in [node for node in tree.body if isinstance(node, ast.ClassDef)]:
            init = constructor(class_node)
            if init is not None and LEGACY in {arg.arg for arg in function_args(init)}:
                offenders.append(f"{path.relative_to(ROOT)}::{class_node.name}")
            if class_node.name in renamed:
                for node in ast.walk(class_node):
                    if (
                        isinstance(node, ast.Attribute)
                        and node.attr == LEGACY
                        and isinstance(node.value, ast.Name)
                        and node.value.id == "self"
                    ):
                        stale_state.append(f"{path.relative_to(ROOT)}::{class_node.name}")
    if offenders:
        raise RuntimeError("legacy constructor arguments remain:\n" + "\n".join(offenders))
    if stale_state:
        raise RuntimeError("legacy model state remains:\n" + "\n".join(stale_state))

    binary = read(MODELS / "classification" / "binary" / "base" / "models.py")
    if '"GPClassificationModel"' in binary:
        raise RuntimeError("stale GPClassificationModel annotation remains")


if __name__ == "__main__":
    renamed_classes, base_map = collect_renamed_classes()
    if not renamed_classes:
        raise RuntimeError("no num_inducing_points model constructors found")
    descendants = descendants_of(renamed_classes, base_map)
    canonicalize_model_constructors(renamed_classes)
    migrate_python_call_sites(renamed_classes, descendants)
    fix_stale_binary_annotation()
    extend_contract_test()
    validate(renamed_classes)
    print(f"canonicalized {len(renamed_classes)} model constructor classes")
