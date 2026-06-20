from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


script_dir = Path(__file__).resolve().parent
impl_path = script_dir / "_binary_review_impl.py"
source = subprocess.check_output(
    [
        "git",
        "show",
        "2afadd23a5a11cd3c75edc7154cc6c20e3ffeffe:.github/scripts/apply_binary_review_fixes.py",
    ],
    text=True,
)
impl_path.write_text(source, encoding="utf-8")

spec = importlib.util.spec_from_file_location("binary_review_impl", impl_path)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load binary review migration implementation.")
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


def replace_first(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(
            f"Expected pattern was not found in {path.relative_to(migration.ROOT)}.\n"
            f"Pattern:\n{old}"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


migration.replace_once = replace_first
migration.main()

impl_path.unlink(missing_ok=True)
(script_dir / "run_binary_review_fixes.py").unlink(missing_ok=True)
(script_dir / "sitecustomize.py").unlink(missing_ok=True)
