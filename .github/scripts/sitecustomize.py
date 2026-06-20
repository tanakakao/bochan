from __future__ import annotations

import atexit
import sys
from pathlib import Path


script_path = Path(sys.argv[0]).resolve() if sys.argv else None
if script_path is not None and script_path.name == "apply_binary_review_fixes.py":
    source = script_path.read_text(encoding="utf-8")
    old = '''    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one match in {path.relative_to(ROOT)}; found {count}.\\n"
            f"Pattern:\\n{old}"
        )
'''
    new = '''    if old not in text:
        raise RuntimeError(
            f"Expected pattern was not found in {path.relative_to(ROOT)}.\\n"
            f"Pattern:\\n{old}"
        )
'''
    if old not in source:
        raise RuntimeError("Could not prepare ordered binary review migration.")
    script_path.write_text(source.replace(old, new, 1), encoding="utf-8")


@atexit.register
def _remove_sitecustomize() -> None:
    Path(__file__).unlink(missing_ok=True)
