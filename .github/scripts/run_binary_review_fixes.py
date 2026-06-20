from __future__ import annotations

from pathlib import Path

import apply_binary_review_fixes as migration


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
