from __future__ import annotations

import ast
from pathlib import Path

from prepare_binary_likelihood_migration import main as prepare_main


prepare_main()

from apply_binary_likelihood_fix_impl import main as apply_main


apply_main()

root = Path(__file__).resolve().parents[2]
test_path = root / "tests/test_binary_likelihood_consistency.py"
test_text = test_path.read_text(encoding="utf-8")
test_text = test_text.replace(
    'assert not violations, "\n".join(violations)',
    'assert not violations, "\\n".join(violations)',
)
test_path.write_text(test_text, encoding="utf-8")

for path in sorted((root / "src/bochan/acquisition/binary").rglob("*.py")):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
ast.parse(
    test_path.read_text(encoding="utf-8"),
    filename="tests/test_binary_likelihood_consistency.py",
)

script_dir = Path(__file__).resolve().parent
(script_dir / "apply_binary_likelihood_fix_impl.py").unlink(missing_ok=True)
(script_dir / "prepare_binary_likelihood_migration.py").unlink(missing_ok=True)
