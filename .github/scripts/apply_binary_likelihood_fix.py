from __future__ import annotations

from pathlib import Path

from prepare_binary_likelihood_migration import main as prepare_main


prepare_main()

from apply_binary_likelihood_fix_impl import main as apply_main


apply_main()

script_dir = Path(__file__).resolve().parent
(script_dir / "apply_binary_likelihood_fix_impl.py").unlink(missing_ok=True)
(script_dir / "prepare_binary_likelihood_migration.py").unlink(missing_ok=True)
