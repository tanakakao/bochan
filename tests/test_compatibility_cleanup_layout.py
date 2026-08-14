from __future__ import annotations

from pathlib import Path

import bochan.constraints as constraints


def test_constraints_use_single_k_sparse_module() -> None:
    root = Path(constraints.__file__).resolve().parent

    assert (root / "k_sparse.py").is_file()
    assert not (root / "ksparse.py").exists()


def test_obsolete_levelset_objective_compat_modules_are_removed() -> None:
    root = (
        Path(constraints.__file__).resolve().parents[1]
        / "acquisition"
        / "regression"
        / "levelset_estimation"
    )

    assert not (root / "objective_compat.py").exists()
    assert not (root / "hetero_objective_compat.py").exists()
