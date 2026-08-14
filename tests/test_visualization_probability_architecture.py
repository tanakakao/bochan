from __future__ import annotations

from pathlib import Path

import bochan.visualization as visualization
from bochan.visualization import multiclass, ordinal


def test_probability_helpers_have_concrete_module_owners() -> None:
    assert visualization.multiclass_probabilities is multiclass.multiclass_probabilities
    assert visualization.ordinal_probabilities is ordinal.ordinal_probabilities
    assert multiclass.multiclass_probabilities.__module__ == (
        "bochan.visualization.multiclass"
    )
    assert ordinal.ordinal_probabilities.__module__ == "bochan.visualization.ordinal"


def test_probability_input_perturbation_patch_module_is_removed() -> None:
    package_dir = Path(visualization.__file__).resolve().parent

    assert not (package_dir / "probability_input_perturbation.py").exists()


def test_package_root_does_not_patch_probability_modules() -> None:
    source = Path(visualization.__file__).read_text(encoding="utf-8")

    assert "_multiclass.multiclass_probabilities =" not in source
    assert "_multiclass_ternary.multiclass_probabilities =" not in source
    assert "_multiclass_yy.multiclass_probabilities =" not in source
    assert "_ordinal.ordinal_probabilities =" not in source
