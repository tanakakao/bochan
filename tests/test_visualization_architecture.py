"""Architecture contract for :mod:`bochan.visualization`."""

from __future__ import annotations

import ast
from pathlib import Path

import bochan.visualization as visualization
from bochan.visualization import input_perturbation, multiclass, ordinal, plots
from bochan.visualization.data import frames, grids, study, ternary


_VISUALIZATION_DIR = Path(__file__).parents[1] / "src" / "bochan" / "visualization"
_EXPECTED_ROOT = {
    "ARCHITECTURE.md",
    "README.md",
    "__init__.py",
    "_heatmap_layout.py",
    "categorical_axis.py",
    "data",
    "feature_importance.py",
    "heteroscedastic_1d.py",
    "input_perturbation.py",
    "multiclass.py",
    "multiclass_ternary.py",
    "multiclass_yy.py",
    "ordinal.py",
    "ordinal_display.py",
    "plots.py",
    "probability_1d.py",
    "study.py",
    "target_relation.py",
    "utils.py",
}


def test_visualization_root_has_only_declared_owners() -> None:
    """Keep the package root intentional instead of accumulating ad-hoc modules."""

    actual = {
        path.name
        for path in _VISUALIZATION_DIR.iterdir()
        if path.name != "__pycache__"
    }
    assert actual == _EXPECTED_ROOT


def test_removed_compatibility_modules_do_not_return() -> None:
    """Prevent removed forwarding / patch modules from being recreated."""

    assert not (_VISUALIZATION_DIR / "data.py").exists()
    assert not (_VISUALIZATION_DIR / "probability_input_perturbation.py").exists()


def test_package_root_does_not_patch_imported_modules() -> None:
    """The public package initializer may export names but may not monkey-patch modules."""

    source = (_VISUALIZATION_DIR / "__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    attribute_assignments: list[ast.Attribute] = []
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets.extend(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets.append(node.target)
        elif isinstance(node, ast.AugAssign):
            targets.append(node.target)
        attribute_assignments.extend(
            target for target in targets if isinstance(target, ast.Attribute)
        )

    assert attribute_assignments == []
    assert "setattr(" not in source


def test_public_functions_have_concrete_owners() -> None:
    """Lock the canonical owner for the main visualization contracts."""

    assert frames.prediction_dataframe.__module__ == "bochan.visualization.data.frames"
    assert grids.grid_1d_plot.__module__ == "bochan.visualization.data.grids"
    assert ternary.tri_grid.__module__ == "bochan.visualization.data.ternary"
    assert study.study_target_dataframe.__module__ == "bochan.visualization.data.study"

    assert multiclass.multiclass_probabilities.__module__ == "bochan.visualization.multiclass"
    assert ordinal.ordinal_probabilities.__module__ == "bochan.visualization.ordinal"
    assert (
        input_perturbation.prediction_mean_std.__module__
        == "bochan.visualization.input_perturbation"
    )

    assert plots.show_1dplot_from_optimizer.__module__ == "bochan.visualization.plots"
    assert (
        plots.show_scatter_with_acqf_from_optimizer.__module__
        == "bochan.visualization.plots"
    )
    assert (
        plots.show_triscatter_with_acqf_from_optimizer.__module__
        == "bochan.visualization.plots"
    )
    assert plots.show_yyplot_from_optimizer.__module__ == "bochan.visualization.plots"


def test_package_root_exports_canonical_plot_dispatchers() -> None:
    """Package-root plotting imports must be the canonical ``plots`` functions."""

    assert visualization.show_1dplot_from_optimizer is plots.show_1dplot_from_optimizer
    assert (
        visualization.show_scatter_with_acqf_from_optimizer
        is plots.show_scatter_with_acqf_from_optimizer
    )
    assert (
        visualization.show_triscatter_with_acqf_from_optimizer
        is plots.show_triscatter_with_acqf_from_optimizer
    )
    assert visualization.show_yyplot_from_optimizer is plots.show_yyplot_from_optimizer
