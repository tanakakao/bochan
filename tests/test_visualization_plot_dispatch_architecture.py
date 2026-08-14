from __future__ import annotations

from pathlib import Path

import bochan.visualization as visualization
from bochan.visualization import multiclass, multiclass_ternary, multiclass_yy, plots


def test_public_optimizer_plot_dispatch_is_owned_by_plots() -> None:
    public_functions = (
        "show_1dplot_from_optimizer",
        "show_scatter_with_acqf_from_optimizer",
        "show_triscatter_with_acqf_from_optimizer",
        "show_yyplot_from_optimizer",
    )

    for name in public_functions:
        function = getattr(plots, name)
        assert getattr(visualization, name) is function
        assert function.__module__ == "bochan.visualization.plots"


def test_specialized_fallbacks_keep_generic_plot_implementations() -> None:
    assert multiclass._show_1dplot_from_optimizer is not plots.show_1dplot_from_optimizer
    assert (
        multiclass._show_scatter_with_acqf_from_optimizer
        is not plots.show_scatter_with_acqf_from_optimizer
    )
    assert (
        multiclass_ternary._show_triscatter_with_acqf_from_optimizer
        is not plots.show_triscatter_with_acqf_from_optimizer
    )
    assert multiclass_yy._show_yyplot_from_optimizer is not plots.show_yyplot_from_optimizer

    assert multiclass._show_1dplot_from_optimizer.__module__ == "bochan.visualization.plots"
    assert (
        multiclass._show_scatter_with_acqf_from_optimizer.__module__
        == "bochan.visualization.plots"
    )
    assert (
        multiclass_ternary._show_triscatter_with_acqf_from_optimizer.__module__
        == "bochan.visualization.plots"
    )
    assert multiclass_yy._show_yyplot_from_optimizer.__module__ == "bochan.visualization.plots"


def test_package_root_does_not_patch_plot_functions() -> None:
    source = Path(visualization.__file__).read_text(encoding="utf-8")

    assert "_plots.show_1dplot_from_optimizer" not in source
    assert "_plots.show_scatter_with_acqf_from_optimizer" not in source
    assert "_plots.show_triscatter_with_acqf_from_optimizer" not in source
    assert "_plots.show_yyplot_from_optimizer" not in source
