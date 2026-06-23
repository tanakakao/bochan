"""Visualization helpers for bochan."""

from . import plots as _plots
from .data import (
    candidates_dataframe,
    create_grid,
    get_const_array,
    get_yyplot_data,
    grid_1d_plot,
    grid_2d,
    prediction_dataframe,
    training_dataframe,
    tri_grid,
)
from .multiclass import (
    MulticlassHeatmapMode,
    infer_class_labels,
    is_multiclass_object,
    multiclass_grid_1d,
    multiclass_grid_2d,
    multiclass_prediction_dataframe,
    multiclass_probabilities,
    show_multiclass_1dplot,
    show_multiclass_1dplot_from_optimizer,
    show_multiclass_heatmap,
    show_multiclass_heatmap_from_optimizer,
)
from .multiclass_ternary import (
    multiclass_tri_grid,
    show_multiclass_triscatter,
    show_multiclass_triscatter_from_optimizer,
)
from .ordinal import (
    OrdinalDisplayMode,
    OrdinalProbabilityMode,
    is_ordinal_object,
    ordinal_grid_1d,
    ordinal_grid_2d,
    ordinal_prediction_dataframe,
    ordinal_probabilities,
    ordinal_tri_grid,
    show_1dplot_from_optimizer,
    show_ordinal_1dplot_from_optimizer,
    show_ordinal_heatmap_from_optimizer,
    show_ordinal_triscatter_from_optimizer,
    show_scatter_with_acqf_from_optimizer,
    show_triscatter_with_acqf_from_optimizer,
)
from .plots import (
    show_1dplot_with_pred,
    show_pareto_plot,
    show_scatter_with_acqf,
    show_target_over_cycle_study,
    show_triscatter_with_acqf,
    show_yyplot,
    show_yyplot_from_optimizer,
)
from .utils import CYCLE_COLORS

# Keep direct imports from ``bochan.visualization.plots`` consistent with the
# package-level API. The wrappers delegate to the existing implementations when
# the optional multiclass / ordinal probability displays are not selected.
_plots.show_1dplot_from_optimizer = show_1dplot_from_optimizer
_plots.show_scatter_with_acqf_from_optimizer = show_scatter_with_acqf_from_optimizer
_plots.show_triscatter_with_acqf_from_optimizer = show_triscatter_with_acqf_from_optimizer

__all__ = [
    "CYCLE_COLORS",
    "MulticlassHeatmapMode",
    "OrdinalDisplayMode",
    "OrdinalProbabilityMode",
    "candidates_dataframe",
    "create_grid",
    "get_const_array",
    "get_yyplot_data",
    "grid_1d_plot",
    "grid_2d",
    "infer_class_labels",
    "is_multiclass_object",
    "is_ordinal_object",
    "multiclass_grid_1d",
    "multiclass_grid_2d",
    "multiclass_prediction_dataframe",
    "multiclass_probabilities",
    "multiclass_tri_grid",
    "ordinal_grid_1d",
    "ordinal_grid_2d",
    "ordinal_prediction_dataframe",
    "ordinal_probabilities",
    "ordinal_tri_grid",
    "prediction_dataframe",
    "training_dataframe",
    "tri_grid",
    "show_1dplot_from_optimizer",
    "show_1dplot_with_pred",
    "show_multiclass_1dplot",
    "show_multiclass_1dplot_from_optimizer",
    "show_multiclass_heatmap",
    "show_multiclass_heatmap_from_optimizer",
    "show_multiclass_triscatter",
    "show_multiclass_triscatter_from_optimizer",
    "show_ordinal_1dplot_from_optimizer",
    "show_ordinal_heatmap_from_optimizer",
    "show_ordinal_triscatter_from_optimizer",
    "show_pareto_plot",
    "show_scatter_with_acqf",
    "show_scatter_with_acqf_from_optimizer",
    "show_target_over_cycle_study",
    "show_triscatter_with_acqf",
    "show_triscatter_with_acqf_from_optimizer",
    "show_yyplot",
    "show_yyplot_from_optimizer",
]
