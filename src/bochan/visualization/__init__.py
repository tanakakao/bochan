"""Visualization helpers for bochan."""

from . import data as _data
from . import plots as _plots
from . import utils as _utils
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
from .input_perturbation import (
    prediction_mean_std as _prediction_mean_std_with_input_perturbation,
)

# ``data.py`` imports prediction_mean_std directly. Replace both references so
# package and direct submodule imports use perturbation-aware visualization.
_data.prediction_mean_std = _prediction_mean_std_with_input_perturbation
_utils.prediction_mean_std = _prediction_mean_std_with_input_perturbation

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
from .multiclass_yy import (
    show_multiclass_yyplot,
    show_yyplot_from_optimizer,
)
from .ordinal import (
    is_ordinal_object,
    ordinal_grid_1d,
    ordinal_grid_2d,
    ordinal_prediction_dataframe,
    ordinal_probabilities,
    ordinal_tri_grid,
    show_ordinal_1dplot_from_optimizer,
    show_ordinal_heatmap_from_optimizer,
    show_ordinal_triscatter_from_optimizer,
)

# Probability models with input perturbation can return ``n_points * n_w`` rows.
# Replace every module-level reference so 1D, 2D, ternary, YY, and direct helper
# calls consistently aggregate those rows back to one probability vector per
# original input point.
from . import multiclass as _multiclass
from . import multiclass_ternary as _multiclass_ternary
from . import multiclass_yy as _multiclass_yy
from . import ordinal as _ordinal
from .probability_input_perturbation import (
    multiclass_probabilities as _multiclass_probabilities_with_input_perturbation,
    ordinal_probabilities as _ordinal_probabilities_with_input_perturbation,
)

_multiclass.multiclass_probabilities = (
    _multiclass_probabilities_with_input_perturbation
)
_multiclass_ternary.multiclass_probabilities = (
    _multiclass_probabilities_with_input_perturbation
)
_multiclass_yy.multiclass_probabilities = (
    _multiclass_probabilities_with_input_perturbation
)
_ordinal.ordinal_probabilities = _ordinal_probabilities_with_input_perturbation
multiclass_probabilities = _multiclass_probabilities_with_input_perturbation
ordinal_probabilities = _ordinal_probabilities_with_input_perturbation

from .ordinal_display import (
    OrdinalDisplayMode,
    OrdinalProbabilityMode,
    show_1dplot_from_optimizer,
    show_scatter_with_acqf_from_optimizer,
    show_triscatter_with_acqf_from_optimizer,
)
from .study import (
    show_optimization_history_study,
    show_pareto_front_study,
    study_history_dataframe,
    study_pareto_dataframe,
)
from .plots import (
    show_1dplot_with_pred,
    show_pareto_plot,
    show_scatter_with_acqf,
    show_target_over_cycle_study,
    show_triscatter_with_acqf,
    show_yyplot,
)
from .utils import CYCLE_COLORS

# Keep direct imports from ``bochan.visualization.plots`` consistent with the
# package-level API. The wrappers delegate to the existing implementations when
# the optional multiclass / ordinal probability displays are not selected.
_plots.show_1dplot_from_optimizer = show_1dplot_from_optimizer
_plots.show_scatter_with_acqf_from_optimizer = show_scatter_with_acqf_from_optimizer
_plots.show_triscatter_with_acqf_from_optimizer = show_triscatter_with_acqf_from_optimizer
_plots.show_yyplot_from_optimizer = show_yyplot_from_optimizer

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
    "study_history_dataframe",
    "study_pareto_dataframe",
    "tri_grid",
    "show_1dplot_from_optimizer",
    "show_1dplot_with_pred",
    "show_multiclass_1dplot",
    "show_multiclass_1dplot_from_optimizer",
    "show_multiclass_heatmap",
    "show_multiclass_heatmap_from_optimizer",
    "show_multiclass_triscatter",
    "show_multiclass_triscatter_from_optimizer",
    "show_multiclass_yyplot",
    "show_ordinal_1dplot_from_optimizer",
    "show_ordinal_heatmap_from_optimizer",
    "show_ordinal_triscatter_from_optimizer",
    "show_optimization_history_study",
    "show_pareto_front_study",
    "show_pareto_plot",
    "show_scatter_with_acqf",
    "show_scatter_with_acqf_from_optimizer",
    "show_target_over_cycle_study",
    "show_triscatter_with_acqf",
    "show_triscatter_with_acqf_from_optimizer",
    "show_yyplot",
    "show_yyplot_from_optimizer",
]
