"""Visualization helpers for bochan."""

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
from .feature_importance import (
    ard_diagnostics_dataframe,
    build_feature_importance_figures,
    cross_validated_feature_importance_dataframe,
    feature_importance_dataframe,
    show_ard_diagnostics,
    show_cross_validated_feature_importance,
    show_feature_importance,
    show_pca_explained_variance,
    show_task_correlation_diagnostics,
)
from .input_perturbation import (
    prediction_mean_std as _prediction_mean_std_with_input_perturbation,
)

# Direct ``utils`` imports retain the perturbation-aware display semantics while
# prediction DataFrame builders now own that dependency directly.
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
from .ordinal_display import (
    OrdinalDisplayMode,
    OrdinalProbabilityMode,
    show_1dplot_from_optimizer,
    show_scatter_with_acqf_from_optimizer,
    show_triscatter_with_acqf_from_optimizer,
)
from .probability_1d import (
    show_1dplot_from_optimizer as _show_probability_1dplot_from_optimizer,
)
from .study import (
    show_optimization_history_study,
    show_pareto_front_study,
    study_history_dataframe,
    study_pareto_dataframe,
)
from .target_relation import show_target_relation_plot
from .plots import (
    show_1dplot_with_pred,
    show_pareto_plot,
    show_scatter_with_acqf,
    show_target_over_cycle_study,
    show_triscatter_with_acqf,
    show_yyplot,
)
from .utils import CYCLE_COLORS

show_1dplot_from_optimizer = _show_probability_1dplot_from_optimizer

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
    "ard_diagnostics_dataframe",
    "build_feature_importance_figures",
    "candidates_dataframe",
    "create_grid",
    "cross_validated_feature_importance_dataframe",
    "feature_importance_dataframe",
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
    "show_1dplot_from_optimizer",
    "show_1dplot_with_pred",
    "show_ard_diagnostics",
    "show_cross_validated_feature_importance",
    "show_feature_importance",
    "show_multiclass_1dplot",
    "show_multiclass_1dplot_from_optimizer",
    "show_multiclass_heatmap",
    "show_multiclass_heatmap_from_optimizer",
    "show_multiclass_triscatter",
    "show_multiclass_triscatter_from_optimizer",
    "show_multiclass_yyplot",
    "show_optimization_history_study",
    "show_ordinal_1dplot_from_optimizer",
    "show_ordinal_heatmap_from_optimizer",
    "show_ordinal_triscatter_from_optimizer",
    "show_pareto_front_study",
    "show_pareto_plot",
    "show_pca_explained_variance",
    "show_scatter_with_acqf",
    "show_scatter_with_acqf_from_optimizer",
    "show_target_over_cycle_study",
    "show_target_relation_plot",
    "show_task_correlation_diagnostics",
    "show_triscatter_with_acqf",
    "show_triscatter_with_acqf_from_optimizer",
    "show_yyplot",
    "show_yyplot_from_optimizer",
    "study_history_dataframe",
    "study_pareto_dataframe",
    "training_dataframe",
    "tri_grid",
]
