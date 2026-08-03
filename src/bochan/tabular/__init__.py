'''Tabular pandas / numpy API for bochan.'''

from .builders import (
    UNSET,
    drop_unset,
    make_acquisition_config,
    make_fit_config,
    make_model_config,
    make_objective_config,
    make_optimize_config,
    make_repair_config,
)
from .candidate_outputs import apply_tabular_candidate_outputs
from .composition import (
    ATOMIC_NUMBERS,
    ATOMIC_WEIGHTS,
    CompositionColumnConfig,
    CompositionDescriptorCalculator,
    CompositionSearchSpace,
    CompositionTabularPreprocessor,
    CompositionTransformer,
    SimplexTransform,
    close_compositions,
    element_order,
    format_formula,
    ilr_basis,
    normalize_composition,
    parse_formula,
)
from .config import ColumnKey, TabularDataConfig, TabularFeatureGroup
from .converter import (
    TabularDataset,
    bounds_to_tensor,
    dataframe_to_tensors,
    numpy_to_tensors,
    resolve_column_indices,
    resolve_optimize_config_columns,
    resolve_repair_config_columns,
    tensor_to_dataframe,
)
from .multi_output_categories import apply_tabular_multi_output_categories
from .optimizer_api import TabularBayesianOptimizer
from .ordinal_rank_labels import apply_tabular_ordinal_rank_labels
from .prediction_labels import apply_tabular_prediction_labels

apply_tabular_multi_output_categories()
apply_tabular_ordinal_rank_labels()
apply_tabular_candidate_outputs()
apply_tabular_prediction_labels()

__all__ = [
    "ATOMIC_NUMBERS",
    "ATOMIC_WEIGHTS",
    "ColumnKey",
    "CompositionColumnConfig",
    "CompositionDescriptorCalculator",
    "CompositionSearchSpace",
    "CompositionTabularPreprocessor",
    "CompositionTransformer",
    "SimplexTransform",
    "TabularBayesianOptimizer",
    "TabularDataConfig",
    "TabularDataset",
    "TabularFeatureGroup",
    "UNSET",
    "bounds_to_tensor",
    "close_compositions",
    "dataframe_to_tensors",
    "drop_unset",
    "element_order",
    "format_formula",
    "ilr_basis",
    "make_acquisition_config",
    "make_fit_config",
    "make_model_config",
    "make_objective_config",
    "make_optimize_config",
    "make_repair_config",
    "normalize_composition",
    "numpy_to_tensors",
    "parse_formula",
    "resolve_column_indices",
    "resolve_optimize_config_columns",
    "resolve_repair_config_columns",
    "tensor_to_dataframe",
]
