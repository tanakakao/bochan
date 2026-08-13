"""Target semantics for tabular outputs."""

from .categories import (
    category_map_from_output_config,
    extract_output_category_maps,
    merge_target_category_metadata,
    resolve_acquisition_config_columns,
    resolve_constraint_target_classes,
    resolve_outcome_constraint_config_columns,
    resolve_target_class_value,
    target_category_map_for_output,
)
from .ordinal import (
    resolve_acquisition_ordinal_ranks,
    resolve_ordinal_rank_config,
    resolve_ordinal_rank_constraint,
)

__all__ = [
    "category_map_from_output_config",
    "extract_output_category_maps",
    "merge_target_category_metadata",
    "resolve_acquisition_config_columns",
    "resolve_acquisition_ordinal_ranks",
    "resolve_constraint_target_classes",
    "resolve_ordinal_rank_config",
    "resolve_ordinal_rank_constraint",
    "resolve_outcome_constraint_config_columns",
    "resolve_target_class_value",
    "target_category_map_for_output",
]
