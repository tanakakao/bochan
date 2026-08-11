"""Shared infrastructure for correlated multi-task models."""

from .kronecker import (
    BlockDesignVariationalELBO,
    LatentKroneckerMultiTaskGP,
    canonicalize_block_design_targets,
    canonicalize_shared_inducing_points,
)
from .mixed import WideMixedMultiTaskGP
from .task_feature import (
    PerturbationAwareStratifiedStandardize,
    PerturbationAwareWidePosterior,
    TaskFeatureInputTransform,
    WideMultiTaskBinaryClassificationGPModel,
    WideMultiTaskGP,
    WideMultiTaskMulticlassClassificationGPModel,
    WideMultiTaskOrdinalGPModel,
    wide_to_long,
)
from .validation import (
    long_to_sparse_wide,
    validate_complete_block,
    validate_long_multitask_data,
)

__all__ = [
    "BlockDesignVariationalELBO",
    "LatentKroneckerMultiTaskGP",
    "PerturbationAwareStratifiedStandardize",
    "PerturbationAwareWidePosterior",
    "TaskFeatureInputTransform",
    "WideMixedMultiTaskGP",
    "WideMultiTaskBinaryClassificationGPModel",
    "WideMultiTaskGP",
    "WideMultiTaskMulticlassClassificationGPModel",
    "WideMultiTaskOrdinalGPModel",
    "canonicalize_block_design_targets",
    "canonicalize_shared_inducing_points",
    "long_to_sparse_wide",
    "validate_complete_block",
    "validate_long_multitask_data",
    "wide_to_long",
]
