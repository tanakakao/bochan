"""Compatibility exports for the Web target workflow."""

from .workflows_tabular import (
    _build_outcome_constraint_config,
    _figure_payload,
    _resolve_target_settings,
    _resolve_targets,
    run_regression_web_workflow,
)

__all__ = [
    "_build_outcome_constraint_config",
    "_figure_payload",
    "_resolve_target_settings",
    "_resolve_targets",
    "run_regression_web_workflow",
]
