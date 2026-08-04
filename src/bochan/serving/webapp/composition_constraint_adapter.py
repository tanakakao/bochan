"""Keep ordinary Web linear constraints aligned after formula expansion."""

from __future__ import annotations

from typing import Any

from .composition_web_support import _ACTIVE_CONFIG

_INSTALLED = False


def install_composition_constraint_adapter() -> None:
    """Use transformed feature names when building BoTorch linear constraints."""

    global _INSTALLED
    if _INSTALLED:
        return
    from . import workflows_tabular

    original = workflows_tabular.botorch_linear_constraints

    def adapted(
        constraints: list[Any],
        *,
        feature_columns: list[str],
    ) -> tuple[list[Any], list[Any]]:
        config = _ACTIVE_CONFIG.get()
        if config is None:
            return original(constraints, feature_columns=feature_columns)
        model_columns: list[str] = []
        for column in feature_columns:
            if column == config.get("column"):
                model_columns.extend(config.get("feature_names") or ())
            else:
                model_columns.append(column)
        return original(constraints, feature_columns=model_columns)

    workflows_tabular.botorch_linear_constraints = adapted
    _INSTALLED = True


__all__ = ["install_composition_constraint_adapter"]
