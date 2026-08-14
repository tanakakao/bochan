"""Composition validation and optimization routes for the Web API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter

from ..composition.support import (
    _composition_transformer,
    normalize_web_composition_settings,
)
from ..schemas.composition import (
    CompositionRegressionRunRequest,
    CompositionValidationRequest,
)
from ..schemas.regression import RegressionRunRequest


def create_composition_router(
    *,
    run_regression: Callable[[RegressionRunRequest], dict[str, Any]],
) -> APIRouter:
    """Create typed composition validation and optimization routes."""

    router = APIRouter(tags=["web-composition"])

    @router.post("/composition/validate")
    def validate_composition(
        request: CompositionValidationRequest,
    ) -> dict[str, Any]:
        import pandas as pd

        from bochan.tabular.composition import (
            format_formula,
            normalize_composition,
            parse_formula,
        )

        settings = request.settings.model_dump(exclude_none=True)
        config = normalize_web_composition_settings(settings)
        frame = pd.DataFrame({config["column"]: request.formulas})
        _transformed, resolved = _composition_transformer(frame, config)
        elements = list(resolved["elements"])
        rows: list[dict[str, Any]] = []
        for formula in request.formulas:
            parsed = parse_formula(formula)
            normalized = normalize_composition(
                parsed,
                mode=resolved["normalization"],
            )
            fractions = {
                element: float(normalized.get(element, 0.0))
                for element in elements
            }
            atomic = normalize_composition(parsed, mode="atomic_fraction")
            rows.append(
                {
                    "input": formula,
                    "formula": format_formula(
                        atomic,
                        order=elements,
                        precision=resolved["precision"],
                    ),
                    "fractions": fractions,
                }
            )
        return {
            "column": resolved["column"],
            "elements": elements,
            "representation": resolved["representation"],
            "normalization": resolved["normalization"],
            "feature_names": list(resolved["feature_names"]),
            "rows": rows,
        }

    @router.post("/composition/regression/run")
    def run_composition_regression(
        request: CompositionRegressionRunRequest,
    ) -> dict[str, Any]:
        model_kwargs = dict(request.run.model_kwargs or {})
        model_kwargs["web_composition"] = request.composition.model_dump(exclude_none=True)
        validated = request.run.model_copy(update={"model_kwargs": model_kwargs})
        return run_regression(validated)

    return router


__all__ = ["create_composition_router"]
