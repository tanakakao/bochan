"""FastAPI routes for validating single-formula composition settings."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .composition_web_support import (
    _composition_transformer,
    normalize_web_composition_settings,
)


class CompositionValidationRequest(BaseModel):
    """Formula rows and the same settings used by the Web optimization request."""

    model_config = ConfigDict(extra="forbid")

    formulas: list[str] = Field(min_length=1)
    settings: dict[str, Any]


def register_composition_routes(app: Any, *, api_prefix: str = "/api/v1") -> None:
    """Register formula/config validation without changing the regression schema."""

    route_path = f"{api_prefix.rstrip('/')}/composition/validate"
    if any(getattr(route, "path", None) == route_path for route in app.routes):
        return

    def validate_composition(request: CompositionValidationRequest) -> dict[str, Any]:
        import pandas as pd

        from bochan.tabular.composition import (
            format_formula,
            normalize_composition,
            parse_formula,
        )

        config = normalize_web_composition_settings(request.settings)
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

    app.post(route_path)(validate_composition)


__all__ = ["CompositionValidationRequest", "register_composition_routes"]
