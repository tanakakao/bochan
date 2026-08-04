"""FastAPI routes for validating and optimizing one composition formula column."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .app import RegressionRunRequest
from .composition_web_support import (
    _composition_transformer,
    normalize_web_composition_settings,
)


class _CompositionSchema(BaseModel):
    """Strict base schema for the composition-specific Web API."""

    model_config = ConfigDict(extra="forbid")


class CompositionElementTermSchema(_CompositionSchema):
    """One coefficient multiplied by one element amount."""

    element: str
    coefficient: float = 1.0


class CompositionElementConstraintSchema(_CompositionSchema):
    """Linear equality or inequality between element amounts."""

    terms: list[CompositionElementTermSchema] = Field(min_length=1)
    operator: Literal["=", "<=", ">="] = "="
    rhs: float = 0.0
    basis: Literal["atomic_amount", "weight_amount"] = "atomic_amount"


class CompositionSettingsSchema(_CompositionSchema):
    """Single-formula ratio settings accepted by validation and optimization."""

    enabled: bool = True
    column: str = Field(min_length=1)
    elements: list[str] = Field(default_factory=list)
    normalization: Literal["atomic_fraction", "weight_fraction"] = "atomic_fraction"
    representation: Literal["fractions", "fraction", "clr", "alr", "ilr"] = "ilr"
    reference_element: str | None = None
    pseudocount: float = Field(default=1e-12, gt=0.0)
    precision: int = Field(default=6, ge=1, le=12)
    total: float = Field(default=1.0, gt=0.0)
    coordinate_bounds: tuple[float, float] = (-8.0, 8.0)
    min_components: int = Field(default=1, ge=1)
    max_components: int | None = Field(default=None, ge=1)
    required_components: list[str] = Field(default_factory=list)
    bounds: dict[str, tuple[float, float]] = Field(default_factory=dict)
    steps: dict[str, float] = Field(default_factory=dict)
    element_constraints: list[CompositionElementConstraintSchema] = Field(
        default_factory=list
    )


class CompositionValidationRequest(_CompositionSchema):
    """Formula rows and the same settings used by the optimization request."""

    formulas: list[str] = Field(min_length=1)
    settings: CompositionSettingsSchema


class CompositionRegressionRunRequest(_CompositionSchema):
    """Existing Web regression payload plus one typed composition configuration."""

    run: RegressionRunRequest
    composition: CompositionSettingsSchema


def _route_endpoint(app: Any, path: str, method: str) -> Any:
    for route in app.routes:
        if getattr(route, "path", None) != path:
            continue
        methods = set(getattr(route, "methods", ()) or ())
        if method.upper() in methods:
            return route.endpoint
    raise RuntimeError(f"Required Web route is not registered: {method} {path}")


def register_composition_routes(app: Any, *, api_prefix: str = "/api/v1") -> None:
    """Register typed validation and composition-optimization endpoints."""

    prefix = api_prefix.rstrip("/")
    validation_path = f"{prefix}/composition/validate"
    optimization_path = f"{prefix}/composition/regression/run"
    existing_paths = {getattr(route, "path", None) for route in app.routes}

    if validation_path not in existing_paths:

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

        app.post(validation_path)(validate_composition)

    if optimization_path not in existing_paths:

        def run_composition_regression(
            request: CompositionRegressionRunRequest,
        ) -> dict[str, Any]:
            model_kwargs = dict(request.run.model_kwargs or {})
            model_kwargs["web_composition"] = request.composition.model_dump(
                exclude_none=True
            )
            validated = request.run.model_copy(
                update={"model_kwargs": model_kwargs}
            )
            base_run = _route_endpoint(
                app,
                f"{prefix}/regression/run",
                "POST",
            )
            return base_run(validated)

        app.post(optimization_path)(run_composition_regression)


__all__ = [
    "CompositionElementConstraintSchema",
    "CompositionElementTermSchema",
    "CompositionRegressionRunRequest",
    "CompositionSettingsSchema",
    "CompositionValidationRequest",
    "register_composition_routes",
]
