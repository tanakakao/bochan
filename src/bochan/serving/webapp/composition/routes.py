"""FastAPI routes for validating and optimizing one composition formula column."""

from __future__ import annotations

from typing import Any

from ..schemas.composition import (
    CompositionRegressionRunRequest,
    CompositionValidationRequest,
)
from .support import (
    _composition_transformer,
    normalize_web_composition_settings,
)


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
            model_kwargs["web_composition"] = request.composition.model_dump(exclude_none=True)
            validated = request.run.model_copy(update={"model_kwargs": model_kwargs})
            base_run = _route_endpoint(
                app,
                f"{prefix}/regression/run",
                "POST",
            )
            return base_run(validated)

        app.post(optimization_path)(run_composition_regression)


__all__ = ["register_composition_routes"]
