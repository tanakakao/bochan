"""Single-formula composition support for the React/FastAPI workbench.

The Web regression request keeps its backward-compatible public schema and sends
composition settings through ``model_kwargs.web_composition``.  This module
adapts the existing Web workflow to the public composition-aware
``TabularBayesianOptimizer`` without duplicating the target/acquisition logic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import replace
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np

_SITE_NAME = "composition"
_ACTIVE_CONFIG: ContextVar[dict[str, Any] | None] = ContextVar(
    "bochan_web_composition_config",
    default=None,
)
_INSTALLED = False

_REPRESENTATION_ALIASES = {
    "fraction": "fractions",
    "fractions": "fractions",
    "none": "fractions",
    "clr": "clr",
    "alr": "alr",
    "ilr": "ilr",
}
_NORMALIZATION_ALIASES = {
    "atomic": "atomic_fraction",
    "atomic_fraction": "atomic_fraction",
    "molar": "atomic_fraction",
    "weight": "weight_fraction",
    "weight_fraction": "weight_fraction",
    "mass_fraction": "weight_fraction",
}


def _finite(value: Any, default: float) -> float:
    parsed = float(value)
    return parsed if np.isfinite(parsed) else float(default)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",")]
    elif isinstance(value, Sequence):
        values = [str(item).strip() for item in value]
    else:
        raise TypeError("Composition element settings must be a sequence or comma-separated string.")
    return list(dict.fromkeys(item for item in values if item))


def normalize_web_composition_settings(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one Web composition-ratio configuration."""

    if not isinstance(raw, Mapping):
        raise TypeError("web_composition must be a mapping.")
    column = str(raw.get("column") or "").strip()
    if not column:
        raise ValueError("web_composition.column is required.")

    representation_key = str(raw.get("representation", "ilr")).lower()
    try:
        representation = _REPRESENTATION_ALIASES[representation_key]
    except KeyError as exc:
        raise ValueError("Composition representation must be fractions, clr, alr, or ilr.") from exc

    normalization_key = str(raw.get("normalization", "atomic_fraction")).lower()
    try:
        normalization = _NORMALIZATION_ALIASES[normalization_key]
    except KeyError as exc:
        raise ValueError("Composition normalization must be atomic_fraction or weight_fraction.") from exc

    elements = _string_list(raw.get("elements"))
    required = _string_list(raw.get("required_components", raw.get("required_elements")))
    bounds: dict[str, tuple[float, float]] = {}
    for element, pair in dict(raw.get("bounds") or {}).items():
        values = tuple(pair)
        if len(values) != 2:
            raise ValueError(f"Composition bounds for {element!r} must have two values.")
        lower, upper = map(float, values)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower < 0 or lower > upper:
            raise ValueError(f"Invalid composition bounds for {element!r}.")
        bounds[str(element)] = (lower, upper)

    steps: dict[str, float] = {}
    for element, value in dict(raw.get("steps") or {}).items():
        step = float(value)
        if not np.isfinite(step) or step <= 0:
            raise ValueError(f"Composition step for {element!r} must be positive.")
        steps[str(element)] = step

    coordinate_pair = tuple(raw.get("coordinate_bounds") or (-8.0, 8.0))
    if len(coordinate_pair) != 2:
        raise ValueError("coordinate_bounds must have two values.")
    coordinate_lower, coordinate_upper = map(float, coordinate_pair)
    if not np.isfinite(coordinate_lower) or not np.isfinite(coordinate_upper) or coordinate_lower >= coordinate_upper:
        raise ValueError("coordinate_bounds must be finite and increasing.")

    total = _finite(raw.get("total", 1.0), 1.0)
    if total <= 0:
        raise ValueError("Composition total must be positive.")
    min_components = int(raw.get("min_components", 1))
    max_components_raw = raw.get("max_components")
    max_components = None if max_components_raw in (None, "") else int(max_components_raw)
    if min_components < 1:
        raise ValueError("min_components must be at least 1.")
    if max_components is not None and max_components < min_components:
        raise ValueError("max_components must be greater than or equal to min_components.")

    constraints: list[dict[str, Any]] = []
    for index, item in enumerate(raw.get("element_constraints") or raw.get("constraints") or []):
        if not isinstance(item, Mapping):
            raise TypeError(f"Composition element constraint {index} must be a mapping.")
        terms: list[dict[str, Any]] = []
        for term_index, term in enumerate(item.get("terms") or []):
            if not isinstance(term, Mapping):
                raise TypeError(f"Term {term_index} in composition constraint {index} must be a mapping.")
            element = str(term.get("element") or "").strip()
            if not element:
                raise ValueError(f"Term {term_index} in composition constraint {index} requires an element.")
            coefficient = float(term.get("coefficient", 1.0))
            if not np.isfinite(coefficient):
                raise ValueError("Composition constraint coefficients must be finite.")
            terms.append(
                {
                    "site": _SITE_NAME,
                    "element": element,
                    "coefficient": coefficient,
                }
            )
        if not terms:
            raise ValueError(f"Composition element constraint {index} requires one or more terms.")
        operator = str(item.get("operator", "="))
        if operator == "==":
            operator = "="
        if operator not in {"=", "<=", ">="}:
            raise ValueError("Composition constraint operator must be =, <=, or >=.")
        rhs = float(item.get("rhs", 0.0))
        if not np.isfinite(rhs):
            raise ValueError("Composition constraint rhs must be finite.")
        basis = str(item.get("basis", "atomic_amount"))
        constraints.append(
            {
                "terms": terms,
                "operator": operator,
                "rhs": rhs,
                "basis": basis,
            }
        )

    return {
        "enabled": bool(raw.get("enabled", True)),
        "column": column,
        "elements": elements,
        "normalization": normalization,
        "representation": representation,
        "reference_element": raw.get("reference_element") or None,
        "pseudocount": _finite(raw.get("pseudocount", 1e-12), 1e-12),
        "precision": max(1, int(raw.get("precision", 6))),
        "total": total,
        "bounds": bounds,
        "steps": steps,
        "min_components": min_components,
        "max_components": max_components,
        "required_components": required,
        "coordinate_bounds": (coordinate_lower, coordinate_upper),
        "element_constraints": constraints,
    }


def _request_without_web_composition(request: Any) -> tuple[Any, dict[str, Any] | None]:
    model_kwargs = dict(getattr(request, "model_kwargs", None) or {})
    raw = model_kwargs.pop("web_composition", None)
    if not raw or not bool(dict(raw).get("enabled", True)):
        return request, None
    config = normalize_web_composition_settings(raw)
    if hasattr(request, "model_copy"):
        return request.model_copy(update={"model_kwargs": model_kwargs}), config
    values = dict(vars(request))
    values["model_kwargs"] = model_kwargs
    return SimpleNamespace(**values), config


def _composition_transformer(data: Any, config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    from bochan.tabular.composition import CompositionTransformer

    column = config["column"]
    if column not in data.columns:
        raise KeyError(f"Unknown composition formula column {column!r}.")
    transformer = CompositionTransformer(
        elements=config["elements"] or None,
        normalization=config["normalization"],
        representation=config["representation"],
        reference_element=config["reference_element"],
        pseudocount=config["pseudocount"],
        include_descriptors=False,
        prefix=column,
        precision=config["precision"],
    )
    transformer.fit(data.loc[:, column])
    elements = list(transformer._require_fitted())
    resolved = dict(config)
    resolved["elements"] = elements
    if resolved["max_components"] is None:
        resolved["max_components"] = len(elements)
    unknown = set(resolved["required_components"]) - set(elements)
    unknown.update(set(resolved["bounds"]) - set(elements))
    unknown.update(set(resolved["steps"]) - set(elements))
    for constraint in resolved["element_constraints"]:
        unknown.update(term["element"] for term in constraint["terms"] if term["element"] not in elements)
    if unknown:
        raise ValueError(f"Composition settings reference unknown elements: {sorted(unknown)!r}.")
    transformed = transformer.transform_frame(data, column, drop_formula=True)
    resolved["feature_names"] = list(transformer.feature_names_ or ())
    return transformed, resolved


def _coordinate_specs(config: dict[str, Any]) -> list[Any]:
    names = list(config["feature_names"])
    elements = list(config["elements"])
    specs: list[Any] = []
    if config["representation"] == "fractions":
        for index, name in enumerate(names):
            element = elements[index]
            lower, upper = config["bounds"].get(element, (0.0, config["total"]))
            specs.append(
                SimpleNamespace(
                    name=name,
                    type="numeric",
                    lower=float(lower) / config["total"],
                    upper=float(upper) / config["total"],
                    step=None,
                    fixed=False,
                    fixed_value=None,
                )
            )
    else:
        lower, upper = config["coordinate_bounds"]
        specs.extend(
            SimpleNamespace(
                name=name,
                type="numeric",
                lower=float(lower),
                upper=float(upper),
                step=None,
                fixed=False,
                fixed_value=None,
            )
            for name in names
        )
    return specs


def _composition_encode_features(original: Any, *, data: Any, feature_columns: list[str], search_space: list[Any]) -> dict[str, Any]:
    config = _ACTIVE_CONFIG.get()
    if config is None:
        return original(data=data, feature_columns=feature_columns, search_space=search_space)
    if config["column"] not in feature_columns:
        raise ValueError("The composition formula column must be selected as an explanatory variable.")

    transformed, resolved = _composition_transformer(data, config)
    config.clear()
    config.update(resolved)
    transformed_columns: list[str] = []
    for column in feature_columns:
        if column == config["column"]:
            transformed_columns.extend(config["feature_names"])
        else:
            transformed_columns.append(column)
    filtered_specs = [spec for spec in search_space if getattr(spec, "name", None) != config["column"]]
    filtered_specs.extend(_coordinate_specs(config))
    encoded = original(
        data=transformed,
        feature_columns=transformed_columns,
        search_space=filtered_specs,
    )
    encoded["web_composition"] = dict(config)
    encoded["source_feature_columns"] = list(feature_columns)
    return encoded


def _composition_site(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "column": config["column"],
        "elements": config["elements"],
        "normalization": config["normalization"],
        "representation": config["representation"],
        "reference_element": config["reference_element"],
        "pseudocount": config["pseudocount"],
        "include_descriptors": False,
        "prefix": config["column"],
        "precision": config["precision"],
        "total": config["total"],
        "bounds": config["bounds"],
        "steps": config["steps"],
        "min_components": config["min_components"],
        "max_components": config["max_components"],
        "required_components": config["required_components"],
        "coordinate_bounds": config["coordinate_bounds"],
    }


def _replace_candidate_result(result: Any, *, candidates: Any, acq_value: Any) -> Any:
    try:
        return replace(result, candidates=candidates, acq_value=acq_value)
    except TypeError:
        result.candidates = candidates
        result.acq_value = acq_value
        return result


def _install_candidate_repair(optimizer: Any) -> None:
    if getattr(optimizer, "_web_composition_candidate_repair", False):
        return
    original = optimizer.candidate

    def candidate_with_repair(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if not kwargs.get("return_result"):
            return result
        from bochan.tabular.converter import dataframe_to_tensors

        raw_frame = self.candidates_to_dataframe(result.candidates)
        restored = self.inverse_compositions(raw_frame, repair=True, keep_coordinates=False)
        transformed = self.transform_compositions(restored)
        data_config = replace(
            self.data_config,
            input_cols=self.dataset.feature_names,
            target_cols=None,
        )
        repaired_x = dataframe_to_tensors(transformed, data_config).X
        repaired_acq = result.acq_value
        try:
            import torch

            with torch.no_grad():
                scores = result.acqf(repaired_x.unsqueeze(-2)).detach().reshape(-1)
            if scores.numel() == repaired_x.shape[0]:
                repaired_acq = scores
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
        return _replace_candidate_result(
            result,
            candidates=repaired_x,
            acq_value=repaired_acq,
        )

    optimizer.candidate = MethodType(candidate_with_repair, optimizer)
    optimizer._web_composition_candidate_repair = True


def _composition_fit_tabular_optimizer(original: Any, **kwargs: Any) -> Any:
    encoded_features = kwargs["encoded_features"]
    config = encoded_features.get("web_composition")
    if config is None:
        return original(**kwargs)

    from .feature_importance_outputs import relabel_feature_importance_outputs
    from .logging import current_request_id
    from .model_reuse import (
        current_model_reuse_state,
        register_fitted_model,
        reuse_fitted_tabular_optimizer,
    )
    from .tabular_backend import (
        _mutable_category_frame,
        categorical_feature_columns,
        categorical_target_columns,
        feature_category_maps,
        tabular_bounds,
        target_category_maps,
    )

    data = kwargs["data"]
    feature_columns = kwargs["feature_columns"]
    target_columns = kwargs["target_columns"]
    target_metadata = kwargs["target_metadata"]
    model_config = kwargs["model_config"]
    fit_config = kwargs["fit_config"]
    cross_validation = kwargs.get("cross_validation", False)
    cv_config = kwargs.get("cv_config")

    run_id = current_request_id()
    reuse_state = current_model_reuse_state()
    source_run_id = str((reuse_state or {}).get("source_run_id") or "")
    if source_run_id:
        if not run_id:
            raise RuntimeError("Model reuse requires an active Web request identifier.")
        return reuse_fitted_tabular_optimizer(
            source_run_id=source_run_id,
            current_run_id=run_id,
            data=data,
            feature_columns=feature_columns,
            target_columns=target_columns,
            target_metadata=target_metadata,
            hybrid_model=str(getattr(model_config, "task_type", "")) == "hybrid",
        )

    from bochan.tabular import TabularBayesianOptimizer

    categorical_features = [
        column
        for column in categorical_feature_columns(encoded_features)
        if column in data.columns and column != config["column"]
    ]
    categorical_targets = categorical_target_columns(target_metadata)
    fit_data = _mutable_category_frame(
        data,
        categorical_columns=[*categorical_features, *categorical_targets],
    )
    optimizer = TabularBayesianOptimizer(
        model_config=model_config,
        fit_config=fit_config,
        input_cols=feature_columns,
        target_cols=target_columns,
        categorical_cols=categorical_features,
        target_categorical_cols=categorical_targets,
        bounds=tabular_bounds(encoded_features),
        category_maps=feature_category_maps(data, encoded_features),
        target_category_maps=target_category_maps(target_metadata),
        encode_categories=True,
        return_original_categories=True,
        dropna=False,
        cross_validation=cross_validation,
        cv_config=cv_config,
        composition_sites={_SITE_NAME: _composition_site(config)},
        composition_element_constraints=config["element_constraints"],
        composition_constraint_rerank=True,
    )
    optimizer.fit(fit_data)
    if optimizer.dataset is None:
        raise RuntimeError("TabularBayesianOptimizer did not retain its fitted dataset.")
    _install_candidate_repair(optimizer)
    setattr(optimizer.bo, "_web_tabular_optimizer", optimizer)

    cross_validation_result = optimizer.cross_validation_result_
    if cross_validation_result is not None:
        importance = getattr(cross_validation_result, "feature_importance", None)
        if importance is not None:
            relabel_feature_importance_outputs(importance, target_columns)

    from .visualization_sessions import attach_fitted_tabular_optimizer

    if run_id:
        attach_fitted_tabular_optimizer(
            run_id,
            tabular_optimizer=optimizer,
            data=data,
            feature_columns=feature_columns,
            target_columns=target_columns,
            target_metadata=target_metadata,
            hybrid_model=str(getattr(model_config, "task_type", "")) == "hybrid",
        )
        register_fitted_model(run_id)
    return optimizer


def _element_constraint_results(tabular_optimizer: Any, restored: Any, row_index: Any) -> list[dict[str, Any]]:
    constraints = list(getattr(tabular_optimizer, "composition_element_constraints", ()) or ())
    if not constraints:
        return []
    raw, _totals = tabular_optimizer._row_native_values(restored, row_index)
    results: list[dict[str, Any]] = []
    for index, constraint in enumerate(constraints):
        lhs = 0.0
        for term in constraint["terms"]:
            site = term["site"]
            element = term["element"]
            config = tabular_optimizer.composition_sites[site]
            lhs += (
                float(term["coefficient"])
                * tabular_optimizer._basis_scale(config, element, constraint["basis"])
                * float(raw[(site, element)])
            )
        rhs = float(constraint["rhs"])
        operator = constraint["operator"]
        if operator == "=":
            violation = abs(lhs - rhs)
        elif operator == "<=":
            violation = max(lhs - rhs, 0.0)
        else:
            violation = max(rhs - lhs, 0.0)
        results.append(
            {
                "target": f"composition-constraint-{index + 1}",
                "goal": operator,
                "value": rhs,
                "predicted_mean": lhs,
                "ok": bool(violation <= 1e-7),
                "violation": float(violation),
            }
        )
    return results


def _composition_candidate_rows(original: Any, **kwargs: Any) -> list[dict[str, Any]]:
    rows = original(**kwargs)
    core_optimizer = kwargs["optimizer"]
    tabular_optimizer = getattr(core_optimizer, "_web_tabular_optimizer", None)
    if tabular_optimizer is None:
        return rows
    candidates = kwargs["candidates"]
    raw_frame = tabular_optimizer.candidates_to_dataframe(candidates)
    restored = tabular_optimizer.inverse_compositions(
        raw_frame,
        repair=True,
        keep_coordinates=False,
    )
    config = _ACTIVE_CONFIG.get() or {}
    formula_column = config.get("column")
    coordinate_columns = set(config.get("feature_names") or ())
    output_columns = [
        column
        for column in restored.columns
        if column == formula_column or "__fraction__" in str(column)
    ]
    for row_index, row in enumerate(rows):
        for column in coordinate_columns:
            row["values"].pop(column, None)
        for column in output_columns:
            value = restored.iloc[row_index][column]
            row["values"][str(column)] = value.item() if hasattr(value, "item") else value
        constraint_results = _element_constraint_results(
            tabular_optimizer,
            restored,
            restored.index[row_index],
        )
        row["constraints"].extend(constraint_results)
        row["constraints_ok"] = bool(row["constraints_ok"]) and all(
            result["ok"] for result in constraint_results
        )
    return rows


def install_composition_web_support() -> None:
    """Install composition-aware adapters before the Web app imports workflows."""

    global _INSTALLED
    if _INSTALLED:
        return
    from . import workflows_tabular

    original_encode = workflows_tabular._encode_features
    original_fit = workflows_tabular.fit_tabular_optimizer
    original_rows = workflows_tabular._candidate_rows
    original_workflow = workflows_tabular.run_regression_web_workflow

    def encode_adapter(*, data: Any, feature_columns: list[str], search_space: list[Any]) -> dict[str, Any]:
        return _composition_encode_features(
            original_encode,
            data=data,
            feature_columns=feature_columns,
            search_space=search_space,
        )

    def fit_adapter(**kwargs: Any) -> Any:
        return _composition_fit_tabular_optimizer(original_fit, **kwargs)

    def row_adapter(**kwargs: Any) -> list[dict[str, Any]]:
        return _composition_candidate_rows(original_rows, **kwargs)

    def workflow_adapter(request: Any, store: Any) -> dict[str, Any]:
        processing_request, config = _request_without_web_composition(request)
        if config is None:
            return original_workflow(request, store)
        token = _ACTIVE_CONFIG.set(config)
        try:
            result = original_workflow(processing_request, store)
            metadata = dict(result.get("metadata") or {})
            metadata["composition"] = {
                "column": config["column"],
                "elements": list(config.get("elements") or ()),
                "normalization": config["normalization"],
                "representation": config["representation"],
                "total": config["total"],
                "constraints": len(config["element_constraints"]),
            }
            result["metadata"] = metadata
            return result
        finally:
            _ACTIVE_CONFIG.reset(token)

    workflows_tabular._encode_features = encode_adapter
    workflows_tabular.fit_tabular_optimizer = fit_adapter
    workflows_tabular._candidate_rows = row_adapter
    workflows_tabular.run_regression_web_workflow = workflow_adapter
    _INSTALLED = True


def register_composition_routes(app: Any, *, api_prefix: str = "/api/v1") -> None:
    """Register a typed FastAPI endpoint for formula/config validation."""

    route_path = f"{api_prefix.rstrip('/')}/composition/validate"
    if any(getattr(route, "path", None) == route_path for route in app.routes):
        return
    from pydantic import BaseModel, ConfigDict, Field

    class CompositionValidationRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        formulas: list[str] = Field(min_length=1)
        settings: dict[str, Any]

    @app.post(route_path)
    def validate_composition(request: CompositionValidationRequest) -> dict[str, Any]:
        import pandas as pd

        from bochan.tabular.composition import format_formula, normalize_composition, parse_formula

        config = normalize_web_composition_settings(request.settings)
        frame = pd.DataFrame({config["column"]: request.formulas})
        _transformed, resolved = _composition_transformer(frame, config)
        elements = list(resolved["elements"])
        rows: list[dict[str, Any]] = []
        for formula in request.formulas:
            parsed = parse_formula(formula)
            normalized = normalize_composition(parsed, basis=resolved["normalization"])
            fractions = {element: float(normalized.get(element, 0.0)) for element in elements}
            rows.append(
                {
                    "input": formula,
                    "formula": format_formula(fractions, order=elements, precision=resolved["precision"]),
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


__all__ = [
    "install_composition_web_support",
    "normalize_web_composition_settings",
    "register_composition_routes",
]
