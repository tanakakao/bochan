"""Explicit composition helpers for the React/FastAPI tabular workflow.

The Web workflow calls these functions directly. No import-time function
replacement, ContextVar routing, or instance-method monkey patching is used.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np

from bochan.tabular.composition.constraints import CompositionElementConstraintResolver

_SITE_NAME = "composition"


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
        raise TypeError(
            "Composition element settings must be a sequence or comma-separated string."
        )
    return list(dict.fromkeys(item for item in values if item))


def normalize_web_composition_settings(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one Web composition configuration using canonical field names."""

    if not isinstance(raw, Mapping):
        raise TypeError("web_composition must be a mapping.")
    unknown = set(raw) - {
        "enabled",
        "column",
        "elements",
        "normalization",
        "representation",
        "reference_element",
        "pseudocount",
        "precision",
        "total",
        "bounds",
        "steps",
        "min_components",
        "max_components",
        "required_components",
        "coordinate_bounds",
        "element_constraints",
    }
    if unknown:
        raise KeyError(
            f"Unknown Web composition settings: {sorted(unknown)!r}."
        )

    column = str(raw.get("column") or "").strip()
    if not column:
        raise ValueError("web_composition.column is required.")

    representation = str(raw.get("representation", "ilr")).lower()
    if representation not in {"fractions", "clr", "alr", "ilr"}:
        raise ValueError(
            "Composition representation must be fractions, clr, alr, or ilr."
        )

    normalization = str(raw.get("normalization", "atomic_fraction")).lower()
    if normalization not in {"atomic_fraction", "weight_fraction"}:
        raise ValueError(
            "Composition normalization must be atomic_fraction or weight_fraction."
        )

    elements = _string_list(raw.get("elements"))
    required = _string_list(raw.get("required_components"))

    bounds: dict[str, tuple[float, float]] = {}
    for element, pair in dict(raw.get("bounds") or {}).items():
        values = tuple(pair)
        if len(values) != 2:
            raise ValueError(
                f"Composition bounds for {element!r} must have two values."
            )
        lower, upper = map(float, values)
        if (
            not np.isfinite(lower)
            or not np.isfinite(upper)
            or lower < 0
            or lower > upper
        ):
            raise ValueError(f"Invalid composition bounds for {element!r}.")
        bounds[str(element)] = (lower, upper)

    steps: dict[str, float] = {}
    for element, value in dict(raw.get("steps") or {}).items():
        step = float(value)
        if not np.isfinite(step) or step <= 0:
            raise ValueError(
                f"Composition step for {element!r} must be positive."
            )
        steps[str(element)] = step

    coordinate_pair = tuple(raw.get("coordinate_bounds") or (-8.0, 8.0))
    if len(coordinate_pair) != 2:
        raise ValueError("coordinate_bounds must have two values.")
    coordinate_lower, coordinate_upper = map(float, coordinate_pair)
    if (
        not np.isfinite(coordinate_lower)
        or not np.isfinite(coordinate_upper)
        or coordinate_lower >= coordinate_upper
    ):
        raise ValueError("coordinate_bounds must be finite and increasing.")

    total = _finite(raw.get("total", 1.0), 1.0)
    if total <= 0:
        raise ValueError("Composition total must be positive.")
    min_components = int(raw.get("min_components", 1))
    max_components_raw = raw.get("max_components")
    max_components = (
        None
        if max_components_raw in (None, "")
        else int(max_components_raw)
    )
    if min_components < 1:
        raise ValueError("min_components must be at least 1.")
    if max_components is not None and max_components < min_components:
        raise ValueError(
            "max_components must be greater than or equal to min_components."
        )

    constraints: list[dict[str, Any]] = []
    for index, item in enumerate(raw.get("element_constraints") or []):
        if not isinstance(item, Mapping):
            raise TypeError(
                f"Composition element constraint {index} must be a mapping."
            )
        terms: list[dict[str, Any]] = []
        for term_index, term in enumerate(item.get("terms") or []):
            if not isinstance(term, Mapping):
                raise TypeError(
                    f"Term {term_index} in composition constraint {index} "
                    "must be a mapping."
                )
            element = str(term.get("element") or "").strip()
            if not element:
                raise ValueError(
                    f"Term {term_index} in composition constraint {index} "
                    "requires an element."
                )
            coefficient = float(term.get("coefficient", 1.0))
            if not np.isfinite(coefficient):
                raise ValueError(
                    "Composition constraint coefficients must be finite."
                )
            terms.append(
                {
                    "site": _SITE_NAME,
                    "element": element,
                    "coefficient": coefficient,
                }
            )
        if not terms:
            raise ValueError(
                f"Composition element constraint {index} requires one or more terms."
            )
        operator = str(item.get("operator", "="))
        if operator not in {"=", "<=", ">="}:
            raise ValueError(
                "Composition constraint operator must be =, <=, or >=."
            )
        rhs = float(item.get("rhs", 0.0))
        if not np.isfinite(rhs):
            raise ValueError("Composition constraint rhs must be finite.")
        basis = str(item.get("basis", "atomic_amount"))
        if basis not in {"atomic_amount", "weight_amount"}:
            raise ValueError(
                "Composition constraint basis must be atomic_amount or weight_amount."
            )
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


def extract_web_composition_request(
    request: Any,
) -> tuple[Any, dict[str, Any] | None]:
    """Remove transport-only composition settings from a Web request."""

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


def _composition_transformer(
    data: Any,
    config: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    from bochan.composition import CompositionTransformer
    from bochan.tabular.composition.transformer import transform_composition_frame

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
    elements = list(transformer.fitted_elements)
    resolved = dict(config)
    resolved["elements"] = elements
    if resolved["max_components"] is None:
        resolved["max_components"] = len(elements)
    unknown = set(resolved["required_components"]) - set(elements)
    unknown.update(set(resolved["bounds"]) - set(elements))
    unknown.update(set(resolved["steps"]) - set(elements))
    for constraint in resolved["element_constraints"]:
        unknown.update(
            term["element"]
            for term in constraint["terms"]
            if term["element"] not in elements
        )
    if unknown:
        raise ValueError(
            f"Composition settings reference unknown elements: {sorted(unknown)!r}."
        )
    transformed = transform_composition_frame(
        transformer,
        data,
        column,
        drop_formula=True,
    )
    resolved["feature_names"] = list(transformer.feature_names_ or ())
    return transformed, resolved


def _coordinate_specs(config: dict[str, Any]) -> list[Any]:
    names = list(config["feature_names"])
    elements = list(config["elements"])
    specs: list[Any] = []
    if config["representation"] == "fractions":
        for index, name in enumerate(names):
            element = elements[index]
            lower, upper = config["bounds"].get(
                element,
                (0.0, config["total"]),
            )
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


def prepare_composition_encoded_features(
    *,
    data: Any,
    feature_columns: list[str],
    search_space: list[Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build Web search metadata in the transformed composition feature space."""

    from bochan.serving.workbench.workflow_utils import _encode_features

    if config["column"] not in feature_columns:
        raise ValueError(
            "The composition formula column must be selected as an explanatory variable."
        )

    transformed, resolved = _composition_transformer(data, config)
    transformed_columns: list[str] = []
    for column in feature_columns:
        if column == resolved["column"]:
            transformed_columns.extend(resolved["feature_names"])
        else:
            transformed_columns.append(column)
    filtered_specs = [
        spec
        for spec in search_space
        if getattr(spec, "name", None) != resolved["column"]
    ]
    filtered_specs.extend(_coordinate_specs(resolved))
    encoded = _encode_features(
        data=transformed,
        feature_columns=transformed_columns,
        search_space=filtered_specs,
    )
    encoded["web_composition"] = dict(resolved)
    encoded["source_feature_columns"] = list(feature_columns)
    return encoded, resolved


def composition_model_feature_columns(
    feature_columns: Sequence[str],
    config: Mapping[str, Any] | None,
) -> list[str]:
    """Return model-space feature names for Web linear-constraint resolution."""

    if config is None:
        return list(feature_columns)
    model_columns: list[str] = []
    for column in feature_columns:
        if column == config["column"]:
            model_columns.extend(config.get("feature_names") or ())
        else:
            model_columns.append(column)
    return model_columns


def composition_site(config: Mapping[str, Any]) -> dict[str, Any]:
    """Convert validated Web settings to one canonical composition site."""

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


def _composition_model_columns(optimizer: Any) -> list[str]:
    columns: list[str] = []
    for transformer in optimizer.composition.transformers.values():
        columns.extend(
            str(name)
            for name in (transformer.feature_names_ or ())
        )
    return list(dict.fromkeys(columns))


def _replace_candidate_result(
    result: Any,
    *,
    candidates: Any,
    acq_value: Any,
) -> Any:
    try:
        return replace(
            result,
            candidates=candidates,
            acq_value=acq_value,
        )
    except TypeError:
        result.candidates = candidates
        result.acq_value = acq_value
        return result


def repair_composition_candidate_result(
    optimizer: Any,
    result: Any,
) -> Any:
    """Repair candidate compositions and return them in the fitted model space."""

    from bochan.tabular.data import dataframe_to_tensors

    raw_frame = optimizer.candidates_to_dataframe(result.candidates)
    restored = optimizer.inverse_compositions(
        raw_frame,
        repair=True,
        keep_coordinates=False,
    )
    transform_source = restored.drop(
        columns=_composition_model_columns(optimizer),
        errors="ignore",
    )
    transformed = optimizer.transform_compositions(transform_source)
    data_config = replace(
        optimizer.data_config,
        input_cols=optimizer.dataset.feature_names,
        target_cols=None,
    )
    repaired_x = dataframe_to_tensors(transformed, data_config).X
    expected_dimension = int(result.candidates.shape[-1])
    if int(repaired_x.shape[-1]) != expected_dimension:
        raise RuntimeError(
            "Composition candidate repair changed the model feature dimension: "
            f"expected {expected_dimension}, got {int(repaired_x.shape[-1])}."
        )

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


def _element_constraint_results(
    tabular_optimizer: Any,
    restored: Any,
    row_index: Any,
) -> list[dict[str, Any]]:
    constraints = list(tabular_optimizer.candidates.element_constraints)
    if not constraints:
        return []
    projector = tabular_optimizer.candidates.projector()
    raw, _totals = projector.row_native_values(restored, row_index)
    results: list[dict[str, Any]] = []
    for index, constraint in enumerate(constraints):
        lhs = 0.0
        for term in constraint["terms"]:
            site = term["site"]
            element = term["element"]
            config = tabular_optimizer.composition.sites[site]
            lhs += (
                float(term["coefficient"])
                * CompositionElementConstraintResolver.basis_scale(
                    config,
                    element,
                    constraint["basis"],
                )
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


def add_composition_candidate_rows(
    rows: list[dict[str, Any]],
    *,
    tabular_optimizer: Any,
    candidates: Any,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Replace model coordinates in response rows with repaired composition values."""

    raw_frame = tabular_optimizer.candidates_to_dataframe(candidates)
    restored = tabular_optimizer.inverse_compositions(
        raw_frame,
        repair=True,
        keep_coordinates=False,
    )
    formula_column = config["column"]
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
            row["values"][str(column)] = (
                value.item() if hasattr(value, "item") else value
            )
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


def composition_response_metadata(
    config: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the stable composition metadata included in Web results."""

    if config is None:
        return None
    return {
        "column": config["column"],
        "elements": list(config.get("elements") or ()),
        "normalization": config["normalization"],
        "representation": config["representation"],
        "total": config["total"],
        "constraints": len(config["element_constraints"]),
    }


__all__ = [
    "_composition_transformer",
    "add_composition_candidate_rows",
    "composition_model_feature_columns",
    "composition_response_metadata",
    "composition_site",
    "extract_web_composition_request",
    "normalize_web_composition_settings",
    "prepare_composition_encoded_features",
    "repair_composition_candidate_result",
]