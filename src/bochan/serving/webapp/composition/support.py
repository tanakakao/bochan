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
from bochan.tabular.composition.variable_total import CompositionVariableTotalTransform

_SITE_NAME = "composition"
_DESCRIPTOR_BUILTINS = {"atomic_number", "atomic_weight"}
_DESCRIPTOR_STATISTICS = {"mean", "std", "min", "max", "range"}
_BEST_SUBSET_STRATEGIES = {"exact", "beam", "auto"}
_LOG_RATIO_REPRESENTATIONS = {"clr", "alr", "ilr"}


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


def _element_properties(value: Any) -> dict[str, dict[str, float]]:
    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("element_properties must be a mapping.")
    result: dict[str, dict[str, float]] = {}
    for property_name, raw_values in value.items():
        if not isinstance(raw_values, Mapping):
            raise TypeError(
                f"element_properties[{property_name!r}] must be a mapping."
            )
        values: dict[str, float] = {}
        for element, raw_value in raw_values.items():
            numeric = float(raw_value)
            if not np.isfinite(numeric):
                raise ValueError(
                    "Custom elemental-property values must be finite."
                )
            values[str(element)] = numeric
        result[str(property_name)] = values
    return result


def _descriptor_settings(raw: Mapping[str, Any]) -> dict[str, Any]:
    properties = _string_list(
        raw.get("descriptor_properties", ("atomic_number", "atomic_weight"))
    )
    statistics = _string_list(
        raw.get("descriptor_statistics", ("mean", "std", "min", "max", "range"))
    )
    unknown_statistics = set(statistics) - _DESCRIPTOR_STATISTICS
    if unknown_statistics:
        raise ValueError(
            f"Unknown descriptor statistics: {sorted(unknown_statistics)!r}."
        )

    custom = _element_properties(raw.get("element_properties"))
    unknown_properties = set(properties) - _DESCRIPTOR_BUILTINS - set(custom)
    if unknown_properties:
        raise KeyError(
            f"Unknown elemental properties: {sorted(unknown_properties)!r}."
        )

    include_descriptors = bool(raw.get("include_descriptors", False))
    include_num_elements = bool(
        raw.get("descriptor_include_num_elements", True)
    )
    include_mixing_entropy = bool(
        raw.get("descriptor_include_mixing_entropy", True)
    )
    if (
        include_descriptors
        and not properties
        and not include_num_elements
        and not include_mixing_entropy
    ):
        raise ValueError(
            "Composition descriptors are enabled but no descriptor features "
            "were selected."
        )
    return {
        "include_descriptors": include_descriptors,
        "descriptor_properties": properties,
        "descriptor_statistics": statistics,
        "descriptor_include_num_elements": include_num_elements,
        "descriptor_include_mixing_entropy": include_mixing_entropy,
        "element_properties": custom,
    }


def _positive_int(raw: Mapping[str, Any], key: str, default: int) -> int:
    value = int(raw.get(key, default))
    if value < 1:
        raise ValueError(f"{key} must be at least 1.")
    return value


def _best_subset_settings(raw: Mapping[str, Any]) -> dict[str, Any]:
    strategy = str(raw.get("best_subset_strategy", "auto")).lower()
    if strategy not in _BEST_SUBSET_STRATEGIES:
        raise ValueError(
            "best_subset_strategy must be exact, beam, or auto."
        )
    beam_steps = int(raw.get("best_subset_beam_steps", 4))
    if beam_steps < 0:
        raise ValueError("best_subset_beam_steps must be at least 0.")
    return {
        "best_subset_strategy": strategy,
        "best_subset_max_combinations": _positive_int(
            raw, "best_subset_max_combinations", 2000
        ),
        "best_subset_beam_width": _positive_int(
            raw, "best_subset_beam_width", 8
        ),
        "best_subset_beam_steps": beam_steps,
        "best_subset_max_evaluations": _positive_int(
            raw, "best_subset_max_evaluations", 200
        ),
    }


def _validate_best_subset_contract(config: Mapping[str, Any]) -> None:
    if config.get("support_selection") != "best_subset":
        return
    maximum = config.get("max_components")
    if maximum is None or int(config["min_components"]) != int(maximum):
        raise ValueError(
            "Composition best_subset requires min_components == max_components."
        )


def _uses_logratio_best_subset(config: Mapping[str, Any]) -> bool:
    return (
        config.get("support_selection") == "best_subset"
        and str(config.get("representation", "fractions")).lower()
        in _LOG_RATIO_REPRESENTATIONS
    )


def _total_settings(
    raw: Mapping[str, Any],
    *,
    column: str,
) -> dict[str, Any]:
    raw_total_bounds = raw.get("total_bounds")
    raw_total = raw.get(
        "total",
        None if raw_total_bounds is not None else 1.0,
    )
    if raw_total_bounds is None:
        if raw_total in (None, ""):
            raise ValueError("Fixed-total composition requires total.")
        total = float(raw_total)
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("Composition total must be finite and positive.")
        return {
            "variable_total": False,
            "total": total,
            "total_bounds": None,
            "total_feature": None,
        }

    if raw_total not in (None, ""):
        raise ValueError(
            "Composition settings must specify either total or total_bounds, not both."
        )
    pair = tuple(raw_total_bounds)
    if len(pair) != 2:
        raise ValueError("total_bounds must contain two values.")
    lower, upper = map(float, pair)
    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or lower <= 0.0
        or lower >= upper
    ):
        raise ValueError("total_bounds must be finite, positive, and increasing.")
    return {
        "variable_total": True,
        # A representative total is retained only for Web helper code. The
        # canonical composition site receives total_bounds, never both fields.
        "total": 0.5 * (lower + upper),
        "total_bounds": (lower, upper),
        "total_feature": f"{column}__total",
    }


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
        "total_bounds",
        "bounds",
        "steps",
        "min_components",
        "max_components",
        "required_components",
        "forbidden_components",
        "support_selection",
        "best_subset_strategy",
        "best_subset_max_combinations",
        "best_subset_beam_width",
        "best_subset_beam_steps",
        "best_subset_max_evaluations",
        "coordinate_bounds",
        "element_constraints",
        "include_descriptors",
        "descriptor_properties",
        "descriptor_statistics",
        "descriptor_include_num_elements",
        "descriptor_include_mixing_entropy",
        "element_properties",
    }
    if unknown:
        raise KeyError(
            f"Unknown Web composition settings: {sorted(unknown)!r}."
        )

    column = str(raw.get("column") or "").strip()
    if not column:
        raise ValueError("web_composition.column is required.")

    representation = str(raw.get("representation", "ilr")).lower()
    if representation == "fraction":
        representation = "fractions"
    if representation not in {"fractions", "clr", "alr", "ilr"}:
        raise ValueError(
            "Composition representation must be fractions, clr, alr, or ilr."
        )

    normalization = str(raw.get("normalization", "atomic_fraction")).lower()
    if normalization not in {"atomic_fraction", "weight_fraction"}:
        raise ValueError(
            "Composition normalization must be atomic_fraction or weight_fraction."
        )

    support_selection = str(raw.get("support_selection", "repair")).lower()
    if support_selection not in {"repair", "best_subset"}:
        raise ValueError(
            "Composition support_selection must be repair or best_subset."
        )

    elements = _string_list(raw.get("elements"))
    required = _string_list(raw.get("required_components"))
    forbidden = _string_list(raw.get("forbidden_components"))
    overlap = set(required) & set(forbidden)
    if overlap:
        raise ValueError(
            "Composition elements cannot be both required and forbidden: "
            f"{sorted(overlap)!r}."
        )

    total_settings = _total_settings(raw, column=column)
    total_limit = (
        float(total_settings["total_bounds"][1])
        if total_settings["variable_total"]
        else float(total_settings["total"])
    )

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
            or upper > total_limit + 1e-12
        ):
            raise ValueError(
                f"Invalid composition bounds for {element!r}; upper bounds must not "
                f"exceed the composition total limit {total_limit}."
            )
        bounds[str(element)] = (lower, upper)
    for element in forbidden:
        lower, _upper = bounds.get(element, (0.0, 0.0))
        if lower > 1e-12:
            raise ValueError(
                f"Forbidden composition element {element!r} cannot have a positive lower bound."
            )
        bounds[element] = (0.0, 0.0)

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

    config = {
        "enabled": bool(raw.get("enabled", True)),
        "column": column,
        "elements": elements,
        "normalization": normalization,
        "representation": representation,
        "reference_element": raw.get("reference_element") or None,
        "pseudocount": _finite(raw.get("pseudocount", 1e-12), 1e-12),
        "precision": max(1, int(raw.get("precision", 6))),
        **total_settings,
        "bounds": bounds,
        "steps": steps,
        "min_components": min_components,
        "max_components": max_components,
        "required_components": required,
        "forbidden_components": forbidden,
        "support_selection": support_selection,
        "coordinate_bounds": (coordinate_lower, coordinate_upper),
        "element_constraints": constraints,
        **_best_subset_settings(raw),
        **_descriptor_settings(raw),
    }
    if support_selection == "best_subset" and max_components is not None:
        _validate_best_subset_contract(config)
    return config


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
    # The Web optimizer keeps only true decision coordinates in the tabular
    # dataset. Descriptor features are appended later by a model InputTransform.
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
    unknown.update(set(resolved["forbidden_components"]) - set(elements))
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
    _validate_best_subset_contract(resolved)
    for property_name, property_values in resolved["element_properties"].items():
        missing = set(elements) - set(property_values)
        if missing and property_name in resolved["descriptor_properties"]:
            raise ValueError(
                f"Custom property {property_name!r} is missing elements "
                f"{sorted(missing)!r}."
            )
    transformed = transform_composition_frame(
        transformer,
        data,
        column,
        drop_formula=True,
    )
    if resolved["variable_total"]:
        transformed.loc[:, resolved["total_feature"]] = (
            CompositionVariableTotalTransform.formula_site_totals(
                data.loc[:, column],
                resolved,
            )
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
            if config["variable_total"]:
                lower, upper = 0.0, 1.0
            else:
                component_lower, component_upper = config["bounds"].get(
                    element,
                    (0.0, config["total"]),
                )
                lower = float(component_lower) / float(config["total"])
                upper = float(component_upper) / float(config["total"])
            specs.append(
                SimpleNamespace(
                    name=name,
                    type="numeric",
                    lower=float(lower),
                    upper=float(upper),
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
    if config["variable_total"]:
        lower, upper = config["total_bounds"]
        specs.append(
            SimpleNamespace(
                name=config["total_feature"],
                type="numeric",
                lower=float(lower),
                upper=float(upper),
                step=None,
                fixed=False,
                fixed_value=None,
            )
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
            if resolved["variable_total"]:
                transformed_columns.append(resolved["total_feature"])
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
    if _uses_logratio_best_subset(resolved):
        feature_to_index = {
            str(name): index
            for index, name in enumerate(encoded["feature_columns"])
        }
        encoded["postprocess_passthrough_indices"] = [
            feature_to_index[name]
            for name in resolved["feature_names"]
        ]
    return encoded, resolved


def composition_model_feature_columns(
    feature_columns: Sequence[str],
    config: Mapping[str, Any] | None,
) -> list[str]:
    """Return decision-space feature names for Web linear constraints."""

    if config is None:
        return list(feature_columns)
    model_columns: list[str] = []
    for column in feature_columns:
        if column == config["column"]:
            model_columns.extend(config.get("feature_names") or ())
            if config.get("variable_total"):
                model_columns.append(str(config["total_feature"]))
        else:
            model_columns.append(column)
    return model_columns


def composition_site(config: Mapping[str, Any]) -> dict[str, Any]:
    """Convert validated Web settings to one canonical composition site."""

    # Descriptors deliberately remain disabled in the CompositionAdapter.
    # Candidate optimization operates on composition coordinates only, while
    # the model InputTransform recomputes descriptors from every candidate.
    site = {
        "column": config["column"],
        "elements": config["elements"],
        "normalization": config["normalization"],
        "representation": config["representation"],
        "reference_element": config["reference_element"],
        "pseudocount": config["pseudocount"],
        "include_descriptors": False,
        "prefix": config["column"],
        "precision": config["precision"],
        "bounds": config["bounds"],
        "steps": config["steps"],
        "min_components": config["min_components"],
        "max_components": config["max_components"],
        "required_components": config["required_components"],
        "forbidden_components": config["forbidden_components"],
        "support_selection": config["support_selection"],
        "best_subset_strategy": config["best_subset_strategy"],
        "best_subset_max_combinations": config["best_subset_max_combinations"],
        "best_subset_beam_width": config["best_subset_beam_width"],
        "best_subset_beam_steps": config["best_subset_beam_steps"],
        "best_subset_max_evaluations": config["best_subset_max_evaluations"],
        "coordinate_bounds": config["coordinate_bounds"],
    }
    if config.get("variable_total"):
        site["total_bounds"] = config["total_bounds"]
        site["total_feature"] = config["total_feature"]
    else:
        site["total"] = config["total"]
    return site


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
        replaced = replace(
            result,
            candidates=candidates,
            acq_value=acq_value,
        )
    except TypeError:
        result.candidates = candidates
        result.acq_value = acq_value
        return result
    for attribute in ("raw_composition_candidates", "composition_raw_bridge"):
        if hasattr(result, attribute):
            setattr(replaced, attribute, getattr(result, attribute))
    return replaced


def repair_composition_candidate_result(
    optimizer: Any,
    result: Any,
) -> Any:
    """Repair candidate compositions and return them in the fitted decision space."""

    if (
        getattr(result, "raw_composition_candidates", None) is not None
        and getattr(result, "composition_raw_bridge", None) is not None
    ):
        # Raw best-subset optimization already enforces support, bounds, total,
        # required/forbidden elements, and compatible element constraints. Do
        # not infer support again from finite pseudocount log-ratio coordinates.
        return result

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


def _raw_best_subset_fractions(
    raw_candidates: Any,
    bridge: Any,
) -> tuple[np.ndarray, np.ndarray | None, Any]:
    """Return exact normalized fractions and optional variable totals from raw decisions."""

    if hasattr(bridge, "amount_values") and hasattr(bridge, "base"):
        amounts = (
            bridge.amount_values(raw_candidates)
            .detach()
            .cpu()
            .reshape(-1, len(bridge.amount_names))
            .numpy()
        )
        amounts[np.abs(amounts) < 1e-12] = 0.0
        totals = amounts.sum(axis=1)
        if np.any(totals <= 0.0):
            raise RuntimeError("Raw composition candidates must have positive totals.")
        fractions = amounts / totals[:, None]
        return fractions, totals, bridge.base

    fractions = (
        bridge.fraction_values(raw_candidates)
        .detach()
        .cpu()
        .reshape(-1, bridge.fraction_width)
        .numpy()
    )
    fractions[np.abs(fractions) < 1e-12] = 0.0
    totals = fractions.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise RuntimeError("Raw composition candidates must have positive totals.")
    return fractions / totals, None, bridge


def _restore_exact_raw_best_subset(
    restored: Any,
    *,
    tabular_optimizer: Any,
    candidate_result: Any,
    config: Mapping[str, Any],
) -> Any:
    raw_candidates = getattr(candidate_result, "raw_composition_candidates", None)
    bridge = getattr(candidate_result, "composition_raw_bridge", None)
    if raw_candidates is None or bridge is None:
        return restored

    from bochan.composition import format_formula

    fractions, variable_totals, fraction_bridge = _raw_best_subset_fractions(
        raw_candidates,
        bridge,
    )
    if int(fractions.shape[0]) != int(len(restored)):
        raise RuntimeError(
            "Raw composition candidate count does not match Web response rows."
        )

    transformer = next(
        (
            value
            for value in tabular_optimizer.composition.transformers.values()
            if tuple(
                f"{value.prefix}__fraction__{element}"
                for element in value.fitted_elements
            )
            == tuple(fraction_bridge.fraction_names)
        ),
        None,
    )
    if transformer is None:
        raise RuntimeError(
            "Unable to match the raw composition bridge to its fitted transformer."
        )
    atomic_fractions = transformer.basis_to_atomic_fractions(fractions)
    for index, name in enumerate(fraction_bridge.fraction_names):
        restored.loc[:, name] = fractions[:, index]
    if variable_totals is not None:
        total_feature = config.get("total_feature")
        if not total_feature:
            raise RuntimeError(
                "Variable-total raw candidate is missing its configured total feature."
            )
        restored.loc[:, total_feature] = variable_totals
    restored.loc[:, config["column"]] = [
        format_formula(
            dict(zip(fraction_bridge.elements, row, strict=True)),
            order=fraction_bridge.elements,
            precision=int(config["precision"]),
        )
        for row in atomic_fractions
    ]
    return restored


def add_composition_candidate_rows(
    rows: list[dict[str, Any]],
    *,
    tabular_optimizer: Any,
    candidates: Any,
    config: Mapping[str, Any],
    candidate_result: Any | None = None,
) -> list[dict[str, Any]]:
    """Replace decision coordinates in response rows with composition values."""

    raw_frame = tabular_optimizer.candidates_to_dataframe(candidates)
    has_raw_best_subset = (
        candidate_result is not None
        and getattr(candidate_result, "raw_composition_candidates", None) is not None
        and getattr(candidate_result, "composition_raw_bridge", None) is not None
    )
    restored = tabular_optimizer.inverse_compositions(
        raw_frame,
        repair=not has_raw_best_subset,
        keep_coordinates=False,
    )
    if has_raw_best_subset:
        restored = _restore_exact_raw_best_subset(
            restored,
            tabular_optimizer=tabular_optimizer,
            candidate_result=candidate_result,
            config=config,
        )
    formula_column = config["column"]
    total_feature = config.get("total_feature")
    coordinate_columns = set(config.get("feature_names") or ())
    output_columns = [
        column
        for column in restored.columns
        if (
            column == formula_column
            or "__fraction__" in str(column)
            or (total_feature is not None and column == total_feature)
        )
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
    variable_total = bool(config.get("variable_total", False))
    descriptor_metadata = {
        "enabled": bool(config.get("include_descriptors", False)),
        "properties": list(config.get("descriptor_properties") or ()),
        "statistics": list(config.get("descriptor_statistics") or ()),
        "include_num_elements": bool(
            config.get("descriptor_include_num_elements", True)
        ),
        "include_mixing_entropy": bool(
            config.get("descriptor_include_mixing_entropy", True)
        ),
        "feature_names": list(config.get("descriptor_feature_names") or ()),
        "derived_not_optimized": True,
    }
    return {
        "column": config["column"],
        "elements": list(config.get("elements") or ()),
        "normalization": config["normalization"],
        "representation": config["representation"],
        "variable_total": variable_total,
        "total": None if variable_total else config["total"],
        "total_bounds": (
            list(config["total_bounds"])
            if variable_total
            else None
        ),
        "total_feature": config.get("total_feature"),
        "constraints": len(config["element_constraints"]),
        "support_selection": config.get("support_selection", "repair"),
        "support_space": (
            "raw_amount"
            if config.get("support_selection") == "best_subset" and variable_total
            else "raw_fraction"
            if config.get("support_selection") == "best_subset"
            else None
        ),
        "required_components": list(config.get("required_components") or ()),
        "forbidden_components": list(config.get("forbidden_components") or ()),
        "best_subset_strategy": config.get("best_subset_strategy"),
        "descriptors": descriptor_metadata,
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
