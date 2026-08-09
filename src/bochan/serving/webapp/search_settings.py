"""Web search-constraint and candidate-optimizer helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


_SEARCH_METHODS = {
    "normal",
    "optimize_acqf",
    "torch",
    "optimize_acqf_torch",
    "ga",
    "sa",
    "pso",
    "cmaes",
    "evo",
    "optimize_acqf_evo",
    "thompson_sampling",
    "optimize_thompson_sampling",
    "nsgaii",
    "nsga2",
}

# Core GA intentionally uses a thorough 64 x 100 search. The Web workbench
# evaluates candidates synchronously and automatically routes derivative-free
# external estimators to GA, so use smaller interactive budgets here while
# leaving Core / Tabular optimizer defaults and explicitly selected PSO / SA /
# CMA-ES untouched.
_WEB_GA_OPTIONS: dict[str, int] = {
    "pop_size": 32,
    "num_generations": 40,
}
_NGBOOST_WEB_GA_OPTIONS: dict[str, int] = {
    "pop_size": 24,
    "num_generations": 24,
}
_NGBOOST_PERTURBED_WEB_GA_OPTIONS: dict[str, int] = {
    "pop_size": 16,
    "num_generations": 20,
}
_TABPFN_WEB_GA_OPTIONS: dict[str, int] = {
    "pop_size": 16,
    "num_generations": 20,
}
_TABPFN_PERTURBED_WEB_GA_OPTIONS: dict[str, int] = {
    "pop_size": 12,
    "num_generations": 16,
}


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if hasattr(value, "dict"):
        return dict(value.dict())
    return dict(vars(value))


def _interactive_ga_options() -> dict[str, int]:
    """Return the request-local GA budget for synchronous Web suggestion.

    NGBoost evaluates a complete boosted estimator for every ensemble member at
    every GA population row. TabPFN performs foundation-model inference for each
    estimator call. InputPerturbation multiplies evaluation rows by ``n_w`` again,
    so both expensive model families receive Web-only budgets while the public
    Core optimizer remains unchanged. Outside a Web request context this helper
    falls back to the ordinary interactive budget.
    """

    from .risk_settings import current_web_runtime_context

    runtime = current_web_runtime_context()
    model_type = str(runtime.get("model_type", "")).lower()
    perturbed = bool(runtime.get("input_perturbation", False))

    if model_type == "ngboost_ensemble":
        if perturbed:
            return dict(_NGBOOST_PERTURBED_WEB_GA_OPTIONS)
        return dict(_NGBOOST_WEB_GA_OPTIONS)
    if model_type == "tabpfn":
        if perturbed:
            return dict(_TABPFN_PERTURBED_WEB_GA_OPTIONS)
        return dict(_TABPFN_WEB_GA_OPTIONS)
    return dict(_WEB_GA_OPTIONS)


def normalize_feature_constraints(
    constraints: list[Any],
    *,
    feature_columns: list[str],
) -> list[SimpleNamespace]:
    """Normalize serializable Web constraints to the desktop constraint contract."""

    selected = set(feature_columns)
    normalized: list[SimpleNamespace] = []
    for index, raw_constraint in enumerate(constraints or []):
        constraint = _mapping(raw_constraint)
        if not bool(constraint.get("enabled", True)):
            continue
        sense = str(constraint.get("sense", "le")).lower()
        if sense not in {"le", "ge", "eq"}:
            raise ValueError(
                f"Constraint {index + 1}: sense must be le, ge, or eq."
            )
        raw_terms = list(constraint.get("terms") or [])
        if not raw_terms:
            raise ValueError(f"Constraint {index + 1}: at least one term is required.")
        terms: list[SimpleNamespace] = []
        for raw_term in raw_terms:
            term = _mapping(raw_term)
            column = str(term.get("column", ""))
            if column not in selected:
                raise ValueError(
                    f"Constraint {index + 1}: column is not a selected feature: {column}"
                )
            terms.append(
                SimpleNamespace(
                    column=column,
                    coefficient=float(term.get("coefficient", 1.0)),
                )
            )
        normalized.append(
            SimpleNamespace(
                name=str(constraint.get("name") or f"constraint-{index + 1}"),
                terms=terms,
                sense=sense,
                rhs=float(constraint.get("rhs", 0.0)),
                enabled=True,
            )
        )
    return normalized


def botorch_linear_constraints(
    constraints: list[Any],
    *,
    feature_columns: list[str],
) -> tuple[list[Any], list[Any]]:
    """Convert linear constraints to BoTorch's ``sum(a*x) >= rhs`` convention."""

    import torch

    feature_to_index = {
        name: index for index, name in enumerate(feature_columns)
    }
    equality_constraints: list[Any] = []
    inequality_constraints: list[Any] = []
    for constraint in constraints:
        if not bool(getattr(constraint, "enabled", True)):
            continue
        indices = torch.as_tensor(
            [feature_to_index[term.column] for term in constraint.terms],
            dtype=torch.long,
        )
        coefficients = torch.as_tensor(
            [float(term.coefficient) for term in constraint.terms],
            dtype=torch.double,
        )
        rhs = float(constraint.rhs)
        if constraint.sense == "eq":
            equality_constraints.append((indices, coefficients, rhs))
        elif constraint.sense == "ge":
            inequality_constraints.append((indices, coefficients, rhs))
        elif constraint.sense == "le":
            inequality_constraints.append((indices, -coefficients, -rhs))
        else:  # pragma: no cover - normalization rejects this first
            raise ValueError(f"Unknown constraint sense: {constraint.sense}")
    return equality_constraints, inequality_constraints


def resolve_search_method(
    name: str,
    *,
    multi_objective: bool,
) -> tuple[str, dict[str, Any], bool]:
    """Resolve a Web search-method name to interactive OptimizeConfig settings.

    Returns:
        ``(optimizer, optimizer_kwargs, use_acquisition_side_nsgaii)``.
    """

    method = str(name or "normal").replace("-", "_").lower()
    if method not in _SEARCH_METHODS:
        raise ValueError(f"Unknown search method: {method!r}.")
    if method in {"nsgaii", "nsga2"}:
        if not multi_objective:
            raise ValueError("NSGA-II is available only for multi-objective search.")
        return "optimize_acqf", {}, True
    if method in {"normal", "optimize_acqf"}:
        return "optimize_acqf", {}, False
    if method in {"torch", "optimize_acqf_torch"}:
        return "torch", {}, False
    if method in {"evo", "optimize_acqf_evo"}:
        return "evo", {"options": _interactive_ga_options()}, False
    if method == "ga":
        return "evo", {
            "method": "ga",
            "options": _interactive_ga_options(),
        }, False
    if method in {"sa", "pso", "cmaes"}:
        return "evo", {"method": method}, False
    if method in {"thompson_sampling", "optimize_thompson_sampling"}:
        return "thompson_sampling", {}, False
    raise ValueError(f"Unsupported search method: {method!r}.")


def build_target_constraint_config(
    request: Any,
    *,
    target_settings: list[dict[str, Any]],
    target_metadata: dict[str, dict[str, Any]],
    target_columns: list[str],
    directions: dict[str, str],
    hybrid_model: bool,
    exclude_optimized_boundaries: bool = False,
) -> Any | None:
    """Build target constraints, using class probabilities for classification.

    For level-set estimation, optimized above/below settings define the contour
    itself and must not also become feasibility constraints. Setting
    ``exclude_optimized_boundaries=True`` keeps constraint-only targets active
    while leaving optimized targets free to sample both sides of the boundary.
    """

    if all(bool(setting.get("legacy")) for setting in target_settings):
        if exclude_optimized_boundaries:
            return None
        from .target_settings import _build_outcome_constraint_config

        return _build_outcome_constraint_config(
            request,
            target_columns=target_columns,
            directions=directions,
        )

    from bochan.acquisition.feasible import FeasibilityConstraintSpec
    from bochan.api import OutcomeConstraintConfig

    specs: list[Any] = []
    for setting in target_settings:
        if exclude_optimized_boundaries and bool(setting.get("optimize", True)):
            continue
        goal = str(setting["goal"])
        if goal not in {"above", "below"}:
            continue
        target = str(setting["target"])
        meta = target_metadata[target]
        task = str(meta["internal_task"])
        output: Any = target if hybrid_model else target_columns.index(target)

        if task in {"binary", "multiclass"}:
            class_indices = [
                int(index) for index in meta.get("class_indices", [])
            ]
            kwargs: dict[str, Any] = {}
            if len(class_indices) == 1:
                kwargs["target_class"] = class_indices[0]
            else:
                kwargs["target_classes"] = class_indices
            specs.append(
                FeasibilityConstraintSpec(
                    output=output,
                    threshold=float(meta["configured_value"]),
                    sense="ge" if goal == "above" else "le",
                    **kwargs,
                )
            )
            continue

        raw_threshold = (
            float(meta["class_index"])
            if task == "ordinal"
            else float(meta["configured_value"])
        )
        direction = str(meta.get("direction", "maximize"))
        sign = -1.0 if direction == "minimize" else 1.0
        threshold = sign * raw_threshold
        if sign > 0:
            sense = "ge" if goal == "above" else "le"
        else:
            sense = "le" if goal == "above" else "ge"
        specs.append(
            FeasibilityConstraintSpec(
                output=output,
                threshold=threshold,
                sense=sense,
            )
        )
    return OutcomeConstraintConfig(constraints=specs) if specs else None


def feature_constraint_results(
    values: dict[str, Any],
    constraints: list[Any],
) -> list[dict[str, Any]]:
    """Evaluate decoded candidate values against explanatory-variable constraints."""

    results: list[dict[str, Any]] = []
    for constraint in constraints:
        if not bool(getattr(constraint, "enabled", True)):
            continue
        lhs = sum(
            float(values[term.column]) * float(term.coefficient)
            for term in constraint.terms
        )
        rhs = float(constraint.rhs)
        tolerance = 1e-8
        if constraint.sense == "le":
            ok = lhs <= rhs + tolerance
            violation = max(lhs - rhs, 0.0)
        elif constraint.sense == "ge":
            ok = lhs >= rhs - tolerance
            violation = max(rhs - lhs, 0.0)
        else:
            ok = abs(lhs - rhs) <= tolerance
            violation = abs(lhs - rhs)
        results.append(
            {
                "name": constraint.name,
                "type": "feature",
                "sense": constraint.sense,
                "lhs": float(lhs),
                "rhs": rhs,
                "ok": bool(ok),
                "violation": float(violation),
            }
        )
    return results


__all__ = [
    "botorch_linear_constraints",
    "build_target_constraint_config",
    "feature_constraint_results",
    "normalize_feature_constraints",
    "resolve_search_method",
]
