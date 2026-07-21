"""Structured schemas and normalization for LLM configuration planning.

The base planner keeps a compact general-purpose prompt. This module extends the
public planner API with dedicated JSON structures for hybrid models, outcome
constraints, and candidate-side constraints while preserving the same provider
and offline ``planner_response`` behavior.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .client import make_llm_client
from .configs import coerce_llm_context
from .parser import parse_json_payload
from .planner import PlannerMode
from .planner import build_config_planner_prompt as _base_build_config_planner_prompt


def _hybrid_model_schema() -> dict[str, Any]:
    return {
        "description": (
            "Use when train_Y columns have different task types. output_configs "
            "must follow the exact train_Y column order."
        ),
        "model_config": {
            "task_type": "hybrid",
            "model_type": "base",
            "multi_output_config": {
                "output_names": [
                    "continuous_property",
                    "pass_probability",
                    "quality_rank",
                ],
                "output_configs": [
                    {
                        "name": "continuous_property",
                        "task_type": "regression",
                        "model_type": "base",
                        "model_kwargs": {},
                        "output_spec_kwargs": {},
                    },
                    {
                        "name": "pass_probability",
                        "task_type": "binary",
                        "model_type": "base",
                        "model_kwargs": {},
                        "output_spec_kwargs": {},
                    },
                    {
                        "name": "quality_rank",
                        "task_type": "ordinal",
                        "model_type": "base",
                        "model_kwargs": {"num_classes": 4},
                        "output_spec_kwargs": {},
                    },
                ],
                "use_hybrid": True,
                "fit_submodels": True,
                "fit_wrapper": False,
                "wrapper_kwargs": {},
                "output_spec_kwargs": [{}, {}, {}],
                "train_y_slice_dim": -1,
            },
        },
    }


def _outcome_constraint_schema() -> dict[str, Any]:
    return {
        "description": (
            "Use outcome_constraint_config for modeled target constraints. "
            "Prefer the structured constraints list when output names, class "
            "probabilities, or ordinal ranks are involved."
        ),
        "numeric_threshold_constraint": {
            "kind": "feasibility",
            "output": "shrinkage",
            "threshold": 0.3,
            "sense": "le",
            "margin": 0.0,
            "scale": 1.0,
        },
        "classification_probability_constraint": {
            "kind": "feasibility",
            "output": "pass_probability",
            "threshold": 0.8,
            "sense": "ge",
            "target_class": 1,
            "target_classes": None,
            "margin": 0.0,
            "scale": 1.0,
        },
        "ordinal_rank_probability_constraint": {
            "kind": "ordinal_rank",
            "output": "quality_rank",
            "rank": 2,
            "sense": "ge",
            "probability_threshold": 0.8,
            "scale": 1.0,
        },
        "outcome_constraint_config": {
            "constraints": [
                {
                    "kind": "feasibility",
                    "output": "shrinkage",
                    "threshold": 0.3,
                    "sense": "le",
                },
                {
                    "kind": "feasibility",
                    "output": "pass_probability",
                    "threshold": 0.8,
                    "sense": "ge",
                    "target_class": 1,
                },
                {
                    "kind": "ordinal_rank",
                    "output": "quality_rank",
                    "rank": 2,
                    "sense": "ge",
                    "probability_threshold": 0.8,
                },
            ],
            "output_indices": [],
            "operators": [],
            "thresholds": [],
            "eta": 0.001,
            "reduce_constraints": "prod",
            "reduce_q": "mean",
            "posterior_mode": "objective",
            "min_feasibility": 0.0,
            "detach_feasibility": False,
        },
    }


def _candidate_constraint_schema() -> dict[str, Any]:
    return {
        "description": (
            "Use OptimizeConfig constraints for candidate X. Linear constraints "
            "may use variable names or indices; the public planner normalizes known "
            "variable names to indices."
        ),
        "linear_constraint_entry": {
            "indices": ["raw material 1", "raw material 2", "raw material 3"],
            "coefficients": [1.0, 1.0, 1.0],
            "rhs": 1.0,
        },
        "optimize_config": {
            "optimizer": "optimize_acqf",
            "q": 3,
            "raw_samples": 256,
            "num_restarts": 10,
            "equality_constraints": [
                {
                    "indices": [
                        "raw material 1",
                        "raw material 2",
                        "raw material 3",
                    ],
                    "coefficients": [1.0, 1.0, 1.0],
                    "rhs": 1.0,
                }
            ],
            "inequality_constraints": [],
            "fixed_features": {},
            "repair_config": {
                "numeric_indices": [
                    "raw material 1",
                    "raw material 2",
                    "raw material 3",
                    "temperature",
                    "time",
                ],
                "steps": {
                    "raw material 1": 0.05,
                    "raw material 2": 0.05,
                    "raw material 3": 0.05,
                    "temperature": 10.0,
                },
                "comp_idx": [
                    "raw material 1",
                    "raw material 2",
                    "raw material 3",
                ],
                "k": 0,
                "equality_constraints": [
                    {
                        "indices": [
                            "raw material 1",
                            "raw material 2",
                            "raw material 3",
                        ],
                        "coefficients": [1.0, 1.0, 1.0],
                        "rhs": 1.0,
                    }
                ],
                "inequality_constraints": [],
                "inequality_sense": "le",
                "fixed_features": {"atmosphere": 1.0},
                "final_sum_constraint": {
                    "indices": [
                        "raw material 1",
                        "raw material 2",
                        "raw material 3",
                    ],
                    "rhs": 1.0,
                },
                "diversify": True,
                "diversify_kwargs": {},
                "score": "abs",
                "support_selection": "topk",
                "sample_tau": 0.2,
                "sample_eps": 0.05,
                "max_iters": 12,
                "num_alternations": 2,
                "final_priority": "grid",
                "support_eps": 0.0,
            },
        },
    }


def _augment_prompt_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    important_rules = list(payload.get("important_rules") or [])
    important_rules.extend(
        [
            "For heterogeneous outputs, return model_config.task_type='hybrid' and a complete multi_output_config.output_configs list in train_Y column order.",
            "Use outcome_constraint_config only for modeled target constraints; use OptimizeConfig equality/inequality constraints or repair_config for candidate-X constraints.",
            "Do not invent objective weights, feasibility thresholds, class indices, ordinal ranks, or composition totals. Preserve explicit values and add a warning when they are missing.",
            "Use serializable dictionaries only. Do not return Python callables for constraints, objectives, samplers, or repair functions.",
        ]
    )
    payload["important_rules"] = important_rules
    payload["dedicated_config_schemas"] = {
        "hybrid_model": _hybrid_model_schema(),
        "outcome_constraints": _outcome_constraint_schema(),
        "candidate_constraints_and_repair": _candidate_constraint_schema(),
    }
    payload["constraint_semantics"] = [
        "ObjectiveConfig defines which outputs are optimized and their maximize/minimize directions.",
        "OutcomeConstraintConfig defines feasibility conditions on modeled outputs, class probabilities, or ordinal rank probabilities.",
        "OptimizeConfig equality_constraints and inequality_constraints constrain candidate input X during acquisition optimization.",
        "CandidateRepairConfig rounds, fixes, sparsifies, diversifies, or repairs generated candidates after or during candidate search.",
        "The order of multi_output_config.output_configs must match the final dimension of train_Y exactly.",
    ]

    output_schema = dict(payload.get("output_schema") or {})
    model_schema = dict(output_schema.get("model_config") or {})
    model_schema["multi_output_config"] = _hybrid_model_schema()["model_config"][
        "multi_output_config"
    ]
    output_schema["model_config"] = model_schema

    acquisition_schema = dict(output_schema.get("acquisition_config") or {})
    acquisition_schema["outcome_constraint_config"] = _outcome_constraint_schema()[
        "outcome_constraint_config"
    ]
    output_schema["acquisition_config"] = acquisition_schema

    optimize_schema = dict(output_schema.get("optimize_config") or {})
    candidate_schema = _candidate_constraint_schema()["optimize_config"]
    optimize_schema["equality_constraints"] = candidate_schema["equality_constraints"]
    optimize_schema["inequality_constraints"] = candidate_schema[
        "inequality_constraints"
    ]
    optimize_schema["fixed_features"] = candidate_schema["fixed_features"]
    optimize_schema["repair_config"] = candidate_schema["repair_config"]
    output_schema["optimize_config"] = optimize_schema
    payload["output_schema"] = output_schema
    return payload


def build_config_planner_prompt(
    *,
    goal: Any,
    llm_context: Any | None = None,
    train_X: Any | None = None,
    train_Y: Any | None = None,
    bounds: Any | None = None,
    mode: PlannerMode = "full",
    study_summary: dict[str, Any] | None = None,
    requested_sections: Sequence[str] | None = None,
    section_prompts: Mapping[str, str] | None = None,
    existing_model_config: dict[str, Any] | None = None,
    existing_fit_config: dict[str, Any] | None = None,
    existing_acquisition_config: dict[str, Any] | None = None,
    existing_optimize_config: dict[str, Any] | None = None,
) -> str:
    """Build a planner prompt with dedicated hybrid and constraint schemas."""

    prompt = _base_build_config_planner_prompt(
        goal=goal,
        llm_context=llm_context,
        train_X=train_X,
        train_Y=train_Y,
        bounds=bounds,
        mode=mode,
        study_summary=study_summary,
        requested_sections=requested_sections,
        section_prompts=section_prompts,
        existing_model_config=existing_model_config,
        existing_fit_config=existing_fit_config,
        existing_acquisition_config=existing_acquisition_config,
        existing_optimize_config=existing_optimize_config,
    )
    heading, raw_payload = prompt.split("\n", maxsplit=1)
    payload = _augment_prompt_payload(json.loads(raw_payload))
    return heading + "\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def _index_lookup(names: Sequence[str] | None) -> dict[str, int]:
    return {str(name): index for index, name in enumerate(names or [])}


def _resolve_index(value: Any, lookup: Mapping[str, int]) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value)
    if text in lookup:
        return int(lookup[text])
    try:
        return int(text)
    except ValueError:
        return value


def _normalize_index_sequence(value: Any, lookup: Mapping[str, int]) -> Any:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return value
    return [_resolve_index(item, lookup) for item in value]


def _normalize_fixed_features(value: Any, lookup: Mapping[str, int]) -> Any:
    if not isinstance(value, Mapping):
        return value
    return {
        _resolve_index(key, lookup): item
        for key, item in value.items()
    }


def _normalize_steps(value: Any, lookup: Mapping[str, int]) -> Any:
    if not isinstance(value, Mapping):
        return value
    return {
        _resolve_index(key, lookup): item
        for key, item in value.items()
    }


def _normalize_linear_constraint_entry(
    value: Any,
    lookup: Mapping[str, int],
) -> Any:
    if isinstance(value, Mapping):
        indices = value.get("indices")
        coefficients = value.get("coefficients")
        rhs = value.get("rhs")
        if indices is None or coefficients is None or rhs is None:
            return dict(value)
        return (
            _normalize_index_sequence(indices, lookup),
            list(coefficients),
            rhs,
        )
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 3
    ):
        return (
            _normalize_index_sequence(value[0], lookup),
            list(value[1]),
            value[2],
        )
    return value


def _normalize_linear_constraints(
    value: Any,
    lookup: Mapping[str, int],
) -> Any:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return value
    return [
        _normalize_linear_constraint_entry(item, lookup)
        for item in value
    ]


def _normalize_final_sum_constraint(
    value: Any,
    lookup: Mapping[str, int],
) -> Any:
    if isinstance(value, Mapping):
        indices = value.get("indices")
        rhs = value.get("rhs")
        if indices is None or rhs is None:
            return dict(value)
        return (_normalize_index_sequence(indices, lookup), rhs)
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
    ):
        return (_normalize_index_sequence(value[0], lookup), value[1])
    return value


def _normalize_candidate_config(
    config: Mapping[str, Any],
    lookup: Mapping[str, int],
) -> dict[str, Any]:
    result = dict(config)
    for key in ("equality_constraints", "inequality_constraints"):
        if key in result:
            result[key] = _normalize_linear_constraints(result[key], lookup)
    if "fixed_features" in result:
        result["fixed_features"] = _normalize_fixed_features(
            result["fixed_features"],
            lookup,
        )

    repair = result.get("repair_config")
    if isinstance(repair, Mapping):
        normalized_repair = dict(repair)
        for key in ("numeric_indices", "comp_idx"):
            if key in normalized_repair:
                normalized_repair[key] = _normalize_index_sequence(
                    normalized_repair[key],
                    lookup,
                )
        if "steps" in normalized_repair:
            normalized_repair["steps"] = _normalize_steps(
                normalized_repair["steps"],
                lookup,
            )
        if "fixed_features" in normalized_repair:
            normalized_repair["fixed_features"] = _normalize_fixed_features(
                normalized_repair["fixed_features"],
                lookup,
            )
        for key in ("equality_constraints", "inequality_constraints"):
            if key in normalized_repair:
                normalized_repair[key] = _normalize_linear_constraints(
                    normalized_repair[key],
                    lookup,
                )
        if "final_sum_constraint" in normalized_repair:
            normalized_repair["final_sum_constraint"] = (
                _normalize_final_sum_constraint(
                    normalized_repair["final_sum_constraint"],
                    lookup,
                )
            )
        result["repair_config"] = normalized_repair
    return result


def _normalize_hybrid_model_config(
    config: Mapping[str, Any],
    target_names: Sequence[str] | None,
) -> dict[str, Any]:
    result = dict(config)
    multi_output = result.get("multi_output_config")
    if not isinstance(multi_output, Mapping):
        return result

    normalized_multi_output = dict(multi_output)
    raw_outputs = normalized_multi_output.get("output_configs")
    output_configs: list[Any] = []
    if isinstance(raw_outputs, Sequence) and not isinstance(raw_outputs, (str, bytes)):
        for index, raw in enumerate(raw_outputs):
            if not isinstance(raw, Mapping):
                output_configs.append(raw)
                continue
            output = dict(raw)
            if (
                not output.get("name")
                and target_names is not None
                and index < len(target_names)
            ):
                output["name"] = str(target_names[index])
            output_configs.append(output)
        normalized_multi_output["output_configs"] = output_configs

    if not normalized_multi_output.get("output_names") and target_names:
        normalized_multi_output["output_names"] = list(target_names)

    task_types = {
        str(output.get("task_type"))
        for output in output_configs
        if isinstance(output, Mapping) and output.get("task_type") is not None
    }
    if len(task_types) > 1:
        result["task_type"] = "hybrid"
        normalized_multi_output["use_hybrid"] = True
    elif str(result.get("task_type")) == "hybrid":
        normalized_multi_output.setdefault("use_hybrid", True)

    result["multi_output_config"] = normalized_multi_output
    return result


def _normalize_plan(
    payload: Mapping[str, Any],
    *,
    variable_names: Sequence[str] | None,
    target_names: Sequence[str] | None,
) -> dict[str, Any]:
    result = dict(payload)
    model_config = result.get("model_config")
    if isinstance(model_config, Mapping):
        result["model_config"] = _normalize_hybrid_model_config(
            model_config,
            target_names,
        )

    optimize_config = result.get("optimize_config") or result.get("opt_config")
    if isinstance(optimize_config, Mapping):
        normalized = _normalize_candidate_config(
            optimize_config,
            _index_lookup(variable_names),
        )
        if "optimize_config" in result:
            result["optimize_config"] = normalized
        else:
            result["opt_config"] = normalized
    return result


def plan_configs(
    *,
    goal: Any,
    llm_config: Any | None = None,
    llm_context: Any | None = None,
    train_X: Any | None = None,
    train_Y: Any | None = None,
    bounds: Any | None = None,
    mode: PlannerMode = "full",
    planner_response: Any | None = None,
    study_summary: dict[str, Any] | None = None,
    requested_sections: Sequence[str] | None = None,
    section_prompts: Mapping[str, str] | None = None,
    existing_model_config: dict[str, Any] | None = None,
    existing_fit_config: dict[str, Any] | None = None,
    existing_acquisition_config: dict[str, Any] | None = None,
    existing_optimize_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Infer and normalize configs using the dedicated structured schema."""

    context = coerce_llm_context(llm_context)
    if planner_response is not None:
        payload = parse_json_payload(planner_response)
    else:
        prompt = build_config_planner_prompt(
            goal=goal,
            llm_context=llm_context,
            train_X=train_X,
            train_Y=train_Y,
            bounds=bounds,
            mode=mode,
            study_summary=study_summary,
            requested_sections=requested_sections,
            section_prompts=section_prompts,
            existing_model_config=existing_model_config,
            existing_fit_config=existing_fit_config,
            existing_acquisition_config=existing_acquisition_config,
            existing_optimize_config=existing_optimize_config,
        )
        client = make_llm_client(llm_config)
        payload = parse_json_payload(client.generate_json(prompt).text)

    if not isinstance(payload, dict):
        raise ValueError("Planner response must be a JSON object.")
    normalized = _normalize_plan(
        payload,
        variable_names=context.variable_names,
        target_names=context.target_names,
    )
    normalized.setdefault("warnings", [])
    normalized.setdefault("reasoning_summary", "")
    return normalized


__all__ = ["build_config_planner_prompt", "plan_configs"]
