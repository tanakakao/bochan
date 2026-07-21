"""LLM planner for bochan configuration selection.

The planner converts a natural-language goal and lightweight dataset metadata into
serializable bochan configuration dictionaries. It is intentionally separated from
candidate generation so applications can use either:

- plan only: return model / acquisition / optimizer settings for review
- full run: plan, fit, and generate candidates
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from .client import make_llm_client
from .configs import coerce_goal_config, coerce_llm_context
from .parser import parse_json_payload

PlannerMode = Literal[
    "model_config",
    "full",
    "model",
    "acquisition",
    "optimizer",
]
_SECTION_NAMES = ("model", "acquisition", "optimizer")


def _shape(value: Any) -> list[int] | None:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is not None:
        return [int(item) for item in shape]
    try:
        if isinstance(value, list):
            if value and isinstance(value[0], list):
                return [len(value), len(value[0])]
            return [len(value)]
    except TypeError:
        return None
    return None


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _normalize_section(value: Any) -> str:
    normalized = "".join(character for character in str(value).lower() if character.isalnum())
    if normalized in {"model", "modelconfig", "fit", "fitconfig"}:
        return "model"
    if normalized in {"acquisition", "acq", "acquisitionconfig", "acqconfig"}:
        return "acquisition"
    if normalized in {
        "optimizer",
        "optimization",
        "optimize",
        "optimizeconfig",
        "candidateoptimizer",
    }:
        return "optimizer"
    raise ValueError(f"Unknown planner section: {value!r}.")


def _resolve_requested_sections(
    *,
    mode: PlannerMode,
    study_level: bool,
    requested_sections: Sequence[str] | None,
) -> list[str]:
    if requested_sections is not None:
        normalized = [_normalize_section(section) for section in requested_sections]
        return list(dict.fromkeys(normalized))
    if study_level:
        return list(_SECTION_NAMES)
    if mode in {"model", "model_config"}:
        return ["model"]
    if mode == "acquisition":
        return ["acquisition"]
    if mode == "optimizer":
        return ["optimizer"]
    return list(_SECTION_NAMES)


def _normalize_section_prompts(
    section_prompts: Mapping[str, str] | None,
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in dict(section_prompts or {}).items():
        if value is None:
            continue
        if str(key).lower() == "overall":
            normalized["overall"] = str(value)
        else:
            normalized[_normalize_section(key)] = str(value)
    return normalized


def _section_instructions(
    *,
    requested_sections: Sequence[str],
    study_level: bool,
) -> list[str]:
    requested = set(requested_sections)
    if requested == set(_SECTION_NAMES):
        instructions = [
            "Return model_config, fit_config, acquisition_config, and optimize_config.",
            "acquisition_config is required unless the supplied information is insufficient; explain any omission in warnings.",
            "optimize_config is required unless the supplied information is insufficient; explain any omission in warnings.",
        ]
    elif requested == {"model"}:
        instructions = [
            "Return model_config and fit_config.",
            "Focus only on surrogate-model construction and fitting settings.",
            "Do not change acquisition_config or optimize_config.",
        ]
    elif requested == {"acquisition"}:
        instructions = [
            "Return acquisition_config.",
            "Include objective_config, acquisition kwargs, and outcome-constraint settings when required.",
            "Do not change model_config, fit_config, or optimize_config.",
        ]
    elif requested == {"optimizer"}:
        instructions = [
            "Return optimize_config.",
            "Include q, restarts, raw samples, backend, mixed-variable handling, and repair settings when required.",
            "Do not change model_config, fit_config, or acquisition_config.",
        ]
    else:
        names = ", ".join(requested_sections)
        instructions = [
            f"Return only the requested configuration sections: {names}.",
            "Do not replace configuration sections that were not requested.",
        ]

    if study_level:
        instructions.extend(
            [
                "This is a Study-level configuration suggestion.",
                "Use completed and pending trial information when deciding exploration versus exploitation.",
                "Do not automatically replace an explicit objective_config unless it conflicts with the stated goal.",
            ]
        )
    return instructions


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
    """Build the prompt used to infer bochan config dictionaries."""

    goal_config = coerce_goal_config(goal)
    if goal_config is None:
        raise ValueError("goal is required for LLM config planning.")
    context = coerce_llm_context(llm_context)
    is_study_level = study_summary is not None
    sections = _resolve_requested_sections(
        mode=mode,
        study_level=is_study_level,
        requested_sections=requested_sections,
    )
    prompts = _normalize_section_prompts(section_prompts)

    payload = {
        "role": "You are a bochan configuration planner for Bayesian optimization and active learning.",
        "mode": mode,
        "requested_sections": sections,
        "goal": goal_config.text,
        "section_prompts": prompts,
        "important_rules": [
            "Return JSON only.",
            "Do not include API keys or secrets.",
            "Treat section_prompts as user-provided requirements for the corresponding configuration section.",
            "Prefer simple, robust models unless the goal or data clearly requires a specialized model.",
            "Choose an acquisition function that is compatible with the task, objective count, and goal.",
            "Do not confuse the acquisition function with the candidate-optimization backend.",
            "Explicit existing configs should be preserved unless they conflict with the goal or a section-specific prompt.",
            "If uncertain, add warnings instead of overfitting the configuration.",
            "For Study-level suggestions, account for completed, pending, and failed trials.",
        ],
        "dataset_metadata": {
            "train_X_shape": _shape(train_X),
            "train_Y_shape": _shape(train_Y),
            "bounds": _to_jsonable(bounds),
            "variable_names": list(context.variable_names or []),
            "target_names": list(context.target_names or []),
            "variable_descriptions": dict(context.variable_descriptions),
            "target_descriptions": dict(context.target_descriptions),
            "domain_notes": list(context.domain_notes),
            "candidate_policy": context.candidate_policy,
        },
        "study_summary": study_summary or {},
        "available_model_config_choices": {
            "task_type": ["regression", "binary", "multiclass", "ordinal", "hybrid", "multi_objective"],
            "model_type": ["base", "mixed", "kronecker", "multitask", "saas", "deepgp", "deepkernel", "hetero"],
            "input_type": ["normal", "mixed"],
            "notes": [
                "Use task_type='regression' for continuous numeric targets.",
                "Use cat_dims when categorical variables are encoded as integer columns.",
                "Use multi_output_config when train_Y has multiple output columns and outputs should be modeled separately.",
                "Use model_type='base' unless there is a clear reason for a specialized model.",
            ],
        },
        "available_acquisition_choices": {
            "single_objective_optimization": ["EI", "UCB", "PI", "TS"],
            "multi_objective_optimization": ["NEHVI", "EHVI", "NParEGO"],
            "regression_active_learning": ["NIPV"],
            "classification_active_learning": ["entropy", "BALD", "margin", "variance"],
            "level_set_estimation": ["straddle", "ICU", "boundaryvariance", "levelset"],
        },
        "acquisition_selection_guidance": [
            "Use EI as a simple default for single-objective improvement-oriented optimization.",
            "Use UCB when stronger explicit exploration is useful; include beta in acqf_kwargs when appropriate.",
            "Use PI only when probability of improvement is specifically preferred over improvement magnitude.",
            "Use TS when posterior sampling is desired.",
            "Use NEHVI as the robust default for noisy multi-objective observations.",
            "Use EHVI when a multi-objective problem is effectively noise-free and the required baseline/reference data are available.",
            "Use NParEGO when scalarized multi-objective search is intentionally desired.",
            "Use entropy, BALD, margin, or variance for classification active learning rather than EI/UCB.",
            "Use straddle, ICU, boundaryvariance, or levelset when the goal is boundary or threshold estimation.",
            "Use optimizer='nsgaii' only as an optimization backend; it is not an acquisition function name.",
            "When objective directions or output indices are ambiguous, populate warnings instead of guessing silently.",
        ],
        "available_optimizer_choices": [
            "optimize_acqf",
            "llm_candidate_set",
            "torch",
            "evo",
            "nsgaii",
            "thompson_sampling",
        ],
        "existing_configs": {
            "model_config": existing_model_config or {},
            "fit_config": existing_fit_config or {},
            "acquisition_config": existing_acquisition_config or {},
            "optimize_config": existing_optimize_config or {},
        },
        "output_schema": {
            "model_config": {
                "task_type": "regression",
                "model_type": "base",
                "cat_dims": [],
                "input_transform_config": {"normalize": True},
                "outcome_transform": True,
            },
            "fit_config": {"method": "auto"},
            "acquisition_config": {
                "name": "EI or NEHVI or another compatible bochan acquisition alias",
                "acqf_kwargs": {},
                "objective_config": {
                    "mode": "auto or scalar or multi_output",
                    "outputs": ["target names or indices"],
                    "directions": ["maximize or minimize"],
                    "weights": [1.0],
                },
            },
            "optimize_config": {
                "optimizer": "optimize_acqf or llm_candidate_set or another backend",
                "q": 1,
                "raw_samples": 256,
                "num_restarts": 10,
            },
            "llm_context": {
                "variable_names": [],
                "target_names": [],
                "variable_descriptions": {},
                "target_descriptions": {},
                "domain_notes": [],
                "candidate_policy": None,
            },
            "warnings": [],
            "reasoning_summary": "short summary",
        },
        "instructions": _section_instructions(
            requested_sections=sections,
            study_level=is_study_level,
        ),
    }
    return "Plan bochan settings from the following JSON payload.\n" + json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


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
    """Infer serializable bochan configs with an LLM or explicit planner response.

    ``planner_response`` is useful for tests and offline workflows. When supplied,
    no LLM provider is called.
    """

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
    payload.setdefault("warnings", [])
    payload.setdefault("reasoning_summary", "")
    return dict(payload)
