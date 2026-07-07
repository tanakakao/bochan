"""Prompt builders for LLM-assisted bochan workflows."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from .configs import GoalConfig, LLMContextConfig, coerce_goal_config, coerce_llm_context


def _to_list(value: Any) -> Any:
    """Tensor / numpy / scalar を JSON へ入れやすい形に変換する。"""

    if value is None:
        return None
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _bounds_to_variable_schema(bounds: Any, context: LLMContextConfig) -> list[dict[str, Any]]:
    bounds_list = _to_list(bounds)
    if bounds_list is None or len(bounds_list) != 2:
        raise ValueError("bounds must have shape [2, d] to build an LLM candidate prompt.")
    lower, upper = bounds_list
    d = len(lower)
    names = list(context.variable_names or [f"x{i}" for i in range(d)])
    if len(names) != d:
        raise ValueError(f"Expected {d} variable_names, got {len(names)}.")

    variables: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        variables.append(
            {
                "index": i,
                "name": str(name),
                "type": "continuous_or_encoded",
                "bounds": [lower[i], upper[i]],
                "description": context.variable_descriptions.get(str(name), ""),
            }
        )
    return variables


def _target_schema(context: LLMContextConfig) -> list[dict[str, Any]]:
    targets = []
    for i, name in enumerate(context.target_names or []):
        targets.append(
            {
                "index": i,
                "name": str(name),
                "description": context.target_descriptions.get(str(name), ""),
            }
        )
    for name, description in context.target_descriptions.items():
        if name not in {item["name"] for item in targets}:
            targets.append({"name": str(name), "description": description})
    return targets


def build_candidate_prompt(
    *,
    bounds: Any,
    n_candidates: int,
    llm_context: LLMContextConfig | dict[str, Any] | None = None,
    goal: GoalConfig | dict[str, Any] | str | None = None,
    acquisition_name: str | None = None,
    history_summary: dict[str, Any] | None = None,
    pending_candidates: Any | None = None,
) -> str:
    """LLM 候補集合生成用 prompt を作成する。

    Args:
        bounds: 探索範囲 ``[2, d]``。
        n_candidates: LLM に出力させる候補数。
        llm_context: 列名説明やドメインノート。
        goal: 自然言語の探索目的。明示 ObjectiveConfig がある場合でも補足文脈として使えます。
        acquisition_name: 最終的に再ランキングする acquisition 名。
        history_summary: 過去データ要約。未指定なら prompt では空扱い。
        pending_candidates: 評価中候補。
    """

    context = coerce_llm_context(llm_context)
    goal_config = coerce_goal_config(goal)
    variables = _bounds_to_variable_schema(bounds, context)
    payload = {
        "role": "You are an experimental candidate generator for Bayesian optimization.",
        "important_rules": [
            "You do not choose the final candidates.",
            "bochan will validate, repair, and rerank your candidates with an acquisition function.",
            "Return diverse and feasible candidate_set points, not a single best guess.",
            "Respect bounds exactly.",
            "Return JSON only.",
        ],
        "goal": None if goal_config is None else goal_config.text,
        "final_acquisition": acquisition_name,
        "n_candidates": int(n_candidates),
        "variables": variables,
        "targets": _target_schema(context),
        "domain_notes": list(context.domain_notes),
        "candidate_policy": context.candidate_policy,
        "history_summary": history_summary or {},
        "pending_candidates": _to_list(pending_candidates),
        "output_schema": {
            "candidates": [
                {
                    "x": ["value for variable index 0", "value for variable index 1"],
                    "reason": "brief reason",
                }
            ]
        },
    }

    return (
        "Generate a JSON candidate set for bochan. Follow the payload strictly.\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def build_goal_planner_prompt(
    *,
    goal: GoalConfig | dict[str, Any] | str,
    target_names: Sequence[str] | None = None,
    target_descriptions: dict[str, str] | None = None,
) -> str:
    """自然言語 goal から ObjectiveConfig 候補を作るための prompt を作成する。

    この helper は planner 実装のための下地です。現時点では candidate API の
    自動 ObjectiveConfig 生成までは行いません。
    """

    goal_config = coerce_goal_config(goal)
    if goal_config is None:
        raise ValueError("goal is required.")
    payload = {
        "role": "You convert a natural-language optimization goal into bochan configuration.",
        "goal": goal_config.text,
        "available_targets": [
            {"name": name, "description": (target_descriptions or {}).get(name, "")}
            for name in (target_names or [])
        ],
        "return_json_only": True,
        "output_schema": {
            "objective_config": {
                "mode": "scalar or multi_output",
                "outputs": ["target name"],
                "directions": ["maximize or minimize"],
                "weights": [1.0],
            },
            "acquisition_config": {"name": "EI or NEHVI or another alias"},
            "warnings": [],
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
