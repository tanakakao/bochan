"""LLM-assisted helpers for bochan."""

from .candidate_explainer_overall import (
    CandidateExplanation,
    CandidateExplanationConfig,
    CandidatePointExplanation,
    build_candidate_explanation_prompt,
    select_representative_candidates,
)
from .candidate_generator import (
    build_llm_candidate_set,
    sanitize_candidate_set,
    score_candidate_set,
    select_top_candidates,
)
from .client import BaseLLMClient, GeminiClient, LLMResponse, OpenAIClient, make_llm_client
from .configs import GoalConfig, LLMConfig, LLMContextConfig, LLMSettings
from .prompt_builder import build_candidate_prompt, build_goal_planner_prompt
from .structured_planner import build_config_planner_prompt, plan_configs


def _install_candidate_explanation_api() -> None:
    """Attach explanation methods when the public LLM package is imported."""

    from bochan.api.engine_defaults import BayesianOptimizer
    from bochan.api.llm_candidate_explanation import (
        install_bayesian_optimizer_candidate_explanation_api,
    )

    install_bayesian_optimizer_candidate_explanation_api(BayesianOptimizer)


_install_candidate_explanation_api()


__all__ = [
    "BaseLLMClient",
    "CandidateExplanation",
    "CandidateExplanationConfig",
    "CandidatePointExplanation",
    "GeminiClient",
    "GoalConfig",
    "LLMConfig",
    "LLMContextConfig",
    "LLMResponse",
    "LLMSettings",
    "OpenAIClient",
    "build_candidate_explanation_prompt",
    "build_candidate_prompt",
    "build_config_planner_prompt",
    "build_goal_planner_prompt",
    "build_llm_candidate_set",
    "make_llm_client",
    "plan_configs",
    "sanitize_candidate_set",
    "score_candidate_set",
    "select_representative_candidates",
    "select_top_candidates",
]
