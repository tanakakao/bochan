"""LLM-assisted helpers for bochan."""

from bochan.api.engine_defaults import BayesianOptimizer as _BayesianOptimizer
from bochan.api.llm_suggestion import (
    BayesianOptimizerSuggestion,
    install_bayesian_optimizer_llm_api,
)

from .candidate_generator import (
    build_llm_candidate_set,
    sanitize_candidate_set,
    score_candidate_set,
    select_top_candidates,
)
from .client import BaseLLMClient, GeminiClient, LLMResponse, OpenAIClient, make_llm_client
from .configs import GoalConfig, LLMConfig, LLMContextConfig, LLMSettings
from .planner import build_config_planner_prompt, plan_configs
from .prompt_builder import build_candidate_prompt, build_goal_planner_prompt

install_bayesian_optimizer_llm_api(_BayesianOptimizer)

__all__ = [
    "BaseLLMClient",
    "BayesianOptimizerSuggestion",
    "GeminiClient",
    "GoalConfig",
    "LLMConfig",
    "LLMContextConfig",
    "LLMResponse",
    "LLMSettings",
    "OpenAIClient",
    "build_candidate_prompt",
    "build_config_planner_prompt",
    "build_goal_planner_prompt",
    "build_llm_candidate_set",
    "install_bayesian_optimizer_llm_api",
    "make_llm_client",
    "plan_configs",
    "sanitize_candidate_set",
    "score_candidate_set",
    "select_top_candidates",
]
