"""LLM-assisted helpers for bochan.

Importing this package exports planning, candidate-generation and explanation
utilities only. Public ``BayesianOptimizer`` methods are composed explicitly by
the API layer rather than installed from this package at import time.
"""

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
