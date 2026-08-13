"""Canonical candidate-optimizer names shared by config and runtime layers."""

from __future__ import annotations

from typing import Literal

EvolutionaryMethod = Literal["ga", "pso", "sa", "cmaes"]
OptimizerName = Literal[
    "optimize_acqf",
    "evo",
    "ga",
    "pso",
    "sa",
    "cmaes",
    "torch",
    "nsgaii",
    "thompson_sampling",
    "llm_candidate_set",
]

_CANONICAL_OPTIMIZERS = {
    "optimize_acqf",
    "evo",
    "torch",
    "nsgaii",
    "thompson_sampling",
    "llm_candidate_set",
}
_EVOLUTIONARY_METHODS = {"ga", "pso", "sa", "cmaes"}
_MIXED_OPTIMIZERS = {
    "optimize_acqf_mixed",
    "evo_mixed",
    "optimize_acqf_evo_mixed",
    "torch_mixed",
    "optimize_acqf_torch_mixed",
    "thompson_sampling_mixed",
    "optimize_thompson_sampling_mixed",
}
_ALIASES = {
    "optimize_acqf_mixed": "optimize_acqf",
    "optimize_acqf_evo": "evo",
    "evo_mixed": "evo",
    "optimize_acqf_evo_mixed": "evo",
    "optimize_acqf_torch": "torch",
    "torch_mixed": "torch",
    "optimize_acqf_torch_mixed": "torch",
    "optimize_acqf_nsgaii": "nsgaii",
    "optimize_thompson_sampling": "thompson_sampling",
    "thompson_sampling_mixed": "thompson_sampling",
    "optimize_thompson_sampling_mixed": "thompson_sampling",
    "thompson": "thompson_sampling",
    "llm": "llm_candidate_set",
    "llm_candidate": "llm_candidate_set",
    "optimize_acqf_llm": "llm_candidate_set",
    "optimize_acqf_llm_candidate_set": "llm_candidate_set",
}


class _InternalMixedOptimizerName(str):
    """Mark a mixed backend selected internally from categorical dimensions."""


def _optimizer_name(optimizer: str) -> str:
    """Normalize an optimizer identifier for registry and dispatch lookup."""

    return optimizer.replace("-", "_").lower()
