"""BochanStudy LLM suggestion smoke test without provider calls.

This example uses `planner_response`, so it does not call OpenAI or Gemini.
It demonstrates Phase 1 and Phase 2:

1. BochanStudy carries shared `LLMSettings` into its internal BayesianOptimizer.
2. `study.suggest(mode="config")` proposes configs and `apply_suggestion()` adopts them.
"""

from __future__ import annotations

import torch

from bochan.api import BochanStudy, ModelConfig, OptimizeConfig
from bochan.llm import LLMSettings


def main() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    train_X = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.2]], dtype=torch.double)
    train_Y = torch.tensor([[0.0], [1.0], [0.4]], dtype=torch.double)

    study = BochanStudy(
        model_config=ModelConfig(model_type="llm_selected"),
        opt_config=OptimizeConfig(optimizer="llm_candidate_set", q=1),
        bounds=bounds,
        llm_settings=LLMSettings(
            goal="yを大きくしたい",
            planner_response={
                "model_config": {
                    "task_type": "regression",
                    "model_type": "base",
                    "outcome_transform": True,
                },
                "fit_config": {"method": "auto", "skip_fit": True},
                "acquisition_config": {"name": "UCB", "acqf_kwargs": {"beta": 0.2}},
                "optimize_config": {
                    "optimizer": "llm_candidate_set",
                    "q": 1,
                    "optimizer_kwargs": {
                        "candidate_set": [[0.2, 0.3], [0.8, 0.9]],
                        "n_llm_candidates": 2,
                    },
                },
                "warnings": ["offline study suggestion"],
                "reasoning_summary": "Use simple regression and UCB for this smoke test.",
            },
        ),
    )
    study.add_observations(train_X, train_Y)

    suggestion = study.suggest(mode="config")
    print("suggestion:")
    print(suggestion.to_dict())

    study.apply_suggestion(suggestion)
    print("\napplied configs:")
    print(study.model_config)
    print(study.fit_config)
    print(study.acq_config)
    print(study.opt_config)

    batch = study.ask(return_batch=True)
    print("\nnext candidate batch:")
    print(batch.candidates)
    print(batch.trial_ids)


if __name__ == "__main__":
    main()
