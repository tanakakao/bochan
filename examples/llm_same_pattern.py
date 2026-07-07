"""LLM model selection and candidate generation with the same config pattern.

This example intentionally uses offline `planner_response` and `candidate_set` so it can run
without OpenAI / Gemini API keys. Replace these with `llm_config` when calling providers.
"""

from __future__ import annotations

import torch

from bochan.api import AcquisitionConfig, BayesianOptimizer, ModelConfig, ObjectiveConfig, OptimizeConfig


def main() -> None:
    train_X = torch.tensor(
        [
            [800.0, 2.0, 0.0],
            [850.0, 3.0, 1.0],
            [900.0, 4.0, 2.0],
            [830.0, 2.5, 1.0],
        ],
        dtype=torch.double,
    )
    train_Y = torch.tensor(
        [
            [10.2, 5.1],
            [12.4, 4.8],
            [11.0, 7.2],
            [11.8, 4.9],
        ],
        dtype=torch.double,
    )
    bounds = torch.tensor(
        [
            [700.0, 1.0, 0.0],
            [950.0, 5.0, 2.0],
        ],
        dtype=torch.double,
    )

    # Model selection follows the same style as candidate generation:
    # - model selection: ModelConfig(model_type="llm_selected", model_kwargs={...})
    # - candidate generation: OptimizeConfig(optimizer="llm_candidate_set", optimizer_kwargs={...})
    bo = BayesianOptimizer(
        model_config=ModelConfig(
            model_type="llm_selected",
            model_kwargs={
                "goal": "導電率を高くし、収縮率を低くしたい",
                # Offline test path. No provider call is made when planner_response is supplied.
                "planner_response": {
                    "model_config": {
                        "task_type": "regression",
                        "model_type": "base",
                        "outcome_transform": True,
                    },
                    "fit_config": {
                        "method": "auto",
                    },
                    "warnings": ["offline planner response for local verification"],
                    "reasoning_summary": "Use a base regression model for two continuous outputs.",
                },
            },
        ),
        bounds=bounds,
    )
    bo.fit(train_X, train_Y)

    print("resolved model_config:")
    print(bo.model_config)
    print("\nllm plan:")
    print(bo.llm_plan)

    acq_config = AcquisitionConfig(
        name="NEHVI",
        objective_config=ObjectiveConfig(
            mode="multi_output",
            outputs=[0, 1],
            directions=["maximize", "minimize"],
            weights=[1.0, 0.5],
        ),
    )
    opt_config = OptimizeConfig(
        optimizer="llm_candidate_set",
        q=2,
        raw_samples=4,
        optimizer_kwargs={
            # Offline test path. No provider call is made when candidate_set is supplied.
            "candidate_set": [
                [840.0, 3.0, 1.0],
                [860.0, 2.5, 2.0],
                [800.0, 4.0, 1.0],
                [920.0, 5.0, 2.0],
            ],
            "n_llm_candidates": 4,
        },
    )

    candidates, acq_value = bo.candidate(acq_config=acq_config, opt_config=opt_config)
    print("\nselected candidates:")
    print(candidates)
    print("\nacq_value:")
    print(acq_value)


if __name__ == "__main__":
    main()
