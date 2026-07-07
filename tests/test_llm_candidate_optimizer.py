from __future__ import annotations

import torch

from bochan.api import BayesianOptimizer, ModelConfig, OptimizeConfig, optimize_candidates
from bochan.llm import LLMSettings, plan_configs
from bochan.optim import optimize_acqf_llm_candidate_set


class SumAcquisition:
    def __call__(self, X):
        # X shape: batch x q x d. The LLM optimizer evaluates candidates with q=1.
        return X.squeeze(-2).sum(dim=-1)


def test_llm_candidate_optimizer_reranks_explicit_candidate_set():
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    candidate_set = torch.tensor(
        [
            [0.1, 0.1],
            [0.9, 0.8],
            [0.2, 0.7],
        ],
        dtype=torch.double,
    )

    candidates, values = optimize_acqf_llm_candidate_set(
        acq_function=SumAcquisition(),
        bounds=bounds,
        q=2,
        candidate_set=candidate_set,
        n_llm_candidates=3,
    )

    assert candidates.shape == torch.Size([2, 2])
    assert torch.allclose(candidates[0], torch.tensor([0.9, 0.8], dtype=torch.double))
    assert torch.all(values[:-1] >= values[1:])


def test_public_optimize_config_dispatches_llm_candidate_set():
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    config = OptimizeConfig(
        optimizer="llm_candidate_set",
        q=1,
        optimizer_kwargs={
            "candidate_set": [[0.1, 0.2], [0.8, 0.9]],
            "n_llm_candidates": 2,
        },
    )

    candidates, values = optimize_candidates(SumAcquisition(), bounds, config)

    assert candidates.shape == torch.Size([1, 2])
    assert torch.allclose(candidates[0], torch.tensor([0.8, 0.9], dtype=torch.double))
    assert values.shape == torch.Size([1])


def test_plan_configs_accepts_explicit_planner_response_without_provider_call():
    plan = plan_configs(
        goal="導電率を高くし、収縮率を低くしたい",
        mode="model_config",
        planner_response={
            "model_config": {"task_type": "regression", "model_type": "base"},
            "fit_config": {"method": "auto"},
            "warnings": ["weights were not specified"],
        },
    )

    assert plan["model_config"]["task_type"] == "regression"
    assert plan["fit_config"]["method"] == "auto"
    assert plan["warnings"] == ["weights were not specified"]
    assert "reasoning_summary" in plan


def test_model_config_llm_selected_uses_shared_settings_before_fit_without_provider_call():
    train_X = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.2]], dtype=torch.double)
    train_Y = torch.tensor([[0.0], [1.0], [0.4]], dtype=torch.double)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

    optimizer = BayesianOptimizer(
        model_config=ModelConfig(model_type="llm_selected"),
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
                "warnings": [],
            },
        ),
    )

    optimizer.fit(train_X, train_Y)

    assert optimizer.model_config.model_type == "base"
    assert optimizer.fit_config.skip_fit is True
    assert optimizer.llm_plan["model_config"]["model_type"] == "base"
    assert optimizer.bundle.metadata["llm_plan"]["warnings"] == []


def test_configure_llm_injects_shared_candidate_kwargs():
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    optimizer = BayesianOptimizer(model_config=ModelConfig(), bounds=bounds)
    optimizer.configure_llm(
        goal="large sum",
        candidate_set=[[0.1, 0.2], [0.8, 0.9]],
        n_llm_candidates=2,
    )

    config = OptimizeConfig(optimizer="llm_candidate_set", q=1)
    merged = optimizer._merge_llm_settings_into_opt_config(config)

    assert merged.optimizer_kwargs["goal"] == "large sum"
    assert merged.optimizer_kwargs["candidate_set"] == [[0.1, 0.2], [0.8, 0.9]]
    assert merged.optimizer_kwargs["n_llm_candidates"] == 2
