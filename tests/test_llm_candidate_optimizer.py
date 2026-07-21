from __future__ import annotations

import json

import torch

from bochan.api import (
    BayesianOptimizer,
    BochanStudy,
    ModelConfig,
    OptimizeConfig,
    StudySuggestion,
    optimize_candidates,
)
from bochan.api.llm_suggestion import BayesianOptimizerSuggestion
from bochan.llm import LLMSettings, build_config_planner_prompt, plan_configs
from bochan.optim import optimize_acqf_llm_candidate_set


class SumAcquisition:
    def __call__(self, X):
        # X shape: batch x q x d. The LLM optimizer evaluates candidates with q=1.
        return X.squeeze(-2).sum(dim=-1)


def _planner_payload(prompt: str) -> dict:
    return json.loads(prompt.split("\n", maxsplit=1)[1])


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


def test_study_level_prompt_requires_acquisition_and_optimizer_configs():
    prompt = build_config_planner_prompt(
        goal="導電率を高くし、収縮率を低くしたい",
        mode="model_config",
        study_summary={"n_completed": 12, "n_pending": 2, "n_failed": 1},
    )
    payload = _planner_payload(prompt)

    assert payload["requested_sections"] == ["model", "acquisition", "optimizer"]
    assert "Return model_config, fit_config, acquisition_config, and optimize_config." in payload[
        "instructions"
    ]
    assert any("acquisition_config is required" in item for item in payload["instructions"])
    assert "NEHVI" in payload["available_acquisition_choices"]["multi_objective_optimization"]
    assert "nsgaii" not in payload["available_acquisition_choices"]["multi_objective_optimization"]
    assert "nsgaii" in payload["available_optimizer_choices"]


def test_model_only_prompt_requests_only_model_and_fit_configs():
    prompt = build_config_planner_prompt(
        goal="yを大きくしたい",
        mode="model_config",
    )
    payload = _planner_payload(prompt)

    assert payload["requested_sections"] == ["model"]
    assert payload["instructions"][0] == "Return model_config and fit_config."
    assert any("Do not change acquisition_config" in item for item in payload["instructions"])


def test_section_specific_prompt_is_preserved_for_acquisition_planning():
    prompt = build_config_planner_prompt(
        goal="yを大きくしたい",
        mode="acquisition",
        requested_sections=["acquisition"],
        section_prompts={
            "acquisition": "初期探索なので活用より探索を強め、UCBを優先して検討する。",
        },
    )
    payload = _planner_payload(prompt)

    assert payload["requested_sections"] == ["acquisition"]
    assert payload["section_prompts"]["acquisition"].startswith("初期探索")
    assert payload["instructions"][0] == "Return acquisition_config."
    assert any("objective_config" in item for item in payload["instructions"])


def test_all_setting_prompt_accepts_independent_section_prompts():
    prompt = build_config_planner_prompt(
        goal="導電率を高くし、収縮率を低くしたい",
        mode="full",
        section_prompts={
            "model": "まずは単純で安定したモデルを選ぶ。",
            "acquisition": "実験ノイズを考慮する。",
            "optimizer": "q=3でカテゴリ変数を厳密に扱う。",
        },
    )
    payload = _planner_payload(prompt)

    assert payload["requested_sections"] == ["model", "acquisition", "optimizer"]
    assert set(payload["section_prompts"]) == {"model", "acquisition", "optimizer"}
    assert payload["section_prompts"]["optimizer"].startswith("q=3")


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


def test_bayesian_optimizer_suggest_all_applies_typed_defaults_without_provider_call():
    train_X = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.2]], dtype=torch.double)
    train_Y = torch.tensor([[0.0], [1.0], [0.4]], dtype=torch.double)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    optimizer = BayesianOptimizer(model_config=ModelConfig(), bounds=bounds)

    suggestion = optimizer.suggest_all(
        train_X=train_X,
        train_Y=train_Y,
        model_prompt="単純なGPを優先する。",
        acquisition_prompt="探索を少し強める。",
        optimizer_prompt="LLM候補集合を獲得関数で再順位付けする。",
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
            "warnings": [],
            "reasoning_summary": "Use a base GP, exploratory UCB, and candidate-set reranking.",
        },
        apply=True,
    )

    assert isinstance(suggestion, BayesianOptimizerSuggestion)
    assert optimizer.model_config.model_type == "base"
    assert optimizer.fit_config.skip_fit is True
    assert optimizer.acq_config.name == "UCB"
    assert optimizer.opt_config.optimizer == "llm_candidate_set"

    optimizer.fit(train_X, train_Y)
    candidates, values = optimizer.candidate()

    assert candidates.shape == torch.Size([1, 2])
    assert values.shape == torch.Size([1])


def test_bayesian_optimizer_can_apply_model_acquisition_and_optimizer_separately():
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    optimizer = BayesianOptimizer(model_config=ModelConfig(), bounds=bounds)

    model_suggestion = optimizer.suggest_model(
        "高次元ではないためbaseモデルを優先する。",
        planner_response={
            "model_config": {"task_type": "regression", "model_type": "base"},
            "fit_config": {"maxiter": 64},
        },
        apply=True,
    )
    acquisition_suggestion = optimizer.suggest_acquisition(
        "改善量を重視してEIを使う。",
        planner_response={
            "acquisition_config": {"name": "EI"},
        },
        apply=True,
    )
    optimizer_suggestion = optimizer.suggest_optimizer(
        "候補は2点、標準のoptimize_acqfを使う。",
        planner_response={
            "optimize_config": {
                "optimizer": "optimize_acqf",
                "q": 2,
                "raw_samples": 128,
                "num_restarts": 5,
            },
        },
        apply=True,
    )

    assert model_suggestion.mode == "model"
    assert acquisition_suggestion.mode == "acquisition"
    assert optimizer_suggestion.mode == "optimizer"
    assert optimizer.model_config.model_type == "base"
    assert optimizer.fit_config.maxiter == 64
    assert optimizer.acq_config.name == "EI"
    assert optimizer.opt_config.q == 2
    assert optimizer.opt_config.num_restarts == 5


def test_bochan_study_suggest_config_and_apply_without_provider_call():
    study = BochanStudy(
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        llm_settings=LLMSettings(
            goal="yを大きくしたい",
            planner_response={
                "model_config": {"task_type": "regression", "model_type": "base"},
                "fit_config": {"method": "auto", "skip_fit": True},
                "acquisition_config": {"name": "UCB", "acqf_kwargs": {"beta": 0.2}},
                "optimize_config": {"optimizer": "llm_candidate_set", "q": 1},
                "warnings": ["offline study suggestion"],
                "reasoning_summary": "Use a simple regression model and UCB for this smoke test.",
            },
        ),
    )
    study.add_observations(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        torch.tensor([[0.0], [1.0]], dtype=torch.double),
    )

    suggestion = study.suggest(mode="config")

    assert isinstance(suggestion, StudySuggestion)
    assert suggestion.model_config.model_type == "base"
    assert suggestion.fit_config.skip_fit is True
    assert suggestion.acq_config.name == "UCB"
    assert suggestion.opt_config.optimizer == "llm_candidate_set"
    assert suggestion.warnings == ["offline study suggestion"]

    study.apply_suggestion(suggestion)

    assert study.model_config.model_type == "base"
    assert study.fit_config.skip_fit is True
    assert study.acq_config.name == "UCB"
    assert study.opt_config.optimizer == "llm_candidate_set"
