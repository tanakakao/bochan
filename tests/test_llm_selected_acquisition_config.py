from __future__ import annotations

import pytest

from bochan.api import AcquisitionConfig, ModelConfig, ObjectiveConfig, OptimizeConfig
from bochan.api.llm_selected_acquisition import (
    install_llm_selected_acquisition_api,
)
from bochan.llm import LLMSettings


class _DummyOptimizer:
    def __init__(self, *, llm_settings=None):
        self.model_config = ModelConfig()
        self.fit_config = None
        self.llm_settings = llm_settings
        self.train_X = [[0.0], [1.0]]
        self.train_Y = [[0.0], [1.0]]
        self.bounds = [[0.0], [1.0]]
        self.bundle = object()

    def _check_fitted(self):
        if self.bundle is None:
            raise RuntimeError("not fitted")

    def fit(self, *args, **kwargs):
        return self

    def acquisition(self, acq_config, *, data_context=None):
        return acq_config

    def candidate(
        self,
        acq_config,
        opt_config,
        *,
        data_context=None,
        bounds=None,
        return_result=False,
    ):
        return acq_config, opt_config


install_llm_selected_acquisition_api(_DummyOptimizer)


def test_llm_selected_acquisition_is_resolved_at_candidate_time():
    explicit_objective = ObjectiveConfig(
        mode="multi_output",
        outputs=[0, 1],
        directions=["maximize", "minimize"],
        weights=[1.0, 0.5],
    )
    optimizer = _DummyOptimizer(
        llm_settings=LLMSettings(
            goal="propertyを高くし、property2を低くしたい",
            planner_response={
                "acquisition_config": {
                    "name": "NEHVI",
                    "objective_config": {
                        "mode": "multi_output",
                        "outputs": [0, 1],
                        "directions": ["minimize", "maximize"],
                    },
                    "acqf_kwargs": {"prune_baseline": False},
                },
                "reasoning_summary": (
                    "Use NEHVI for noisy multi-objective optimization."
                ),
            },
        )
    )

    resolved_acq, resolved_opt = optimizer.candidate(
        acq_config=AcquisitionConfig(
            name="llm_selected",
            objective_config=explicit_objective,
            acqf_kwargs={"prune_baseline": True},
        ),
        opt_config=OptimizeConfig(optimizer="llm_candidate_set", q=3),
    )

    assert resolved_acq.name == "NEHVI"
    assert resolved_acq.objective_config is explicit_objective
    assert resolved_acq.acqf_kwargs["prune_baseline"] is True
    assert resolved_opt.optimizer == "llm_candidate_set"
    assert optimizer.acq_config.name == "NEHVI"
    assert optimizer.last_acquisition_suggestion.reasoning_summary.startswith(
        "Use NEHVI"
    )


def test_llm_selected_acquisition_includes_requested_config_in_prompt(monkeypatch):
    captured = {}

    def fake_plan_configs(**kwargs):
        captured.update(kwargs)
        return {"acquisition_config": {"name": "UCB"}}

    monkeypatch.setattr("bochan.llm.plan_configs", fake_plan_configs)
    optimizer = _DummyOptimizer(
        llm_settings=LLMSettings(goal="探索を強めたい")
    )
    requested = AcquisitionConfig(
        name="llm_selected",
        objective_config=ObjectiveConfig(direction="maximize"),
    )

    resolved = optimizer.acquisition(requested)

    assert resolved.name == "UCB"
    assert captured["requested_sections"] == ["acquisition"]
    prompt = captured["section_prompts"]["acquisition"]
    assert '"name": "llm_selected"' in prompt
    assert '"direction": "maximize"' in prompt


def test_llm_selected_acquisition_requires_shared_llm_settings():
    optimizer = _DummyOptimizer(llm_settings=None)

    with pytest.raises(ValueError, match="requires llm_settings"):
        optimizer.acquisition(AcquisitionConfig(name="llm_selected"))


def test_llm_selected_acquisition_rejects_recursive_selector_response():
    optimizer = _DummyOptimizer(
        llm_settings=LLMSettings(
            goal="獲得関数を選ぶ",
            planner_response={
                "acquisition_config": {"name": "llm_selected"},
            },
        )
    )

    with pytest.raises(ValueError, match="concrete acquisition name"):
        optimizer.acquisition(AcquisitionConfig(name="llm_selected"))
