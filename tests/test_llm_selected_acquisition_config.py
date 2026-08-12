from __future__ import annotations

from types import SimpleNamespace

import pytest

from bochan.api import AcquisitionConfig, ModelConfig, ObjectiveConfig
from bochan.api.llm_selected_acquisition import (
    resolve_llm_selected_acquisition,
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
        self.acq_config = None
        self.last_acquisition_suggestion = None
        self._llm_refit_required = False
        self.captured_prompt = None

    def _check_fitted(self):
        if self.bundle is None:
            raise RuntimeError("not fitted")

    def suggest_acquisition(self, *, prompt, apply=False):
        assert apply is False
        self.captured_prompt = prompt
        response = dict(self.llm_settings.planner_response or {})
        payload = response.get("acquisition_config")
        acq_config = AcquisitionConfig(**payload) if payload is not None else None
        return SimpleNamespace(
            acq_config=acq_config,
            reasoning_summary=response.get("reasoning_summary", ""),
        )


def test_llm_selected_acquisition_is_resolved_explicitly():
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

    resolved = resolve_llm_selected_acquisition(
        optimizer,
        AcquisitionConfig(
            name="llm_selected",
            objective_config=explicit_objective,
            acqf_kwargs={"prune_baseline": True},
        ),
    )

    assert resolved.name == "NEHVI"
    assert resolved.objective_config is explicit_objective
    assert resolved.acqf_kwargs["prune_baseline"] is True
    assert optimizer.acq_config.name == "NEHVI"
    assert optimizer.last_acquisition_suggestion.reasoning_summary.startswith(
        "Use NEHVI"
    )


def test_llm_selected_acquisition_includes_requested_config_in_prompt():
    optimizer = _DummyOptimizer(
        llm_settings=LLMSettings(
            goal="探索を強めたい",
            planner_response={"acquisition_config": {"name": "UCB"}},
        )
    )
    requested = AcquisitionConfig(
        name="llm_selected",
        objective_config=ObjectiveConfig(direction="maximize"),
    )

    resolved = resolve_llm_selected_acquisition(optimizer, requested)

    assert resolved.name == "UCB"
    assert '"name": "llm_selected"' in optimizer.captured_prompt
    assert '"direction": "maximize"' in optimizer.captured_prompt


def test_llm_selected_acquisition_requires_shared_llm_settings():
    optimizer = _DummyOptimizer(llm_settings=None)

    with pytest.raises(ValueError, match="requires llm_settings"):
        resolve_llm_selected_acquisition(
            optimizer,
            AcquisitionConfig(name="llm_selected"),
        )


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
        resolve_llm_selected_acquisition(
            optimizer,
            AcquisitionConfig(name="llm_selected"),
        )
