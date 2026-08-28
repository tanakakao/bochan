from __future__ import annotations

import torch

from bochan.api.configs import FitConfig, ModelBundle, ModelConfig, MultiOutputConfig
from bochan.api.modeling.fit import fit_model
from bochan.api.progress import emit_progress, progress_reporting


class _DummyModel:
    pass


def _recording_fit(model):
    assert isinstance(model, _DummyModel)
    return "fitted"


def _bundle(*, name: str = "base") -> ModelBundle:
    return ModelBundle(
        model=_DummyModel(),
        train_X=torch.zeros(4, 2, dtype=torch.double),
        train_Y=torch.zeros(4, 1, dtype=torch.double),
        model_config=ModelConfig(
            task_type="regression",
            model_type=name,
            outcome_transform=False,
        ),
        task_type="regression",
        model_type=name,
    )


def test_progress_reporting_is_request_local_and_optional():
    events: list[tuple[str, dict[str, object]]] = []

    emit_progress("outside", value=0)
    with progress_reporting(lambda event, payload: events.append((event, dict(payload)))):
        emit_progress("inside", value=1)
    emit_progress("outside_again", value=2)

    assert events == [("inside", {"value": 1})]


def test_fit_model_emits_single_fit_lifecycle():
    events: list[tuple[str, dict[str, object]]] = []
    fit_config = FitConfig(fit_func=_recording_fit)
    bundle = _bundle()

    with progress_reporting(lambda event, payload: events.append((event, dict(payload)))):
        result = fit_model(bundle, fit_config)

    assert result is bundle
    assert [event for event, _ in events] == [
        "model_fit_started",
        "model_fit_completed",
    ]
    assert events[0][1]["fit_mode"] == "single"
    assert events[0][1]["output_total"] == 1
    assert float(events[-1][1]["duration_ms"]) >= 0


def test_fit_model_emits_independent_output_progress_without_splitting_joint_models():
    events: list[tuple[str, dict[str, object]]] = []
    fit_config = FitConfig(fit_func=_recording_fit)
    sub_bundles = [_bundle(name="a"), _bundle(name="b")]
    multi_output = MultiOutputConfig(output_names=["strength", "hardness"])
    bundle = ModelBundle(
        model=_DummyModel(),
        train_X=torch.zeros(4, 2, dtype=torch.double),
        train_Y=torch.zeros(4, 2, dtype=torch.double),
        model_config=ModelConfig(
            task_type="multi_objective",
            model_type="base",
            outcome_transform=False,
            multi_output_config=multi_output,
        ),
        task_type="multi_objective",
        model_type="base",
        metadata={"sub_bundles": sub_bundles, "embedded_fit_configs": [None, None]},
    )

    with progress_reporting(lambda event, payload: events.append((event, dict(payload)))):
        fit_model(bundle, fit_config)

    output_started = [payload for event, payload in events if event == "model_output_fit_started"]
    output_completed = [payload for event, payload in events if event == "model_output_fit_completed"]
    assert [item["output_name"] for item in output_started] == ["strength", "hardness"]
    assert [item["output_index"] for item in output_started] == [1, 2]
    assert len(output_completed) == 2
    assert events[0][0] == "model_fit_started"
    assert events[0][1]["fit_mode"] == "independent"
    assert events[-1][0] == "model_fit_completed"

    joint_events: list[tuple[str, dict[str, object]]] = []
    joint_bundle = ModelBundle(
        model=_DummyModel(),
        train_X=torch.zeros(4, 2, dtype=torch.double),
        train_Y=torch.zeros(4, 2, dtype=torch.double),
        model_config=ModelConfig(
            task_type="multi_objective",
            model_type="multitask",
            outcome_transform=False,
            multi_output_config=None,
        ),
        task_type="multi_objective",
        model_type="multitask",
    )
    with progress_reporting(
        lambda event, payload: joint_events.append((event, dict(payload)))
    ):
        fit_model(joint_bundle, fit_config)

    assert [event for event, _ in joint_events] == [
        "model_fit_started",
        "model_fit_completed",
    ]
    assert joint_events[0][1]["fit_mode"] == "joint"
    assert joint_events[0][1]["output_total"] == 2
