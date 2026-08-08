from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from botorch.acquisition.joint_entropy_search import qJointEntropySearch
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    qHypervolumeKnowledgeGradient,
)

import bochan.api.engine_defaults as engine_defaults
import bochan.api.information_acquisition_defaults as info_defaults
from bochan.api import (
    AcquisitionConfig,
    DataContext,
    ModelBundle,
    ModelConfig,
    MultiObjectiveConfig,
)
from bochan.api.acquisition_registry import available_acqf_names, resolve_acqf_cls


def _make_bundle(
    *,
    train_Y: torch.Tensor | None = None,
    task_type: str = "regression",
) -> ModelBundle:
    train_X = torch.tensor(
        [[0.0, 0.2], [0.5, 0.7], [1.0, 0.4]],
        dtype=torch.double,
    )
    if train_Y is None:
        train_Y = torch.tensor([[0.0], [1.0], [0.4]], dtype=torch.double)
    config = ModelConfig(
        task_type=task_type,
        model_type="base",
        outcome_transform=False,
    )
    return ModelBundle(
        model=SimpleNamespace(),
        train_X=train_X,
        train_Y=train_Y,
        model_config=config,
        task_type=task_type,
        model_type="base",
    )


@pytest.mark.parametrize(
    ("name", "expected", "multi_output"),
    [
        ("mes", qMaxValueEntropy, False),
        ("qmes", qMaxValueEntropy, False),
        ("jes", qJointEntropySearch, False),
        ("qjes", qJointEntropySearch, False),
        ("hvkg", qHypervolumeKnowledgeGradient, True),
        ("qhvkg", qHypervolumeKnowledgeGradient, True),
    ],
)
def test_information_aliases_resolve(name, expected, multi_output) -> None:
    resolved = resolve_acqf_cls(
        name,
        task_type="regression",
        multi_output=multi_output,
    )
    assert resolved is expected


@pytest.mark.parametrize("name", ["mes", "jes", "hvkg"])
def test_information_short_aliases_reject_classification(name: str) -> None:
    with pytest.raises(ValueError, match="regression / hybrid"):
        resolve_acqf_cls(
            name,
            task_type="binary",
            multi_output=True,
        )


def test_hvkg_short_alias_requires_multi_output() -> None:
    with pytest.raises(ValueError, match="multi-output"):
        resolve_acqf_cls(
            "hvkg",
            task_type="regression",
            multi_output=False,
        )


def test_information_canonical_names_resolve_directly() -> None:
    assert resolve_acqf_cls("qMaxValueEntropy") is qMaxValueEntropy
    assert resolve_acqf_cls("qJointEntropySearch") is qJointEntropySearch
    assert (
        resolve_acqf_cls("qHypervolumeKnowledgeGradient")
        is qHypervolumeKnowledgeGradient
    )


def test_information_aliases_are_listed() -> None:
    names = set(available_acqf_names())
    assert {"mes", "qmes", "jes", "qjes", "hvkg", "qhvkg"} <= names


def test_mes_generates_candidate_set_from_bounds(monkeypatch) -> None:
    bundle = _make_bundle()
    candidate_set = torch.tensor([[0.1, 0.2], [0.8, 0.9]], dtype=torch.double)
    captured = {}

    def fake_constructor(**kwargs):
        captured.update(kwargs)
        return {
            "model": kwargs["model"],
            "candidate_set": candidate_set,
            "maximize": kwargs["maximize"],
        }

    monkeypatch.setattr(
        info_defaults,
        "_get_botorch_input_constructor",
        lambda cls: fake_constructor,
    )

    resolved, _ = engine_defaults.resolve_acquisition_defaults(
        bundle,
        AcquisitionConfig(name="mes", acqf_cls=qMaxValueEntropy),
        DataContext(
            bounds=torch.tensor([[0.0, -1.0], [1.0, 2.0]], dtype=torch.double),
            extra={"mes_candidate_size": 321},
        ),
    )

    assert resolved.acqf_kwargs["candidate_set"] is candidate_set
    assert captured["bounds"] == [(0.0, 1.0), (-1.0, 2.0)]
    assert captured["candidate_size"] == 321
    assert captured["maximize"] is True


def test_mes_preserves_explicit_candidate_set(monkeypatch) -> None:
    bundle = _make_bundle()
    candidate_set = torch.tensor([[0.2, 0.3]], dtype=torch.double)
    monkeypatch.setattr(
        info_defaults,
        "_get_botorch_input_constructor",
        lambda cls: pytest.fail("MES constructor must not run"),
    )

    resolved, _ = info_defaults.resolve_information_acquisition_defaults(
        bundle,
        AcquisitionConfig(
            name="mes",
            acqf_cls=qMaxValueEntropy,
            acqf_kwargs={"candidate_set": candidate_set},
        ),
        DataContext(),
    )

    assert resolved.acqf_kwargs["candidate_set"] is candidate_set


def test_mes_multi_output_requires_posterior_transform() -> None:
    bundle = _make_bundle(
        train_Y=torch.tensor(
            [[0.0, 1.0], [1.0, 0.0], [0.4, 0.5]],
            dtype=torch.double,
        )
    )
    with pytest.raises(ValueError, match="posterior_transform"):
        info_defaults.resolve_information_acquisition_defaults(
            bundle,
            AcquisitionConfig(name="mes", acqf_cls=qMaxValueEntropy),
            DataContext(bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]])),
        )


def test_mes_rejects_mc_objective_configuration() -> None:
    bundle = _make_bundle()
    with pytest.raises(ValueError, match="does not consume"):
        info_defaults.resolve_information_acquisition_defaults(
            bundle,
            AcquisitionConfig(
                name="mes",
                acqf_cls=qMaxValueEntropy,
                objective=object(),
            ),
            DataContext(bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]])),
        )


def test_jes_generates_optimal_samples(monkeypatch) -> None:
    bundle = _make_bundle()
    optimal_inputs = torch.tensor([[[0.2, 0.3]]], dtype=torch.double)
    optimal_outputs = torch.tensor([[[1.2]]], dtype=torch.double)
    captured = {}

    def fake_constructor(**kwargs):
        captured.update(kwargs)
        return {
            "model": kwargs["model"],
            "optimal_inputs": optimal_inputs,
            "optimal_outputs": optimal_outputs,
            "condition_noiseless": kwargs["condition_noiseless"],
            "estimation_type": kwargs["estimation_type"],
            "num_samples": kwargs["num_samples"],
        }

    monkeypatch.setattr(
        info_defaults,
        "_get_botorch_input_constructor",
        lambda cls: fake_constructor,
    )

    resolved, _ = engine_defaults.resolve_acquisition_defaults(
        bundle,
        AcquisitionConfig(name="jes", acqf_cls=qJointEntropySearch),
        DataContext(
            bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
            extra={"jes_num_optima": 19},
        ),
    )

    assert resolved.acqf_kwargs["optimal_inputs"] is optimal_inputs
    assert resolved.acqf_kwargs["optimal_outputs"] is optimal_outputs
    assert captured["num_optima"] == 19
    assert captured["estimation_type"] == "LB"
    assert captured["num_samples"] == 64


def test_jes_requires_explicit_optimal_samples_as_pair() -> None:
    bundle = _make_bundle()
    with pytest.raises(ValueError, match="supplied together"):
        info_defaults.resolve_information_acquisition_defaults(
            bundle,
            AcquisitionConfig(
                name="jes",
                acqf_cls=qJointEntropySearch,
                acqf_kwargs={"optimal_inputs": torch.zeros(1, 1, 2)},
            ),
            DataContext(),
        )


def test_hvkg_generates_current_value_with_existing_ref_point(monkeypatch) -> None:
    bundle = _make_bundle(
        train_Y=torch.tensor(
            [[0.0, 1.0], [1.0, 0.0], [0.4, 0.5]],
            dtype=torch.double,
        )
    )
    ref_point = torch.tensor([-0.1, -0.2], dtype=torch.double)
    current_value = torch.tensor(0.7, dtype=torch.double)
    captured = {}

    def fake_constructor(**kwargs):
        captured.update(kwargs)
        return {
            "model": kwargs["model"],
            "ref_point": kwargs["ref_point"],
            "num_fantasies": kwargs["num_fantasies"],
            "num_pareto": kwargs["num_pareto"],
            "current_value": current_value,
            "objective": kwargs["objective"],
        }

    monkeypatch.setattr(
        info_defaults,
        "_get_botorch_input_constructor",
        lambda cls: fake_constructor,
    )

    resolved, context = engine_defaults.resolve_acquisition_defaults(
        bundle,
        AcquisitionConfig(name="hvkg", acqf_cls=qHypervolumeKnowledgeGradient),
        DataContext(
            bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
            ref_point=ref_point,
        ),
    )

    assert resolved.acqf_kwargs["current_value"] is current_value
    assert captured["ref_point"] is ref_point
    assert captured["objective_thresholds"] is None
    assert captured["num_fantasies"] == 8
    assert captured["num_pareto"] == 10
    assert context.ref_point is ref_point


def test_hvkg_disables_generic_multiobjective_scalarization(monkeypatch) -> None:
    bundle = _make_bundle(
        train_Y=torch.tensor(
            [[0.0, 1.0], [1.0, 0.0], [0.4, 0.5]],
            dtype=torch.double,
        )
    )
    ref_point = torch.tensor([-0.1, -0.2], dtype=torch.double)
    mo_config = MultiObjectiveConfig(
        ref_point=ref_point,
        scalarization_weights=torch.tensor([0.3, 0.7], dtype=torch.double),
        auto_scalarization=True,
    )

    def fake_constructor(**kwargs):
        assert kwargs["objective"] is None
        return {
            "model": kwargs["model"],
            "ref_point": kwargs["ref_point"],
            "current_value": torch.tensor(0.2),
        }

    monkeypatch.setattr(
        info_defaults,
        "_get_botorch_input_constructor",
        lambda cls: fake_constructor,
    )

    resolved, context = info_defaults.resolve_information_acquisition_defaults(
        bundle,
        AcquisitionConfig(name="hvkg", acqf_cls=qHypervolumeKnowledgeGradient),
        DataContext(
            bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
            multi_objective=mo_config,
        ),
    )

    assert resolved.objective is None
    assert context.multi_objective is not mo_config
    assert context.multi_objective.auto_scalarization is False
    assert mo_config.auto_scalarization is True


def test_hvkg_preserves_explicit_current_value(monkeypatch) -> None:
    bundle = _make_bundle(
        train_Y=torch.tensor(
            [[0.0, 1.0], [1.0, 0.0], [0.4, 0.5]],
            dtype=torch.double,
        )
    )
    current_value = torch.tensor(0.5, dtype=torch.double)
    monkeypatch.setattr(
        info_defaults,
        "_get_botorch_input_constructor",
        lambda cls: pytest.fail("HVKG constructor must not run"),
    )

    resolved, _ = info_defaults.resolve_information_acquisition_defaults(
        bundle,
        AcquisitionConfig(
            name="hvkg",
            acqf_cls=qHypervolumeKnowledgeGradient,
            acqf_kwargs={"current_value": current_value},
        ),
        DataContext(ref_point=torch.tensor([-0.1, -0.2], dtype=torch.double)),
    )

    assert resolved.acqf_kwargs["current_value"] is current_value


def test_hvkg_requires_multi_output() -> None:
    bundle = _make_bundle()
    with pytest.raises(ValueError, match="multi-output"):
        info_defaults.resolve_information_acquisition_defaults(
            bundle,
            AcquisitionConfig(name="hvkg", acqf_cls=qHypervolumeKnowledgeGradient),
            DataContext(),
        )
