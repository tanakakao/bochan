from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from botorch.acquisition.multi_objective.joint_entropy_search import (
    qLowerBoundMultiObjectiveJointEntropySearch,
)
from botorch.acquisition.multi_objective.max_value_entropy_search import (
    qLowerBoundMultiObjectiveMaxValueEntropySearch,
)

import bochan.api.information_acquisition_defaults as info_defaults
from bochan.api import (
    AcquisitionConfig,
    DataContext,
    ModelBundle,
    ModelConfig,
    MultiObjectiveConfig,
    OptimizeConfig,
)
from bochan.api.acquisition import defaults as engine_defaults
from bochan.api.registry.acquisition import available_acqf_names, resolve_acqf_cls


def _make_bundle(
    *,
    task_type: str = "regression",
    input_type: str = "normal",
    cat_dims: list[int] | None = None,
) -> ModelBundle:
    train_X = torch.tensor(
        [[0.0, 0.2], [0.5, 0.7], [1.0, 0.4]],
        dtype=torch.double,
    )
    train_Y = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [0.4, 0.5]],
        dtype=torch.double,
    )
    config = ModelConfig(
        task_type=task_type,
        model_type="base",
        input_type=input_type,
        cat_dims=cat_dims,
        outcome_transform=False,
    )
    return ModelBundle(
        model=SimpleNamespace(),
        train_X=train_X,
        train_Y=train_Y,
        model_config=config,
        input_type=input_type,
        task_type=task_type,
        model_type="base",
        cat_dims=list(cat_dims or []),
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("mo_mes", qLowerBoundMultiObjectiveMaxValueEntropySearch),
        ("qmo_mes", qLowerBoundMultiObjectiveMaxValueEntropySearch),
        ("mesmo", qLowerBoundMultiObjectiveMaxValueEntropySearch),
        ("qmesmo", qLowerBoundMultiObjectiveMaxValueEntropySearch),
        ("multi_objective_mes", qLowerBoundMultiObjectiveMaxValueEntropySearch),
        ("mo_jes", qLowerBoundMultiObjectiveJointEntropySearch),
        ("qmo_jes", qLowerBoundMultiObjectiveJointEntropySearch),
        ("multi_objective_jes", qLowerBoundMultiObjectiveJointEntropySearch),
    ],
)
def test_multiobjective_entropy_aliases_resolve(name: str, expected: type) -> None:
    assert (
        resolve_acqf_cls(
            name,
            task_type="regression",
            multi_output=True,
        )
        is expected
    )


def test_multiobjective_entropy_canonical_names_resolve_directly() -> None:
    assert (
        resolve_acqf_cls("qLowerBoundMultiObjectiveMaxValueEntropySearch")
        is qLowerBoundMultiObjectiveMaxValueEntropySearch
    )
    assert (
        resolve_acqf_cls("qLowerBoundMultiObjectiveJointEntropySearch")
        is qLowerBoundMultiObjectiveJointEntropySearch
    )


@pytest.mark.parametrize("name", ["mo_mes", "mesmo", "mo_jes"])
def test_multiobjective_entropy_aliases_require_multi_output(name: str) -> None:
    with pytest.raises(ValueError, match="multi-output"):
        resolve_acqf_cls(
            name,
            task_type="regression",
            multi_output=False,
        )


@pytest.mark.parametrize("name", ["mo_mes", "mo_jes"])
def test_multiobjective_entropy_aliases_reject_classification(name: str) -> None:
    with pytest.raises(ValueError, match="regression / hybrid"):
        resolve_acqf_cls(
            name,
            task_type="binary",
            multi_output=True,
        )


def test_multiobjective_entropy_aliases_are_listed() -> None:
    names = set(available_acqf_names())
    assert {
        "momes",
        "qmomes",
        "mesmo",
        "qmesmo",
        "mojes",
        "qmojes",
    } <= names


def test_mo_mes_generates_hypercell_bounds(monkeypatch) -> None:
    bundle = _make_bundle()
    pareto_sets = torch.zeros(3, 4, 2, dtype=torch.double)
    pareto_fronts = torch.ones(3, 4, 2, dtype=torch.double)
    hypercell_bounds = torch.full((3, 2, 5, 2), 2.0, dtype=torch.double)
    captured = {}

    def fake_sample(**kwargs):
        captured["sample"] = kwargs
        return pareto_sets, pareto_fronts

    def fake_box(fronts, *, maximize):
        captured["box"] = {"fronts": fronts, "maximize": maximize}
        return hypercell_bounds

    optimizer = object()
    monkeypatch.setattr(
        info_defaults,
        "_sample_multiobjective_optimal_points",
        fake_sample,
    )
    monkeypatch.setattr(
        info_defaults,
        "_compute_multiobjective_hypercell_bounds",
        fake_box,
    )

    resolved, _ = engine_defaults.resolve_acquisition_defaults(
        bundle,
        AcquisitionConfig(
            name="mo_mes",
            acqf_cls=qLowerBoundMultiObjectiveMaxValueEntropySearch,
        ),
        DataContext(
            bounds=torch.tensor([[0.0, -1.0], [1.0, 2.0]], dtype=torch.double),
            extra={
                "mo_entropy_num_pareto_samples": 3,
                "mo_entropy_num_pareto_points": 4,
                "mo_entropy_num_samples": 17,
                "mo_entropy_estimation_type": "LB2",
                "mo_entropy_maximize": False,
                "mo_entropy_optimizer": optimizer,
                "mo_entropy_optimizer_kwargs": {"pop_size": 77},
            },
        ),
    )

    assert resolved.acqf_kwargs["hypercell_bounds"] is hypercell_bounds
    assert resolved.acqf_kwargs["estimation_type"] == "LB2"
    assert resolved.acqf_kwargs["num_samples"] == 17
    assert captured["sample"]["num_samples"] == 3
    assert captured["sample"]["num_points"] == 4
    assert captured["sample"]["maximize"] is False
    assert captured["sample"]["optimizer"] is optimizer
    assert captured["sample"]["optimizer_kwargs"] == {"pop_size": 77}
    assert torch.equal(
        captured["sample"]["bounds"],
        torch.tensor([[0.0, -1.0], [1.0, 2.0]], dtype=torch.double),
    )
    assert captured["box"]["fronts"] is pareto_fronts
    assert captured["box"]["maximize"] is False


def test_mo_mes_preserves_explicit_hypercell_bounds(monkeypatch) -> None:
    bundle = _make_bundle()
    hypercell_bounds = torch.zeros(2, 2, 3, 2, dtype=torch.double)
    monkeypatch.setattr(
        info_defaults,
        "_sample_multiobjective_optimal_points",
        lambda **kwargs: pytest.fail("MO-MES Pareto sampling must not run"),
    )

    resolved, _ = info_defaults.resolve_information_acquisition_defaults(
        bundle,
        AcquisitionConfig(
            name="mo_mes",
            acqf_cls=qLowerBoundMultiObjectiveMaxValueEntropySearch,
            acqf_kwargs={"hypercell_bounds": hypercell_bounds},
        ),
        DataContext(),
    )

    assert resolved.acqf_kwargs["hypercell_bounds"] is hypercell_bounds


def test_mo_jes_generates_pareto_samples_and_hypercells(monkeypatch) -> None:
    bundle = _make_bundle()
    pareto_sets = torch.zeros(2, 3, 2, dtype=torch.double)
    pareto_fronts = torch.ones(2, 3, 2, dtype=torch.double)
    hypercell_bounds = torch.zeros(2, 2, 4, 2, dtype=torch.double)

    monkeypatch.setattr(
        info_defaults,
        "_sample_multiobjective_optimal_points",
        lambda **kwargs: (pareto_sets, pareto_fronts),
    )
    monkeypatch.setattr(
        info_defaults,
        "_compute_multiobjective_hypercell_bounds",
        lambda fronts, *, maximize: hypercell_bounds,
    )

    resolved, _ = engine_defaults.resolve_acquisition_defaults(
        bundle,
        AcquisitionConfig(
            name="mo_jes",
            acqf_cls=qLowerBoundMultiObjectiveJointEntropySearch,
        ),
        DataContext(
            bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
            extra={
                "mo_entropy_num_pareto_samples": 2,
                "mo_entropy_num_pareto_points": 3,
            },
        ),
    )

    assert resolved.acqf_kwargs["pareto_sets"] is pareto_sets
    assert resolved.acqf_kwargs["pareto_fronts"] is pareto_fronts
    assert resolved.acqf_kwargs["hypercell_bounds"] is hypercell_bounds
    assert resolved.acqf_kwargs["estimation_type"] == "LB"
    assert resolved.acqf_kwargs["num_samples"] == 64


def test_mo_jes_explicit_pareto_pair_only_computes_hypercells(monkeypatch) -> None:
    bundle = _make_bundle()
    pareto_sets = torch.zeros(2, 3, 2, dtype=torch.double)
    pareto_fronts = torch.ones(2, 3, 2, dtype=torch.double)
    hypercell_bounds = torch.zeros(2, 2, 4, 2, dtype=torch.double)
    captured = {}

    monkeypatch.setattr(
        info_defaults,
        "_sample_multiobjective_optimal_points",
        lambda **kwargs: pytest.fail("MO-JES Pareto sampling must not run"),
    )

    def fake_box(fronts, *, maximize):
        captured["fronts"] = fronts
        captured["maximize"] = maximize
        return hypercell_bounds

    monkeypatch.setattr(
        info_defaults,
        "_compute_multiobjective_hypercell_bounds",
        fake_box,
    )

    resolved, _ = info_defaults.resolve_information_acquisition_defaults(
        bundle,
        AcquisitionConfig(
            name="mo_jes",
            acqf_cls=qLowerBoundMultiObjectiveJointEntropySearch,
            acqf_kwargs={
                "pareto_sets": pareto_sets,
                "pareto_fronts": pareto_fronts,
            },
        ),
        DataContext(),
    )

    assert resolved.acqf_kwargs["pareto_sets"] is pareto_sets
    assert resolved.acqf_kwargs["pareto_fronts"] is pareto_fronts
    assert resolved.acqf_kwargs["hypercell_bounds"] is hypercell_bounds
    assert captured["fronts"] is pareto_fronts
    assert captured["maximize"] is True


def test_mo_jes_requires_explicit_pareto_samples_as_pair() -> None:
    bundle = _make_bundle()
    with pytest.raises(ValueError, match="supplied together"):
        info_defaults.resolve_information_acquisition_defaults(
            bundle,
            AcquisitionConfig(
                name="mo_jes",
                acqf_cls=qLowerBoundMultiObjectiveJointEntropySearch,
                acqf_kwargs={"pareto_sets": torch.zeros(1, 2, 2)},
            ),
            DataContext(),
        )


def test_mo_jes_rejects_hypercells_without_matching_pareto_pair() -> None:
    bundle = _make_bundle()
    with pytest.raises(ValueError, match="cannot be supplied without"):
        info_defaults.resolve_information_acquisition_defaults(
            bundle,
            AcquisitionConfig(
                name="mo_jes",
                acqf_cls=qLowerBoundMultiObjectiveJointEntropySearch,
                acqf_kwargs={"hypercell_bounds": torch.zeros(1, 2, 2, 2)},
            ),
            DataContext(),
        )


@pytest.mark.parametrize(
    ("name", "cls"),
    [
        ("mo_mes", qLowerBoundMultiObjectiveMaxValueEntropySearch),
        ("mo_jes", qLowerBoundMultiObjectiveJointEntropySearch),
    ],
)
def test_native_multiobjective_entropy_requires_two_outputs(name: str, cls: type) -> None:
    bundle = _make_bundle()
    bundle.train_Y = bundle.train_Y[:, :1]
    with pytest.raises(ValueError, match="at least two objectives"):
        info_defaults.resolve_information_acquisition_defaults(
            bundle,
            AcquisitionConfig(name=name, acqf_cls=cls),
            DataContext(),
        )


@pytest.mark.parametrize(
    ("name", "cls"),
    [
        ("mo_mes", qLowerBoundMultiObjectiveMaxValueEntropySearch),
        ("mo_jes", qLowerBoundMultiObjectiveJointEntropySearch),
    ],
)
def test_native_multiobjective_entropy_rejects_scalar_objective(
    name: str,
    cls: type,
) -> None:
    bundle = _make_bundle()
    with pytest.raises(ValueError, match="directly on all model outputs"):
        info_defaults.resolve_information_acquisition_defaults(
            bundle,
            AcquisitionConfig(name=name, acqf_cls=cls, objective=object()),
            DataContext(),
        )


def test_native_multiobjective_entropy_rejects_auto_scalarization() -> None:
    bundle = _make_bundle()
    context = DataContext(
        multi_objective=MultiObjectiveConfig(
            scalarization_weights=torch.tensor([0.4, 0.6], dtype=torch.double),
            auto_scalarization=True,
        )
    )
    with pytest.raises(ValueError, match="must not use.*scalarization"):
        info_defaults.resolve_information_acquisition_defaults(
            bundle,
            AcquisitionConfig(
                name="mo_mes",
                acqf_cls=qLowerBoundMultiObjectiveMaxValueEntropySearch,
                acqf_kwargs={"hypercell_bounds": torch.zeros(1, 2, 2, 2)},
            ),
            context,
        )


def test_native_multiobjective_entropy_disables_empty_auto_scalarization() -> None:
    bundle = _make_bundle()
    mo_config = MultiObjectiveConfig(auto_scalarization=True)
    resolved, context = info_defaults.resolve_information_acquisition_defaults(
        bundle,
        AcquisitionConfig(
            name="mo_mes",
            acqf_cls=qLowerBoundMultiObjectiveMaxValueEntropySearch,
            acqf_kwargs={"hypercell_bounds": torch.zeros(1, 2, 2, 2)},
        ),
        DataContext(multi_objective=mo_config),
    )

    assert resolved.objective is None
    assert context.multi_objective is not mo_config
    assert context.multi_objective.auto_scalarization is False
    assert mo_config.auto_scalarization is True


def test_automatic_multiobjective_entropy_rejects_mixed_inputs() -> None:
    bundle = _make_bundle(input_type="mixed", cat_dims=[1])
    with pytest.raises(ValueError, match="mixed/categorical"):
        info_defaults.resolve_information_acquisition_defaults(
            bundle,
            AcquisitionConfig(
                name="mo_mes",
                acqf_cls=qLowerBoundMultiObjectiveMaxValueEntropySearch,
            ),
            DataContext(
                bounds=torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.double)
            ),
        )


def test_explicit_multiobjective_entropy_inputs_allow_mixed_model() -> None:
    bundle = _make_bundle(input_type="mixed", cat_dims=[1])
    hypercell_bounds = torch.zeros(1, 2, 2, 2, dtype=torch.double)

    resolved, _ = info_defaults.resolve_information_acquisition_defaults(
        bundle,
        AcquisitionConfig(
            name="mo_mes",
            acqf_cls=qLowerBoundMultiObjectiveMaxValueEntropySearch,
            acqf_kwargs={"hypercell_bounds": hypercell_bounds},
        ),
        DataContext(),
    )

    assert resolved.acqf_kwargs["hypercell_bounds"] is hypercell_bounds


@pytest.mark.parametrize(
    ("name", "cls"),
    [
        ("mo_mes", qLowerBoundMultiObjectiveMaxValueEntropySearch),
        ("mo_jes", qLowerBoundMultiObjectiveJointEntropySearch),
    ],
)
def test_multiobjective_entropy_q_greater_than_one_is_sequential(
    name: str,
    cls: type,
) -> None:
    resolved = info_defaults.resolve_information_optimizer_defaults(
        AcquisitionConfig(name=name, acqf_cls=cls),
        OptimizeConfig(q=3, sequential=False),
    )
    assert resolved.sequential is True


def test_multiobjective_entropy_rejects_unknown_estimation_type() -> None:
    bundle = _make_bundle()
    with pytest.raises(ValueError, match="estimation_type"):
        info_defaults.resolve_information_acquisition_defaults(
            bundle,
            AcquisitionConfig(
                name="mo_mes",
                acqf_cls=qLowerBoundMultiObjectiveMaxValueEntropySearch,
                acqf_kwargs={
                    "hypercell_bounds": torch.zeros(1, 2, 2, 2),
                    "estimation_type": "unknown",
                },
            ),
            DataContext(),
        )
