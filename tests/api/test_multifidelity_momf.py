from __future__ import annotations

from types import SimpleNamespace

import torch

from bochan.api import ModelConfig, MultiOutputConfig
from bochan.api.acquisition import multifidelity_momf as momf_module
from bochan.api.acquisition.service import resolve_acquisition_class
from bochan.api.configs import AcquisitionConfig, DataContext
from bochan.api.modeling.build import build_model


def _training_data():
    train_X = torch.tensor(
        [
            [0.00, 0.25],
            [0.20, 0.50],
            [0.40, 1.00],
            [0.60, 0.25],
            [0.80, 0.50],
            [1.00, 1.00],
        ],
        dtype=torch.double,
    )
    x = train_X[:, :1]
    fidelity = train_X[:, 1:2]
    train_Y = torch.cat(
        [
            0.2 + x + 0.15 * fidelity,
            1.2 - 0.7 * x + 0.10 * fidelity,
        ],
        dim=-1,
    )
    return train_X, train_Y


def _bundle():
    train_X, train_Y = _training_data()
    return build_model(
        train_X,
        train_Y,
        ModelConfig(
            task_type="regression",
            model_type="multifidelity_gp",
            input_type="normal",
            model_kwargs={
                "fidelity_features": [-1],
                "target_fidelities": {-1: 1.0},
            },
            multi_output_config=MultiOutputConfig(),
        ),
    )


def _context():
    return DataContext(
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        ref_point=torch.tensor([-0.5, -0.5], dtype=torch.double),
    )


def test_momf_aliases():
    for name in (
        "momf",
        "qmomf",
        "multi_objective_multi_fidelity",
        "qMultiObjectiveMultiFidelity",
    ):
        assert momf_module.is_momf_name(name)


def test_momf_service_routing_requires_multioutput_multifidelity():
    optimizer = SimpleNamespace(
        _check_fitted=lambda: None,
        _acquisition_routing_context=lambda: ("regression", "multifidelity_gp", True),
    )
    resolved = resolve_acquisition_class(optimizer, AcquisitionConfig(name="momf"))
    assert resolved.acqf_factory is momf_module.build_momf_acquisition


def test_momf_builds_augmented_hypervolume_problem(monkeypatch):
    captured = {}

    class FakeMOMF:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.model = kwargs["model"]

    monkeypatch.setattr(momf_module, "MOMF", FakeMOMF)
    acqf = momf_module.build_momf_acquisition(
        bundle=_bundle(),
        config=AcquisitionConfig(
            name="momf",
            acqf_kwargs={
                "cost_config": {
                    "fixed_cost": 1.0,
                    "fidelity_weights": {-1: 2.0},
                }
            },
        ),
        data_context=_context(),
    )

    assert getattr(acqf, "_bochan_multifidelity_kind") == "momf"
    assert captured["ref_point"].shape == torch.Size([3])
    assert captured["partitioning"].num_outcomes == 3
    assert callable(captured["cost_call"])

    samples = torch.zeros(2, 1, 1, 2, dtype=torch.double)
    X = torch.tensor([[[0.4, 0.5]]], dtype=torch.double)
    augmented = captured["objective"](samples, X=X)
    assert augmented.shape[-1] == 3
    assert torch.allclose(augmented[..., -1], torch.full_like(augmented[..., -1], 0.5))


def test_momf_custom_fidelity_objective_is_used(monkeypatch):
    captured = {}

    class FakeMOMF:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.model = kwargs["model"]

    monkeypatch.setattr(momf_module, "MOMF", FakeMOMF)
    fidelity_objective = lambda X: X[..., 1].square()
    momf_module.build_momf_acquisition(
        bundle=_bundle(),
        config=AcquisitionConfig(
            name="momf",
            acqf_kwargs={
                "fidelity_objective": fidelity_objective,
                "fidelity_ref_point": -0.1,
            },
        ),
        data_context=_context(),
    )

    assert torch.isclose(captured["ref_point"][-1], torch.tensor(-0.1, dtype=torch.double))
    samples = torch.zeros(1, 1, 1, 2, dtype=torch.double)
    X = torch.tensor([[[0.2, 0.5]]], dtype=torch.double)
    augmented = captured["objective"](samples, X=X)
    assert torch.allclose(augmented[..., -1], torch.full_like(augmented[..., -1], 0.25))


def test_momf_numerical_smoke():
    torch.manual_seed(0)
    acqf = momf_module.build_momf_acquisition(
        bundle=_bundle(),
        config=AcquisitionConfig(name="momf"),
        data_context=_context(),
    )
    X = torch.tensor([[[0.35, 0.50]]], dtype=torch.double)
    value = acqf(X)
    assert torch.isfinite(value).all()
