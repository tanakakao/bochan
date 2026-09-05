from types import SimpleNamespace

import pytest
import torch

from bochan.api import AcquisitionConfig, OptimizeConfig
from bochan.api.acquisition import multifidelity as mf_acquisition
from bochan.serving.fastapi.schemas.configs import OptimizeConfigSchema
from bochan.serving.fastapi.schemas.requests import CandidateRequest
from bochan.serving.fastapi.services.candidates import _inject_multifidelity_options


class _Model:
    fidelity_features = (2,)
    target_fidelities = {2: 1.0}
    cat_dims = ()


def test_fastapi_optimize_schema_exposes_multifidelity_modes():
    discrete = OptimizeConfigSchema(fidelity_values=[0.25, 0.5, 1.0])
    assert discrete.fidelity_values == [0.25, 0.5, 1.0]
    continuous = OptimizeConfigSchema(optimize_fidelity=True)
    assert continuous.optimize_fidelity is True
    with pytest.raises(ValueError, match="Specify only one of fidelity_values, fidelity_assignments, or optimize_fidelity"):
        OptimizeConfigSchema(fidelity_values=[0.5, 1.0], optimize_fidelity=True)


def test_candidate_request_accepts_phase53_convenience_fields():
    request = CandidateRequest.model_validate(
        {
            "acquisition_config": {"name": "mfkg"},
            "target_fidelity": 0.9,
            "cost_config": {
                "kind": "affine",
                "fixed_cost": 1.0,
                "fidelity_weights": {"2": 4.0},
            },
            "fidelity_values": [0.25, 0.5, 1.0],
        }
    )
    assert request.target_fidelity == pytest.approx(0.9)
    assert request.fidelity_values == [0.25, 0.5, 1.0]


def test_transport_convenience_fields_merge_into_core_configs():
    request = SimpleNamespace(
        target_fidelity=0.9,
        cost_config={"kind": "affine", "fixed_cost": 1.0, "fidelity_weights": {2: 4.0}},
        fidelity_values=[0.25, 0.5, 1.0],
        optimize_fidelity=None,
    )
    acq, opt = _inject_multifidelity_options(
        AcquisitionConfig(name="mfkg"),
        OptimizeConfig(),
        request,
    )
    assert acq.acqf_kwargs["target_fidelity"] == pytest.approx(0.9)
    assert acq.acqf_kwargs["cost_config"]["fixed_cost"] == pytest.approx(1.0)
    assert opt.fidelity_values == (0.25, 0.5, 1.0)


def test_target_fidelity_override_does_not_mutate_model(monkeypatch):
    captured = {}

    def fake_mfkg(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(model=kwargs["model"])

    monkeypatch.setattr(mf_acquisition, "qMultiFidelityKnowledgeGradient", fake_mfkg)
    model = _Model()
    bundle = SimpleNamespace(
        model=model,
        train_X=torch.tensor(
            [[0.1, 0.0, 0.25], [0.5, 0.0, 0.5], [0.8, 0.0, 1.0]],
            dtype=torch.double,
        ),
        train_Y=torch.tensor([[0.1], [0.4], [0.8]], dtype=torch.double),
    )
    context = SimpleNamespace(
        bounds=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.double),
        X_pending=None,
    )
    mf_acquisition.build_multifidelity_acquisition(
        bundle=bundle,
        config=AcquisitionConfig(
            name="mfkg",
            acqf_kwargs={
                "target_fidelity": 0.75,
                "current_value": torch.tensor(0.0, dtype=torch.double),
            },
        ),
        data_context=context,
    )
    projected = captured["project"](torch.tensor([[0.2, 0.0, 0.25]], dtype=torch.double))
    assert projected[0, 2].item() == pytest.approx(0.75)
    assert model.target_fidelities == {2: 1.0}


def test_target_fidelity_override_rejects_out_of_bounds():
    bundle = SimpleNamespace(
        model=_Model(),
        train_X=torch.tensor([[0.1, 0.0, 0.25]], dtype=torch.double),
        train_Y=torch.tensor([[0.1]], dtype=torch.double),
    )
    context = SimpleNamespace(
        bounds=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.double),
        X_pending=None,
    )
    with pytest.raises(ValueError, match="outside bounds"):
        mf_acquisition.build_multifidelity_acquisition(
            bundle=bundle,
            config=AcquisitionConfig(
                name="mfkg",
                acqf_kwargs={
                    "target_fidelity": 1.2,
                    "current_value": torch.tensor(0.0, dtype=torch.double),
                },
            ),
            data_context=context,
        )
