from types import SimpleNamespace

import pytest
import torch

from bochan.api.acquisition import multifidelity as mf_acquisition
from bochan.api.configs import AcquisitionConfig, DataContext
from bochan.models.multifidelity import FidelityCostConfig, build_fidelity_cost_utility


class _Model:
    fidelity_features = (2,)
    target_fidelities = {2: 1.0}
    cat_dims = ()


class _Bundle(SimpleNamespace):
    pass


def _bundle():
    return _Bundle(
        model=_Model(),
        train_X=torch.tensor(
            [[0.1, 0.0, 0.25], [0.4, 1.0, 0.5], [0.8, 0.0, 1.0]],
            dtype=torch.double,
        ),
        train_Y=torch.tensor([[0.1], [0.3], [0.7]], dtype=torch.double),
    )


def _context():
    return DataContext(
        bounds=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            dtype=torch.double,
        )
    )


def test_fidelity_cost_config_validation():
    config = FidelityCostConfig(fixed_cost=2.0, fidelity_weights={2: 3.0})
    assert config.kind == "affine"
    assert config.fixed_cost == 2.0
    assert config.fidelity_weights == {2: 3.0}

    with pytest.raises(ValueError, match="affine"):
        FidelityCostConfig(kind="learned")
    with pytest.raises(ValueError, match="non-negative"):
        FidelityCostConfig(fixed_cost=-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        FidelityCostConfig(fidelity_weights={2: -1.0})


def test_build_fidelity_cost_utility_uses_default_weight():
    cost_model, utility = build_fidelity_cost_utility(
        FidelityCostConfig(fixed_cost=1.0),
        fidelity_features=(2,),
    )
    X = torch.tensor([[[0.2, 0.0, 0.5]]], dtype=torch.double)
    cost = cost_model(X).squeeze().item()
    assert cost == pytest.approx(1.5)
    assert utility.cost_model is cost_model


def test_rejects_cost_weight_for_non_fidelity_dimension():
    with pytest.raises(ValueError, match="unknown indices"):
        build_fidelity_cost_utility(
            FidelityCostConfig(fidelity_weights={1: 1.0}),
            fidelity_features=(2,),
        )


def test_mfkg_injects_cost_aware_utility(monkeypatch):
    captured = {}

    def fake_mfkg(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(model=kwargs["model"])

    monkeypatch.setattr(mf_acquisition, "qMultiFidelityKnowledgeGradient", fake_mfkg)
    result = mf_acquisition.build_multifidelity_acquisition(
        bundle=_bundle(),
        config=AcquisitionConfig(
            name="mfkg",
            acqf_kwargs={
                "current_value": torch.tensor(0.0, dtype=torch.double),
                "cost_config": FidelityCostConfig(
                    fixed_cost=1.0,
                    fidelity_weights={2: 4.0},
                ),
            },
        ),
        data_context=_context(),
    )

    assert captured["cost_aware_utility"] is result._bochan_cost_aware_utility
    assert result._bochan_cost_model is captured["cost_aware_utility"].cost_model


def test_mfmes_accepts_mapping_cost_config(monkeypatch):
    captured = {}

    def fake_mfmes(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(model=kwargs["model"])

    monkeypatch.setattr(mf_acquisition, "qMultiFidelityMaxValueEntropy", fake_mfmes)
    mf_acquisition.build_multifidelity_acquisition(
        bundle=_bundle(),
        config=AcquisitionConfig(
            name="qmfmes",
            acqf_kwargs={
                "candidate_set_size": 16,
                "cost_config": {
                    "fixed_cost": 2.0,
                    "fidelity_weights": {2: 5.0},
                },
            },
        ),
        data_context=_context(),
    )
    assert "cost_aware_utility" in captured


def test_rejects_both_explicit_cost_utility_and_cost_config():
    with pytest.raises(ValueError, match="either cost_aware_utility or cost_config"):
        mf_acquisition.build_multifidelity_acquisition(
            bundle=_bundle(),
            config=AcquisitionConfig(
                name="mfkg",
                acqf_kwargs={
                    "current_value": torch.tensor(0.0, dtype=torch.double),
                    "cost_aware_utility": object(),
                    "cost_config": FidelityCostConfig(),
                },
            ),
            data_context=_context(),
        )
