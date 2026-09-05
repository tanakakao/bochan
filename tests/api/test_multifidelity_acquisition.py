from types import SimpleNamespace

import pytest
import torch

from bochan.api.acquisition import multifidelity as mf_acquisition
from bochan.api.configs import AcquisitionConfig, DataContext, OptimizeConfig
from bochan.api.optimizer.dispatch import optimize_candidates


class _Model:
    fidelity_features = (2,)
    target_fidelities = {2: 1.0}
    cat_dims = ()


class _Bundle(SimpleNamespace):
    pass


def _bundle(*, cat_dims=()):
    model = _Model()
    model.cat_dims = tuple(cat_dims)
    return _Bundle(
        model=model,
        train_X=torch.tensor(
            [
                [0.1, 0.0, 0.25],
                [0.4, 1.0, 0.5],
                [0.8, 0.0, 1.0],
            ],
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


def test_multifidelity_acquisition_name_aliases():
    for name in ["mfkg", "qmfkg", "qMultiFidelityKnowledgeGradient", "mfmes", "qmfmes"]:
        assert mf_acquisition.is_multifidelity_acquisition_name(name)
    assert not mf_acquisition.is_multifidelity_acquisition_name("qlogei")


def test_requires_target_fidelities():
    bundle = _bundle()
    bundle.model.target_fidelities = None
    with pytest.raises(ValueError, match="target_fidelities"):
        mf_acquisition.build_multifidelity_acquisition(
            bundle=bundle,
            config=AcquisitionConfig(name="mfkg", acqf_kwargs={"current_value": torch.tensor(0.0)}),
            data_context=_context(),
        )


def test_mfkg_factory_injects_target_projection(monkeypatch):
    captured = {}

    def fake_mfkg(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(model=kwargs["model"])

    monkeypatch.setattr(mf_acquisition, "qMultiFidelityKnowledgeGradient", fake_mfkg)
    bundle = _bundle()
    current_value = torch.tensor(0.25, dtype=torch.double)
    result = mf_acquisition.build_multifidelity_acquisition(
        bundle=bundle,
        config=AcquisitionConfig(
            name="mfkg",
            acqf_kwargs={"current_value": current_value, "num_fantasies": 8},
        ),
        data_context=_context(),
    )

    assert result.model is bundle.model
    assert captured["current_value"] is current_value
    assert captured["num_fantasies"] == 8
    X = torch.tensor([[0.2, 0.0, 0.25]], dtype=torch.double)
    projected = captured["project"](X)
    assert projected[0, 2].item() == pytest.approx(1.0)


def test_mfmes_generates_target_fidelity_candidate_set(monkeypatch):
    captured = {}

    def fake_mfmes(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(model=kwargs["model"])

    monkeypatch.setattr(mf_acquisition, "qMultiFidelityMaxValueEntropy", fake_mfmes)
    bundle = _bundle(cat_dims=(1,))
    mf_acquisition.build_multifidelity_acquisition(
        bundle=bundle,
        config=AcquisitionConfig(
            name="qmfmes",
            acqf_kwargs={"candidate_set_size": 32, "num_mv_samples": 4},
        ),
        data_context=_context(),
    )

    candidate_set = captured["candidate_set"]
    assert candidate_set.shape == (32, 3)
    assert torch.all(candidate_set[:, 2] == 1.0)
    assert set(candidate_set[:, 1].tolist()).issubset({0.0, 1.0})
    assert captured["num_mv_samples"] == 4


def test_phase50_mf_acquisition_requires_discrete_fidelity_values():
    AcqType = type("qMultiFidelityKnowledgeGradient", (), {})
    acqf = AcqType()
    acqf.model = _Model()
    bounds = _context().bounds

    with pytest.raises(ValueError, match="fidelity_values"):
        optimize_candidates(
            acqf,
            bounds,
            OptimizeConfig(ensure_unique_candidates=False),
            base_optimize_candidates=lambda **kwargs: (torch.zeros(1, 3), torch.tensor(0.0)),
        )


def test_phase50_mf_acquisition_uses_phase49_discrete_enumeration():
    AcqType = type("qMultiFidelityMaxValueEntropy", (), {})
    acqf = AcqType()
    acqf.model = _Model()
    captured = {}

    def backend(*, acqf, bounds, config):
        captured["config"] = config
        return torch.zeros(1, 3, dtype=torch.double), torch.tensor(0.0)

    optimize_candidates(
        acqf,
        _context().bounds,
        OptimizeConfig(
            fidelity_values=[0.25, 0.5, 1.0],
            ensure_unique_candidates=False,
        ),
        base_optimize_candidates=backend,
    )

    resolved = captured["config"]
    assert resolved.fixed_features is None
    assert resolved.fixed_features_list == [
        {2: 0.25},
        {2: 0.5},
        {2: 1.0},
    ]
    assert "mixed" in str(resolved.optimizer)
