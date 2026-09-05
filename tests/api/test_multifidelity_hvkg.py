from types import SimpleNamespace

import pytest
import torch

from bochan.api.acquisition import multifidelity_hvkg as mf_hvkg
from bochan.api.acquisition.service import resolve_acquisition_class
from bochan.api.configs import AcquisitionConfig, DataContext, OptimizeConfig
from bochan.api.optimizer.dispatch import optimize_candidates


class _Model:
    fidelity_features = (2,)
    target_fidelities = {2: 1.0}
    cat_dims = ()
    num_outputs = 2


class _Bundle(SimpleNamespace):
    pass


def _bundle():
    return _Bundle(
        model=_Model(),
        train_X=torch.tensor(
            [
                [0.1, 0.0, 0.25],
                [0.4, 0.0, 0.5],
                [0.8, 0.0, 1.0],
            ],
            dtype=torch.double,
        ),
        train_Y=torch.tensor(
            [[0.1, 0.5], [0.4, 0.3], [0.8, 0.7]],
            dtype=torch.double,
        ),
        metadata={"multi_output": True},
    )


def _context():
    return DataContext(
        bounds=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            dtype=torch.double,
        ),
        ref_point=torch.tensor([0.0, 0.0], dtype=torch.double),
    )


def test_mfhvkg_aliases():
    for name in [
        "mfhvkg",
        "qmfhvkg",
        "qMultiFidelityHypervolumeKnowledgeGradient",
        "MultiFidelityHypervolumeKnowledgeGradient",
    ]:
        assert mf_hvkg.is_multifidelity_hvkg_name(name)
    assert not mf_hvkg.is_multifidelity_hvkg_name("qhvkg")


def test_mfhvkg_factory_injects_target_projection_and_ref_point(monkeypatch):
    captured = {}

    def fake_hvkg(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(model=kwargs["model"])

    monkeypatch.setattr(
        mf_hvkg,
        "qMultiFidelityHypervolumeKnowledgeGradient",
        fake_hvkg,
    )
    monkeypatch.setattr(
        mf_hvkg,
        "_current_hypervolume_value",
        lambda **kwargs: torch.tensor(0.25, dtype=torch.double),
    )

    bundle = _bundle()
    acqf = mf_hvkg.build_multifidelity_hvkg_acquisition(
        bundle=bundle,
        config=AcquisitionConfig(
            name="qmfhvkg",
            acqf_kwargs={"num_fantasies": 4, "num_pareto": 6},
        ),
        data_context=_context(),
    )

    assert acqf.model is bundle.model
    assert captured["target_fidelities"] == {2: 1.0}
    assert captured["num_fantasies"] == 4
    assert captured["num_pareto"] == 6
    assert torch.equal(
        captured["ref_point"],
        torch.tensor([0.0, 0.0], dtype=torch.double),
    )
    X = torch.tensor([[0.2, 0.0, 0.25]], dtype=torch.double)
    projected = captured["project"](X)
    assert projected[0, 2].item() == pytest.approx(1.0)
    assert captured["current_value"].item() == pytest.approx(0.25)


def test_mfhvkg_cost_config_supports_negative_fidelity_index(monkeypatch):
    captured = {}

    def fake_hvkg(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(model=kwargs["model"])

    monkeypatch.setattr(
        mf_hvkg,
        "qMultiFidelityHypervolumeKnowledgeGradient",
        fake_hvkg,
    )
    monkeypatch.setattr(
        mf_hvkg,
        "_current_hypervolume_value",
        lambda **kwargs: torch.tensor(0.25, dtype=torch.double),
    )

    mf_hvkg.build_multifidelity_hvkg_acquisition(
        bundle=_bundle(),
        config=AcquisitionConfig(
            name="mfhvkg",
            acqf_kwargs={
                "cost_config": {
                    "fixed_cost": 1.0,
                    "fidelity_weights": {-1: 4.0},
                }
            },
        ),
        data_context=_context(),
    )

    assert captured["cost_aware_utility"] is not None


def test_mfhvkg_target_override_is_request_local(monkeypatch):
    captured = {}

    def fake_hvkg(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(model=kwargs["model"])

    monkeypatch.setattr(
        mf_hvkg,
        "qMultiFidelityHypervolumeKnowledgeGradient",
        fake_hvkg,
    )
    monkeypatch.setattr(
        mf_hvkg,
        "_current_hypervolume_value",
        lambda **kwargs: torch.tensor(0.25, dtype=torch.double),
    )

    bundle = _bundle()
    mf_hvkg.build_multifidelity_hvkg_acquisition(
        bundle=bundle,
        config=AcquisitionConfig(
            name="mfhvkg",
            acqf_kwargs={"target_fidelity": 0.75},
        ),
        data_context=_context(),
    )

    assert captured["target_fidelities"] == {2: 0.75}
    assert bundle.model.target_fidelities == {2: 1.0}


def test_mfhvkg_requires_multi_output_model():
    bundle = _bundle()
    bundle.metadata["multi_output"] = False
    with pytest.raises(ValueError, match="multi-output"):
        mf_hvkg.build_multifidelity_hvkg_acquisition(
            bundle=bundle,
            config=AcquisitionConfig(name="mfhvkg"),
            data_context=_context(),
        )


class _Optimizer:
    def __init__(self, *, multi_output=True):
        self.bundle = _bundle()
        self._multi_output = multi_output

    def _check_fitted(self):
        return None

    def _acquisition_routing_context(self):
        return "regression", "multifidelity_gp", self._multi_output


def test_service_routes_mfhvkg_only_for_multi_output_multifidelity():
    resolved = resolve_acquisition_class(
        _Optimizer(multi_output=True),
        AcquisitionConfig(name="mfhvkg"),
    )
    assert resolved.acqf_factory is mf_hvkg.build_multifidelity_hvkg_acquisition

    with pytest.raises(ValueError, match="multi-output"):
        resolve_acquisition_class(
            _Optimizer(multi_output=False),
            AcquisitionConfig(name="mfhvkg"),
        )


def test_mfhvkg_optimizer_requires_query_fidelity_mode():
    AcqType = type("qMultiFidelityHypervolumeKnowledgeGradient", (), {})
    acqf = AcqType()
    acqf.model = _Model()

    with pytest.raises(ValueError, match="fidelity_values"):
        optimize_candidates(
            acqf,
            _context().bounds,
            OptimizeConfig(ensure_unique_candidates=False),
            base_optimize_candidates=lambda **kwargs: (
                torch.zeros(1, 3, dtype=torch.double),
                torch.tensor(0.0),
            ),
        )


def test_mfhvkg_optimizer_accepts_continuous_fidelity_search():
    AcqType = type("qMultiFidelityHypervolumeKnowledgeGradient", (), {})
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
            optimize_fidelity=True,
            ensure_unique_candidates=False,
        ),
        base_optimize_candidates=backend,
    )

    assert captured["config"].optimize_fidelity is True
    assert captured["config"].fixed_features is None
