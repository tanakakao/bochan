from __future__ import annotations

from types import SimpleNamespace

import torch
from botorch.optim.initializers import gen_one_shot_hvkg_initial_conditions

from bochan.api import ModelConfig, MultiOutputConfig
from bochan.api.acquisition import multifidelity_hvkg as mf_hvkg
from bochan.api.configs import AcquisitionConfig, DataContext, OptimizeConfig
from bochan.api.modeling.build import build_model
from bochan.api.optimizer.dispatch import optimize_candidates


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


def _context(**extra):
    return DataContext(
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        ref_point=torch.tensor([-0.5, -0.5], dtype=torch.double),
        extra=extra,
    )


def test_mfhvkg_uses_shared_model_list_fidelity_metadata(monkeypatch):
    captured = {}

    def fake_hvkg(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(model=kwargs["model"])

    monkeypatch.setattr(
        mf_hvkg,
        "qMultiFidelityHypervolumeKnowledgeGradient",
        fake_hvkg,
    )

    bundle = _bundle()
    mf_hvkg.build_multifidelity_hvkg_acquisition(
        bundle=bundle,
        config=AcquisitionConfig(
            name="mfhvkg",
            acqf_kwargs={
                "current_value": 0.0,
                "cost_config": {
                    "fixed_cost": 1.0,
                    "fidelity_weights": {-1: 2.0},
                },
            },
        ),
        data_context=_context(),
    )

    assert captured["target_fidelities"] == {1: 1.0}
    assert captured["cost_aware_utility"] is not None


def test_mfhvkg_resolves_objective_factory_and_evaluation_masks(monkeypatch):
    captured = {}
    sentinel_objective = object()

    def fake_hvkg(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(model=kwargs["model"])

    monkeypatch.setattr(
        mf_hvkg,
        "qMultiFidelityHypervolumeKnowledgeGradient",
        fake_hvkg,
    )

    evaluation_mask = torch.tensor([[True, True]], dtype=torch.bool)
    pending_mask = torch.tensor([[True, False]], dtype=torch.bool)
    mf_hvkg.build_multifidelity_hvkg_acquisition(
        bundle=_bundle(),
        config=AcquisitionConfig(
            name="mfhvkg",
            objective_factory=lambda **kwargs: sentinel_objective,
            acqf_kwargs={"current_value": 0.0},
        ),
        data_context=_context(
            X_evaluation_mask=evaluation_mask,
            X_pending_evaluation_mask=pending_mask,
        ),
    )

    assert captured["objective"] is sentinel_objective
    assert torch.equal(captured["X_evaluation_mask"], evaluation_mask)
    assert torch.equal(captured["X_pending_evaluation_mask"], pending_mask)


def test_standard_mfhvkg_injects_specialized_hvkg_initializer():
    model = SimpleNamespace(fidelity_features=(1,), target_fidelities={1: 1.0})
    acqf = SimpleNamespace(model=model, _bochan_multifidelity_kind="mfhvkg")
    captured = {}

    def backend(*, acqf, bounds, config):
        captured["config"] = config
        return torch.zeros(1, 2, dtype=torch.double), torch.tensor(0.0)

    optimize_candidates(
        acqf,
        _context().bounds,
        OptimizeConfig(
            optimize_fidelity=True,
            ensure_unique_candidates=False,
        ),
        base_optimize_candidates=backend,
    )

    assert (
        captured["config"].optimizer_kwargs["ic_generator"]
        is gen_one_shot_hvkg_initial_conditions
    )


def test_inter_point_constraints_keep_best_subset_compatible_initializer_path():
    model = SimpleNamespace(fidelity_features=(1,), target_fidelities={1: 1.0})
    acqf = SimpleNamespace(model=model, _bochan_multifidelity_kind="mfhvkg")
    captured = {}
    equality = [
        (
            torch.tensor([[0, 0]], dtype=torch.long),
            torch.tensor([1.0], dtype=torch.double),
            0.0,
        )
    ]

    def backend(*, acqf, bounds, config):
        captured["config"] = config
        return torch.zeros(1, 2, dtype=torch.double), torch.tensor(0.0)

    optimize_candidates(
        acqf,
        _context().bounds,
        OptimizeConfig(
            optimize_fidelity=True,
            equality_constraints=equality,
            ensure_unique_candidates=False,
        ),
        base_optimize_candidates=backend,
    )

    assert "ic_generator" not in captured["config"].optimizer_kwargs


def test_mfhvkg_q2_continuous_fidelity_numerical_smoke():
    torch.manual_seed(0)
    bundle = _bundle()
    context = _context()
    acqf = mf_hvkg.build_multifidelity_hvkg_acquisition(
        bundle=bundle,
        config=AcquisitionConfig(
            name="mfhvkg",
            acqf_kwargs={
                "current_value": 0.0,
                "num_fantasies": 2,
                "num_pareto": 2,
            },
        ),
        data_context=context,
    )

    candidates, value = optimize_candidates(
        acqf,
        context.bounds,
        OptimizeConfig(
            q=2,
            num_restarts=2,
            raw_samples=16,
            optimize_fidelity=True,
            ensure_unique_candidates=False,
            optimizer_kwargs={"options": {"maxiter": 5, "batch_limit": 1}},
        ),
    )

    assert candidates.shape == (2, 2)
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(torch.as_tensor(value)).all()
    assert bool(((candidates[:, 1] >= 0.0) & (candidates[:, 1] <= 1.0)).all())
