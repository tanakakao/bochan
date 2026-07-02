from __future__ import annotations

from types import SimpleNamespace

import torch

import bochan.api as api_module
from bochan.api import AcquisitionConfig, OptimizeConfig
from bochan.optim.nsgaii_strategy import NSGAIIStrategy, build_nsgaii_strategy


def test_nsgaii_name_resolves_without_acquisition_registry() -> None:
    engine = SimpleNamespace(
        bundle=SimpleNamespace(
            task_type="regression",
            model_type="base",
            model=SimpleNamespace(num_outputs=2),
            metadata={"multi_output": True},
        ),
        acquisition_registry=None,
        _check_fitted=lambda: None,
    )

    resolved = api_module._resolve_acquisition_config_with_model_outputs(
        engine,
        AcquisitionConfig(name="nsgaii"),
    )

    assert resolved.acqf_cls is None
    assert resolved.acqf_factory is build_nsgaii_strategy


def test_nsgaii_name_ignores_optimizer_but_preserves_other_options(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_candidate(
        self,
        acq_config,
        opt_config,
        *,
        data_context=None,
        bounds=None,
        return_result=False,
    ):
        captured["acq_config"] = acq_config
        captured["opt_config"] = opt_config
        captured["data_context"] = data_context
        captured["bounds"] = bounds
        captured["return_result"] = return_result
        return "result"

    monkeypatch.setattr(api_module, "_original_candidate", fake_candidate)
    original = OptimizeConfig(
        optimizer="evo",
        q=3,
        optimizer_kwargs={"population_size": 80, "max_gen": 120},
    )

    result = api_module._candidate_with_acquisition_side_nsgaii(
        object(),
        AcquisitionConfig(name="NSGA-II"),
        original,
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        return_result=True,
    )

    resolved = captured["opt_config"]
    assert result == "result"
    assert resolved.optimizer == "nsgaii"
    assert resolved.q == 3
    assert resolved.optimizer_kwargs == {
        "population_size": 80,
        "max_gen": 120,
    }
    assert original.optimizer == "evo"


def test_non_nsgaii_name_keeps_selected_optimizer(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_candidate(
        self,
        acq_config,
        opt_config,
        *,
        data_context=None,
        bounds=None,
        return_result=False,
    ):
        captured["opt_config"] = opt_config
        return "result"

    monkeypatch.setattr(api_module, "_original_candidate", fake_candidate)
    original = OptimizeConfig(optimizer="evo", q=2)

    api_module._candidate_with_acquisition_side_nsgaii(
        object(),
        AcquisitionConfig(name="ehvi"),
        original,
    )

    assert captured["opt_config"] is original
    assert captured["opt_config"].optimizer == "evo"


def test_nsgaii_strategy_keeps_objective_constraints_and_ref_point(monkeypatch) -> None:
    objective = object()
    constraints = [lambda Y: 0.2 - Y[..., 0]]
    ref_point = torch.tensor([0.0, 0.0], dtype=torch.double)

    monkeypatch.setattr(
        "bochan.api.factory.build_objective",
        lambda bundle, config, data_context: objective,
    )
    bundle = SimpleNamespace(model=SimpleNamespace(num_outputs=2))
    config = AcquisitionConfig(name="nsgaii")
    context = SimpleNamespace(
        constraints=constraints,
        ref_point=ref_point,
    )

    strategy = build_nsgaii_strategy(
        bundle=bundle,
        config=config,
        data_context=context,
    )

    assert isinstance(strategy, NSGAIIStrategy)
    assert strategy.model is bundle.model
    assert strategy.objective is objective
    assert strategy.outcome_constraints == constraints
    torch.testing.assert_close(strategy.ref_point, ref_point)
