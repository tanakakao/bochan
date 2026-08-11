from __future__ import annotations

from types import SimpleNamespace

import torch

from bochan.api import AcquisitionConfig, BayesianOptimizer, DataContext, OptimizeConfig
from bochan.api.acquisition_service import (
    is_nsgaii_strategy,
    resolve_acquisition_class,
)
import bochan.api.optimizer as optimizer_module
from bochan.optim.nsgaii.strategy import NSGAIIStrategy, build_nsgaii_strategy


def test_nsgaii_name_resolves_without_acquisition_registry() -> None:
    optimizer = SimpleNamespace(
        bundle=SimpleNamespace(
            task_type="regression",
            model_type="base",
            model=SimpleNamespace(num_outputs=2),
            metadata={"multi_output": True},
        ),
        acquisition_registry=None,
        _check_fitted=lambda: None,
        _acquisition_routing_context=lambda: ("regression", "base", True),
    )

    resolved = resolve_acquisition_class(
        optimizer,
        AcquisitionConfig(name="nsgaii"),
    )

    assert resolved.acqf_cls is None
    assert resolved.acqf_factory is build_nsgaii_strategy


def _candidate_optimizer(
    monkeypatch,
    *,
    acq_config: AcquisitionConfig,
    captured: dict[str, object],
) -> BayesianOptimizer:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    optimizer = object.__new__(BayesianOptimizer)
    optimizer.bundle = SimpleNamespace(cat_dims=[])
    optimizer.observations = None
    optimizer.bounds = bounds
    optimizer.train_X = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    optimizer.history = []
    optimizer.llm_settings = None

    monkeypatch.setattr(
        BayesianOptimizer,
        "_prepare_acquisition",
        lambda self, config, data_context: (
            acq_config,
            DataContext(bounds=bounds),
            object(),
        ),
    )

    def fake_optimize_candidates(*, acqf, bounds, config):
        captured["config"] = config
        return torch.tensor([[0.5]], dtype=torch.double), torch.tensor(1.0)

    monkeypatch.setattr(
        optimizer_module,
        "optimize_candidates",
        fake_optimize_candidates,
    )
    return optimizer


def test_nsgaii_name_ignores_optimizer_but_preserves_other_options(monkeypatch) -> None:
    captured: dict[str, object] = {}
    acq_config = AcquisitionConfig(name="NSGA-II")
    optimizer = _candidate_optimizer(
        monkeypatch,
        acq_config=acq_config,
        captured=captured,
    )
    original = OptimizeConfig(
        optimizer="evo",
        q=3,
        optimizer_kwargs={"population_size": 80, "max_gen": 120},
    )

    optimizer.candidate(acq_config=acq_config, opt_config=original)

    resolved = captured["config"]
    assert resolved.optimizer == "nsgaii"
    assert resolved.q == 3
    assert resolved.optimizer_kwargs == {
        "population_size": 80,
        "max_gen": 120,
    }
    assert original.optimizer == "evo"


def test_non_nsgaii_name_keeps_selected_optimizer(monkeypatch) -> None:
    captured: dict[str, object] = {}
    acq_config = AcquisitionConfig(name="ehvi")
    optimizer = _candidate_optimizer(
        monkeypatch,
        acq_config=acq_config,
        captured=captured,
    )
    original = OptimizeConfig(optimizer="evo", q=2)

    optimizer.candidate(acq_config=acq_config, opt_config=original)

    assert captured["config"].optimizer == "evo"
    assert not is_nsgaii_strategy(acq_config)


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
