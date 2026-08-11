from __future__ import annotations

import bochan.api as api
from bochan.api import acquisition_service, engine, factory, optimizer


def test_public_bayesian_optimizer_has_one_canonical_definition() -> None:
    assert api.BayesianOptimizer is optimizer.BayesianOptimizer
    assert api.BayesianOptimizer.__module__ == "bochan.api.optimizer"
    assert issubclass(api.BayesianOptimizer, engine.BayesianOptimizer)


def test_api_import_does_not_replace_engine_or_factory_symbols() -> None:
    assert engine.BayesianOptimizer is not api.BayesianOptimizer
    assert engine.BayesianOptimizer.__module__ == "bochan.api.engine"
    assert factory.build_acquisition.__module__ == "bochan.api.factory"
    assert acquisition_service.build_acquisition.__module__ == (
        "bochan.api.acquisition_service"
    )


def test_canonical_optimizer_owns_observation_and_candidate_entry_points() -> None:
    methods = {
        "fit",
        "fit_observations",
        "refit",
        "tell",
        "tell_observations",
        "acquisition",
        "candidate",
        "ask",
    }
    assert methods.issubset(vars(api.BayesianOptimizer))
