from __future__ import annotations

import bochan.api as api
import bochan.tabular as tabular
from bochan.tabular.composition import CompositionAdapter
from bochan.tabular.observation import ObservationAdapter
from bochan.tabular.optimizer.candidates import CandidateService
from bochan.tabular.optimizer.core import TabularBayesianOptimizer
from bochan.tabular.optimizer.diagnostics import DiagnosticsService


def test_tabular_public_optimizer_has_one_canonical_entry_point() -> None:
    assert tabular.TabularBayesianOptimizer is TabularBayesianOptimizer
    assert TabularBayesianOptimizer.__module__ == "bochan.tabular.optimizer.core"


def test_canonical_optimizer_has_no_functional_mixin_inheritance() -> None:
    assert TabularBayesianOptimizer.__mro__ == (TabularBayesianOptimizer, object)


def test_optimizer_uses_explicit_domain_components() -> None:
    optimizer = TabularBayesianOptimizer(
        task_type="regression",
        model_type="base",
        input_cols=["x"],
        target_cols=["y"],
    )
    assert isinstance(optimizer.composition, CompositionAdapter)
    assert isinstance(optimizer.observation, ObservationAdapter)
    assert isinstance(optimizer.candidates, CandidateService)
    assert isinstance(optimizer.diagnostics, DiagnosticsService)


def test_lower_level_components_do_not_define_optimizer_facades() -> None:
    import bochan.tabular.composition.adapter as composition_adapter
    import bochan.tabular.observation.adapter as observation_adapter
    import bochan.tabular.optimizer.candidates as candidates
    import bochan.tabular.optimizer.diagnostics as diagnostics

    for module in (
        composition_adapter,
        observation_adapter,
        candidates,
        diagnostics,
    ):
        assert "TabularBayesianOptimizer" not in vars(module)


def test_tabular_import_keeps_core_candidate_identity() -> None:
    sentinel_name = "_bochan_" + "candidate_before_tabular_outputs"
    assert not hasattr(api.BayesianOptimizer, sentinel_name)
    assert api.BayesianOptimizer.candidate.__module__ == "bochan.api.optimizer"


def test_tabular_import_does_not_patch_core_prediction() -> None:
    sentinel_name = "_bochan_" + "predict_before_tabular_labels"
    assert not hasattr(api.BayesianOptimizer, sentinel_name)
