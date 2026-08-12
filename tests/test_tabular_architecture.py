from __future__ import annotations

import bochan.api as api
import bochan.tabular as tabular
from bochan.tabular import composition_bounds_optimizer
from bochan.tabular import optimizer as legacy_optimizer
from bochan.tabular import optimizer_api


def test_tabular_public_optimizer_has_one_canonical_entry_point() -> None:
    assert tabular.TabularBayesianOptimizer.__module__ == (
        "bochan.tabular.public_optimizer"
    )


def test_composition_bounds_is_component_not_optimizer_layer() -> None:
    assert "TabularBayesianOptimizer" not in vars(composition_bounds_optimizer)
    assert composition_bounds_optimizer.TabularBayesianOptimizer is (
        tabular.TabularBayesianOptimizer
    )
    assert all(
        base.__module__ != "bochan.tabular.composition_bounds_optimizer"
        for base in tabular.TabularBayesianOptimizer.__mro__
    )
    assert isinstance(
        tabular.TabularBayesianOptimizer.composition_bounds_resolver,
        composition_bounds_optimizer.CompositionBoundsResolver,
    )


def test_tabular_import_does_not_patch_core_candidate_method() -> None:
    assert not hasattr(
        api.BayesianOptimizer,
        "_bochan_candidate_before_tabular_outputs",
    )
    assert api.BayesianOptimizer.candidate.__module__ == "bochan.api.optimizer"


def test_tabular_import_does_not_patch_legacy_init_or_predict_methods() -> None:
    assert not getattr(
        optimizer_api.TabularBayesianOptimizer.__init__,
        "_bochan_supports_output_categories",
        False,
    )
    assert not hasattr(
        legacy_optimizer.TabularBayesianOptimizer,
        "_bochan_predict_before_tabular_labels",
    )
