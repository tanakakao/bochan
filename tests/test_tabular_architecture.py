from __future__ import annotations

import importlib.util

import bochan.api as api
import bochan.tabular as tabular
from bochan.tabular import (
    composition_bounds_optimizer,
    element_column_composition_optimizer,
    optimizer_api,
)
from bochan.tabular import optimizer as legacy_optimizer
from bochan.tabular.composition_element_column_transform import (
    CompositionElementColumnTransform,
)
from bochan.tabular.composition_element_constraint_candidates import (
    CompositionElementConstraintCandidateReranker,
)
from bochan.tabular.composition_element_constraints import (
    CompositionElementConstraintProjector,
    CompositionElementConstraintResolver,
)
from bochan.tabular.composition_total_constraints import (
    CompositionTotalConstraintResolver,
)
from bochan.tabular.composition_variable_total_transform import (
    CompositionVariableTotalTransform,
)


def test_tabular_public_optimizer_has_one_canonical_entry_point() -> None:
    assert tabular.TabularBayesianOptimizer.__module__ == "bochan.tabular.public_optimizer"


def test_composition_bounds_is_component_not_optimizer_layer() -> None:
    assert "TabularBayesianOptimizer" not in vars(composition_bounds_optimizer)
    assert not hasattr(composition_bounds_optimizer, "TabularBayesianOptimizer")
    assert all(
        base.__module__ != "bochan.tabular.composition_bounds_optimizer"
        for base in tabular.TabularBayesianOptimizer.__mro__
    )
    assert isinstance(
        tabular.TabularBayesianOptimizer.composition_bounds_resolver,
        composition_bounds_optimizer.CompositionBoundsResolver,
    )


def test_element_columns_use_explicit_transform_component() -> None:
    assert isinstance(
        tabular.TabularBayesianOptimizer.composition_element_column_transform,
        CompositionElementColumnTransform,
    )
    adapter = element_column_composition_optimizer.TabularBayesianOptimizer
    assert "_with_internal_formula_columns" not in vars(adapter)
    assert "_numeric_site_values" not in vars(adapter)
    assert "_site_source_columns" not in vars(adapter)
    assert adapter._prepare_multi_site_frame.__module__ == (
        "bochan.tabular.element_column_composition_optimizer"
    )


def test_composition_total_constraints_use_explicit_resolver() -> None:
    assert isinstance(
        tabular.TabularBayesianOptimizer.composition_total_constraint_resolver,
        CompositionTotalConstraintResolver,
    )
    assert (
        tabular.TabularBayesianOptimizer._normalize_total_constraints.__func__
        is not None
    )
    assert (
        tabular.TabularBayesianOptimizer._normalize_total_constraints.__module__
        == "bochan.tabular.public_optimizer"
    )


def test_variable_total_behavior_uses_explicit_transform_component() -> None:
    assert isinstance(
        tabular.TabularBayesianOptimizer.composition_variable_total_transform,
        CompositionVariableTotalTransform,
    )
    assert (
        tabular.TabularBayesianOptimizer._prepare_multi_site_frame.__module__
        == "bochan.tabular.public_optimizer"
    )
    assert all(
        cls.__module__ != "bochan.tabular.variable_total_composition_optimizer"
        for cls in tabular.TabularBayesianOptimizer.__mro__
    )
    assert (
        tabular.TabularBayesianOptimizer.inverse_compositions.__module__
        == "bochan.tabular.public_optimizer"
    )


def test_element_constraints_use_explicit_components_without_optimizer_layer() -> None:
    assert isinstance(
        tabular.TabularBayesianOptimizer.composition_element_constraint_resolver,
        CompositionElementConstraintResolver,
    )
    assert isinstance(
        tabular.TabularBayesianOptimizer.composition_element_constraint_candidate_reranker,
        CompositionElementConstraintCandidateReranker,
    )
    assert CompositionElementConstraintProjector.__module__ == (
        "bochan.tabular.composition_element_constraints"
    )
    assert importlib.util.find_spec(
        "bochan.tabular.element_constraint_composition_optimizer"
    ) is None
    assert all(
        cls.__module__ != "bochan.tabular.element_constraint_composition_optimizer"
        for cls in tabular.TabularBayesianOptimizer.__mro__
    )
    assert (
        tabular.TabularBayesianOptimizer.inverse_compositions.__module__
        == "bochan.tabular.public_optimizer"
    )
    assert (
        tabular.TabularBayesianOptimizer.candidate.__module__
        == "bochan.tabular.public_optimizer"
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
