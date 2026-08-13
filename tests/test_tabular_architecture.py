from __future__ import annotations

import importlib.util

import bochan.api as api
import bochan.tabular as tabular
from bochan.tabular import (
    composition_bounds_optimizer,
    multi_site_composition_optimizer,
    optimizer_api,
)
from bochan.tabular import optimizer as tabular_optimizer
from bochan.tabular.composition_element_columns import CompositionElementColumnTransform
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
from bochan.tabular.multi_site_composition_optimizer import MultiSiteCompositionMixin
from bochan.tabular.observation_optimizer import ObservationTabularMixin
from bochan.tabular.optimizer_api import TabularApiMixin


def test_tabular_public_optimizer_has_one_canonical_entry_point() -> None:
    assert tabular.TabularBayesianOptimizer.__module__ == "bochan.tabular.public_optimizer"


def test_internal_behavior_modules_do_not_define_optimizer_classes() -> None:
    assert "TabularBayesianOptimizer" not in vars(optimizer_api)
    assert "TabularBayesianOptimizer" not in vars(multi_site_composition_optimizer)
    assert importlib.util.find_spec("bochan.tabular.composition_optimizer") is None


def test_canonical_optimizer_composes_functional_mixins_and_one_core() -> None:
    mro = tabular.TabularBayesianOptimizer.__mro__
    assert ObservationTabularMixin in mro
    assert MultiSiteCompositionMixin in mro
    assert TabularApiMixin in mro
    assert tabular_optimizer.TabularBayesianOptimizer in mro
    assert sum(cls.__name__ == "TabularBayesianOptimizer" for cls in mro) == 2


def test_composition_bounds_is_component_not_optimizer_layer() -> None:
    assert "TabularBayesianOptimizer" not in vars(composition_bounds_optimizer)
    assert all(
        base.__module__ != "bochan.tabular.composition_bounds_optimizer"
        for base in tabular.TabularBayesianOptimizer.__mro__
    )
    assert isinstance(
        tabular.TabularBayesianOptimizer.composition_bounds_resolver,
        composition_bounds_optimizer.CompositionBoundsResolver,
    )


def test_element_columns_use_explicit_component_without_optimizer_layer() -> None:
    assert isinstance(
        tabular.TabularBayesianOptimizer.composition_element_column_transform,
        CompositionElementColumnTransform,
    )
    assert importlib.util.find_spec(
        "bochan.tabular.element_column_composition_optimizer"
    ) is None
    assert all(
        cls.__module__ != "bochan.tabular.element_column_composition_optimizer"
        for cls in tabular.TabularBayesianOptimizer.__mro__
    )
    assert (
        tabular.TabularBayesianOptimizer._prepare_multi_site_frame.__module__
        == "bochan.tabular.public_optimizer"
    )
    assert tabular.TabularBayesianOptimizer.fit.__module__ == "bochan.tabular.public_optimizer"
    assert (
        tabular.TabularBayesianOptimizer.inverse_compositions.__module__
        == "bochan.tabular.public_optimizer"
    )


def test_composition_total_constraints_use_explicit_resolver() -> None:
    assert isinstance(
        tabular.TabularBayesianOptimizer.composition_total_constraint_resolver,
        CompositionTotalConstraintResolver,
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
        tabular.TabularBayesianOptimizer.candidate.__module__
        == "bochan.tabular.public_optimizer"
    )


def test_tabular_import_keeps_core_candidate_identity() -> None:
    sentinel_name = "_bochan_" + "candidate_before_tabular_outputs"
    assert not hasattr(api.BayesianOptimizer, sentinel_name)
    assert api.BayesianOptimizer.candidate.__module__ == "bochan.api.optimizer"


def test_tabular_import_keeps_internal_method_identity() -> None:
    init_sentinel = "_bochan_" + "supports_output_categories"
    predict_sentinel = "_bochan_" + "predict_before_tabular_labels"
    assert not getattr(TabularApiMixin.__init__, init_sentinel, False)
    assert not hasattr(tabular_optimizer.TabularBayesianOptimizer, predict_sentinel)
