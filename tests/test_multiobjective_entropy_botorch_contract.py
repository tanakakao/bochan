from __future__ import annotations

import inspect

from botorch.acquisition.multi_objective.joint_entropy_search import (
    qLowerBoundMultiObjectiveJointEntropySearch,
)
from botorch.acquisition.multi_objective.max_value_entropy_search import (
    qLowerBoundMultiObjectiveMaxValueEntropySearch,
)
from botorch.acquisition.multi_objective.utils import (
    compute_sample_box_decomposition,
    random_search_optimizer,
    sample_optimal_points,
)


def test_botorch_multiobjective_entropy_utilities_are_public() -> None:
    assert callable(sample_optimal_points)
    assert callable(random_search_optimizer)
    assert callable(compute_sample_box_decomposition)


def test_botorch_mo_mes_constructor_contract() -> None:
    parameters = inspect.signature(
        qLowerBoundMultiObjectiveMaxValueEntropySearch
    ).parameters
    assert {
        "model",
        "hypercell_bounds",
        "X_pending",
        "estimation_type",
        "num_samples",
    } <= set(parameters)


def test_botorch_mo_jes_constructor_contract() -> None:
    parameters = inspect.signature(
        qLowerBoundMultiObjectiveJointEntropySearch
    ).parameters
    assert {
        "model",
        "pareto_sets",
        "pareto_fronts",
        "hypercell_bounds",
        "X_pending",
        "estimation_type",
        "num_samples",
    } <= set(parameters)


def test_botorch_pareto_sampling_contract() -> None:
    parameters = inspect.signature(sample_optimal_points).parameters
    assert {
        "model",
        "bounds",
        "num_samples",
        "num_points",
        "optimizer",
        "maximize",
        "optimizer_kwargs",
    } <= set(parameters)


def test_botorch_box_decomposition_contract() -> None:
    parameters = inspect.signature(compute_sample_box_decomposition).parameters
    assert {"pareto_fronts", "maximize"} <= set(parameters)
