from __future__ import annotations

import pytest
import torch

from bochan.api.configs import CandidateRepairConfig, OptimizeConfig
from bochan.api.factory import optimize_candidates as base_optimize_candidates
from bochan.api.support.multi_group_best_subset import (
    BEST_SUBSET_GROUPS_KWARG,
    enumerate_grouped_best_subset_supports,
)


def _callable_optimizer(
    *,
    acq_function,
    bounds,
    q,
    fixed_features=None,
    post_processing_func=None,
    **kwargs,
):
    candidate = torch.ones(q, bounds.shape[-1], dtype=bounds.dtype, device=bounds.device)
    for index, value in (fixed_features or {}).items():
        candidate[:, int(index)] = float(value)
    if post_processing_func is not None:
        candidate = post_processing_func(candidate)
    return candidate, acq_function(candidate)


def _support_table_acq(
    table: dict[tuple[int, ...], float],
    sparse_indices: tuple[int, ...],
):
    def acq(X: torch.Tensor) -> torch.Tensor:
        active = tuple(
            index
            for index in sparse_indices
            if bool((X[..., index].abs() > 1e-8).any().item())
        )
        return X.new_tensor(table.get(active, -100.0))

    return acq


def _config(
    groups,
    *,
    q: int = 1,
    optimizer_kwargs=None,
    fixed_features=None,
) -> OptimizeConfig:
    sparse = [index for group in groups for index in group["comp_idx"]]
    kwargs = dict(optimizer_kwargs or {})
    kwargs[BEST_SUBSET_GROUPS_KWARG] = groups
    return OptimizeConfig(
        q=q,
        optimizer=_callable_optimizer,
        optimizer_kwargs=kwargs,
        fixed_features=fixed_features,
        repair_config=CandidateRepairConfig(
            comp_idx=sparse,
            k=sum(int(group.get("max_k", group.get("k", 0))) for group in groups),
            support_selection="best_subset",
        ),
    )


def test_grouped_exact_enumerates_cartesian_product() -> None:
    config = _config(
        [
            {"name": "a", "comp_idx": [0, 1, 2], "k": 1},
            {"name": "b", "comp_idx": [3, 4, 5], "k": 2},
        ]
    )

    supports = enumerate_grouped_best_subset_supports(config)

    assert len(supports) == 9
    assert len(set(supports)) == 9
    assert all(sum(index in {0, 1, 2} for index in support) == 1 for support in supports)
    assert all(sum(index in {3, 4, 5} for index in support) == 2 for support in supports)


def test_grouped_variable_cardinality_count_is_product_of_group_counts() -> None:
    config = _config(
        [
            {"name": "a", "comp_idx": [0, 1, 2], "min_k": 1, "max_k": 2},
            {"name": "b", "comp_idx": [3, 4], "k": 1},
        ]
    )

    supports = enumerate_grouped_best_subset_supports(config)

    assert len(supports) == 12
    assert {sum(index in {0, 1, 2} for index in support) for support in supports} == {1, 2}
    assert all(sum(index in {3, 4} for index in support) == 1 for support in supports)


def test_grouped_best_subset_chooses_joint_support_for_q_batch() -> None:
    bounds = torch.tensor([[0.0] * 7, [1.0] * 7])
    groups = [
        {"name": "a", "comp_idx": [0, 1, 2], "k": 1},
        {"name": "b", "comp_idx": [3, 4, 5], "k": 1},
    ]
    acqf = _support_table_acq(
        {
            (0, 3): 1.0,
            (0, 4): 2.0,
            (0, 5): 3.0,
            (1, 3): 4.0,
            (1, 4): 5.0,
            (1, 5): 6.0,
            (2, 3): 7.0,
            (2, 4): 20.0,
            (2, 5): 8.0,
        },
        (0, 1, 2, 3, 4, 5),
    )
    config = _config(groups, q=2)

    candidates, value = base_optimize_candidates(acqf, bounds, config)

    expected = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        ]
    )
    assert torch.equal(candidates, expected)
    assert float(value.item()) == pytest.approx(20.0)


def test_grouped_best_subset_respects_fixed_required_and_forbidden() -> None:
    groups = [
        {"name": "a", "comp_idx": [0, 1, 2], "k": 2},
        {"name": "b", "comp_idx": [3, 4, 5], "k": 1},
    ]
    config = _config(groups, fixed_features={0: 0.25, 2: 0.0, 4: 0.0})

    supports = enumerate_grouped_best_subset_supports(config)

    assert supports == [(0, 1, 3), (0, 1, 5)]


def test_grouped_best_subset_rejects_overlapping_groups() -> None:
    config = _config(
        [
            {"name": "a", "comp_idx": [0, 1], "k": 1},
            {"name": "b", "comp_idx": [1, 2], "k": 1},
        ]
    )

    with pytest.raises(ValueError, match="must be disjoint"):
        enumerate_grouped_best_subset_supports(config)


def test_grouped_best_subset_rejects_global_cardinality_kwargs() -> None:
    config = _config(
        [
            {"name": "a", "comp_idx": [0, 1], "k": 1},
            {"name": "b", "comp_idx": [2, 3], "k": 1},
        ],
        optimizer_kwargs={"best_subset_min_k": 1},
    )

    with pytest.raises(ValueError, match="ambiguous"):
        enumerate_grouped_best_subset_supports(config)


def test_grouped_exact_limit_uses_cartesian_product_count() -> None:
    config = _config(
        [
            {"name": "a", "comp_idx": [0, 1, 2], "k": 1},
            {"name": "b", "comp_idx": [3, 4, 5], "k": 1},
        ],
        optimizer_kwargs={"best_subset_max_combinations": 8},
    )

    with pytest.raises(ValueError, match="9 support combinations"):
        enumerate_grouped_best_subset_supports(config)


def test_grouped_beam_changes_one_group_at_a_time_to_reach_joint_optimum() -> None:
    bounds = torch.tensor([[0.0] * 6, [1.0] * 6])
    groups = [
        {"name": "a", "comp_idx": [0, 1, 2], "k": 1},
        {"name": "b", "comp_idx": [3, 4, 5], "k": 1},
    ]
    acqf = _support_table_acq(
        {
            (0, 3): 0.0,
            (2, 3): 5.0,
            (0, 5): 4.0,
            (2, 5): 20.0,
        },
        (0, 1, 2, 3, 4, 5),
    )
    config = _config(
        groups,
        optimizer_kwargs={
            "best_subset_strategy": "beam",
            "best_subset_beam_width": 2,
            "best_subset_beam_steps": 2,
            "best_subset_max_evaluations": 20,
        },
    )

    candidates, value = base_optimize_candidates(acqf, bounds, config)

    assert torch.equal(candidates, torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]]))
    assert float(value.item()) == pytest.approx(20.0)


def test_grouped_auto_can_switch_to_beam_from_product_count() -> None:
    bounds = torch.tensor([[0.0] * 6, [1.0] * 6])
    groups = [
        {"name": "a", "comp_idx": [0, 1, 2], "k": 1},
        {"name": "b", "comp_idx": [3, 4, 5], "k": 1},
    ]
    acqf = _support_table_acq({(0, 3): 1.0}, (0, 1, 2, 3, 4, 5))
    config = _config(
        groups,
        optimizer_kwargs={
            "best_subset_strategy": "auto",
            "best_subset_max_combinations": 4,
            "best_subset_beam_width": 2,
            "best_subset_beam_steps": 1,
            "best_subset_max_evaluations": 12,
        },
    )

    candidates, value = base_optimize_candidates(acqf, bounds, config)

    assert candidates.shape == (1, 6)
    assert float(value.item()) == pytest.approx(1.0)
