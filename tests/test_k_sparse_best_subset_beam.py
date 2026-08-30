from __future__ import annotations

import pytest
import torch

from bochan.api.configs import CandidateRepairConfig, OptimizeConfig
from bochan.api.factory import optimize_candidates


def _support_table_acq(table: dict[tuple[int, ...], float], comp_idx: tuple[int, ...]):
    def acq(X: torch.Tensor) -> torch.Tensor:
        active = tuple(
            index
            for index in comp_idx
            if bool((X[..., index].abs() > 1e-8).any().item())
        )
        return X.new_tensor(table.get(active, 0.0))

    return acq


def _seeded_callable_optimizer(
    seen: list[tuple[int, ...]],
    *,
    base_values: tuple[float, ...],
):
    def optimizer(
        *,
        acq_function,
        bounds,
        q,
        fixed_features=None,
        post_processing_func=None,
        **kwargs,
    ):
        assert not any(str(key).startswith("best_subset_") for key in kwargs)
        candidate = torch.tensor(base_values, dtype=bounds.dtype, device=bounds.device).repeat(q, 1)
        for index, value in (fixed_features or {}).items():
            candidate[:, int(index)] = float(value)
        if post_processing_func is not None:
            candidate = post_processing_func(candidate)
        active = tuple(
            index
            for index in range(candidate.shape[-1])
            if bool((candidate[..., index].abs() > 1e-8).any().item())
        )
        seen.append(active)
        return candidate, candidate.new_tensor(-999.0)

    return optimizer


def test_best_subset_beam_crosses_one_swap_valley_without_full_enumeration() -> None:
    bounds = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    comp_idx = (0, 1, 2, 3, 4, 5)
    table = {
        (0, 1, 2): 10.0,
        (0, 1, 3): 9.0,
        (0, 2, 3): 8.0,
        (1, 2, 3): 7.0,
        (0, 3, 4): 30.0,
    }
    seen: list[tuple[int, ...]] = []
    config = OptimizeConfig(
        q=1,
        optimizer=_seeded_callable_optimizer(seen, base_values=(0.9, 0.8, 0.7, 0.6, 0.5, 0.4)),
        optimizer_kwargs={
            "best_subset_strategy": "beam",
            "best_subset_beam_width": 2,
            "best_subset_beam_steps": 2,
            "best_subset_max_evaluations": 11,
        },
        repair_config=CandidateRepairConfig(
            comp_idx=comp_idx,
            k=3,
            score="value",
            support_selection="best_subset",
        ),
    )

    candidates, acq_value = optimize_candidates(
        _support_table_acq(table, comp_idx),
        bounds,
        config,
    )

    assert tuple(torch.nonzero(candidates[0] > 1e-8).flatten().tolist()) == (0, 3, 4)
    assert float(acq_value.item()) == pytest.approx(30.0)
    evaluated_exact_supports = [support for support in seen if len(support) == 3]
    assert len(evaluated_exact_supports) <= 11
    assert len(set(evaluated_exact_supports)) < 20  # 6C3 exact enumeration would evaluate 20 supports.


def test_best_subset_auto_keeps_exact_search_for_small_support_space() -> None:
    bounds = torch.tensor([[0.0] * 4, [1.0] * 4])
    comp_idx = (0, 1, 2, 3)
    seen: list[tuple[int, ...]] = []
    config = OptimizeConfig(
        optimizer=_seeded_callable_optimizer(seen, base_values=(0.9, 0.8, 0.7, 0.6)),
        optimizer_kwargs={
            "best_subset_strategy": "auto",
            "best_subset_max_combinations": 10,
        },
        repair_config=CandidateRepairConfig(
            comp_idx=comp_idx,
            k=2,
            support_selection="best_subset",
        ),
    )

    optimize_candidates(_support_table_acq({(2, 3): 5.0}, comp_idx), bounds, config)

    exact_supports = [support for support in seen if len(support) == 2]
    assert len(exact_supports) == 6
    assert all(len(support) != 4 for support in seen)


def test_best_subset_auto_switches_to_beam_above_exact_limit() -> None:
    bounds = torch.tensor([[0.0] * 6, [1.0] * 6])
    comp_idx = (0, 1, 2, 3, 4, 5)
    seen: list[tuple[int, ...]] = []
    config = OptimizeConfig(
        optimizer=_seeded_callable_optimizer(seen, base_values=(0.9, 0.8, 0.7, 0.6, 0.5, 0.4)),
        optimizer_kwargs={
            "best_subset_strategy": "auto",
            "best_subset_max_combinations": 5,
            "best_subset_beam_width": 2,
            "best_subset_beam_steps": 1,
            "best_subset_max_evaluations": 3,
        },
        repair_config=CandidateRepairConfig(
            comp_idx=comp_idx,
            k=3,
            score="value",
            support_selection="best_subset",
        ),
    )

    optimize_candidates(_support_table_acq({}, comp_idx), bounds, config)

    assert any(len(support) == 6 for support in seen)  # top-k seed optimization
    exact_supports = [support for support in seen if len(support) == 3]
    assert 1 <= len(exact_supports) <= 3


def test_best_subset_beam_respects_support_evaluation_budget() -> None:
    bounds = torch.tensor([[0.0] * 7, [1.0] * 7])
    comp_idx = tuple(range(7))
    seen: list[tuple[int, ...]] = []
    config = OptimizeConfig(
        optimizer=_seeded_callable_optimizer(seen, base_values=(0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)),
        optimizer_kwargs={
            "best_subset_strategy": "beam",
            "best_subset_beam_width": 4,
            "best_subset_beam_steps": 10,
            "best_subset_max_evaluations": 4,
        },
        repair_config=CandidateRepairConfig(
            comp_idx=comp_idx,
            k=3,
            score="value",
            support_selection="best_subset",
        ),
    )

    optimize_candidates(_support_table_acq({}, comp_idx), bounds, config)

    exact_supports = [support for support in seen if len(support) == 3]
    assert len(exact_supports) == 4


def test_best_subset_beam_keeps_one_support_shared_across_q_batch() -> None:
    bounds = torch.tensor([[0.0] * 5, [1.0] * 5])
    comp_idx = (0, 1, 2, 3, 4)
    table = {
        (0, 1): 2.0,
        (0, 2): 1.0,
        (1, 2): 8.0,
    }
    seen: list[tuple[int, ...]] = []
    config = OptimizeConfig(
        q=2,
        optimizer=_seeded_callable_optimizer(seen, base_values=(0.9, 0.8, 0.7, 0.6, 0.5)),
        optimizer_kwargs={
            "best_subset_strategy": "beam",
            "best_subset_beam_width": 2,
            "best_subset_beam_steps": 1,
            "best_subset_max_evaluations": 5,
        },
        repair_config=CandidateRepairConfig(
            comp_idx=comp_idx,
            k=2,
            score="value",
            support_selection="best_subset",
        ),
    )

    candidates, acq_value = optimize_candidates(
        _support_table_acq(table, comp_idx),
        bounds,
        config,
    )

    active_by_row = [
        tuple(torch.nonzero(row > 1e-8).flatten().tolist())
        for row in candidates
    ]
    assert active_by_row[0] == active_by_row[1]
    assert float(acq_value.item()) == pytest.approx(table.get(active_by_row[0], 0.0))


def test_best_subset_rejects_unknown_strategy_and_invalid_beam_settings() -> None:
    bounds = torch.tensor([[0.0] * 4, [1.0] * 4])
    base_config = dict(
        optimizer=_seeded_callable_optimizer([], base_values=(0.9, 0.8, 0.7, 0.6)),
        repair_config=CandidateRepairConfig(
            comp_idx=(0, 1, 2, 3),
            k=2,
            support_selection="best_subset",
        ),
    )

    with pytest.raises(ValueError, match="best_subset_strategy"):
        optimize_candidates(
            lambda X: X.sum(),
            bounds,
            OptimizeConfig(**base_config, optimizer_kwargs={"best_subset_strategy": "unknown"}),
        )

    with pytest.raises(ValueError, match="best_subset_beam_width"):
        optimize_candidates(
            lambda X: X.sum(),
            bounds,
            OptimizeConfig(
                **base_config,
                optimizer_kwargs={"best_subset_strategy": "beam", "best_subset_beam_width": 0},
            ),
        )
