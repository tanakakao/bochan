from __future__ import annotations

import pytest
import torch

from bochan.api.configs import CandidateRepairConfig, OptimizeConfig
from bochan.api.factory import optimize_candidates
from bochan.api.support.best_subset import enumerate_best_subset_supports


def _support_table_acq(
    table: dict[tuple[int, ...], float],
    comp_idx: tuple[int, ...],
):
    def acq(X: torch.Tensor) -> torch.Tensor:
        active = tuple(
            index
            for index in comp_idx
            if bool((X[..., index].abs() > 1e-8).any().item())
        )
        return X.new_tensor(table.get(active, -100.0))

    return acq


def _callable_optimizer(
    *,
    acq_function,
    bounds,
    q,
    fixed_features=None,
    post_processing_func=None,
    **kwargs,
):
    assert "best_subset_min_k" not in kwargs
    assert "best_subset_max_k" not in kwargs
    candidate = torch.ones(q, bounds.shape[-1], dtype=bounds.dtype, device=bounds.device)
    for index, value in (fixed_features or {}).items():
        candidate[:, int(index)] = float(value)
    if post_processing_func is not None:
        candidate = post_processing_func(candidate)
    return candidate, acq_function(candidate)


def _recording_optimizer(seen: list[tuple[int, ...]]):
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
        base = torch.linspace(
            1.0,
            0.2,
            bounds.shape[-1],
            dtype=bounds.dtype,
            device=bounds.device,
        )
        candidate = base.repeat(q, 1)
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
        return candidate, acq_function(candidate)

    return optimizer


def _range_config(
    *,
    comp_idx: tuple[int, ...],
    minimum: int,
    maximum: int,
    strategy: str = "exact",
    **optimizer_kwargs,
) -> OptimizeConfig:
    return OptimizeConfig(
        optimizer=_callable_optimizer,
        optimizer_kwargs={
            "best_subset_strategy": strategy,
            "best_subset_min_k": minimum,
            "best_subset_max_k": maximum,
            **optimizer_kwargs,
        },
        repair_config=CandidateRepairConfig(
            comp_idx=comp_idx,
            k=maximum,
            support_selection="best_subset",
        ),
    )


def test_best_subset_range_enumerates_every_allowed_cardinality() -> None:
    config = _range_config(
        comp_idx=(0, 1, 2, 3),
        minimum=1,
        maximum=3,
    )

    supports = enumerate_best_subset_supports(config)

    assert len(supports) == 14  # 4C1 + 4C2 + 4C3
    assert [len(support) for support in supports].count(1) == 4
    assert [len(support) for support in supports].count(2) == 6
    assert [len(support) for support in supports].count(3) == 4


def test_best_subset_range_preserves_legacy_exact_k_default() -> None:
    config = OptimizeConfig(
        repair_config=CandidateRepairConfig(
            comp_idx=(0, 1, 2, 3),
            k=2,
            support_selection="best_subset",
        )
    )

    supports = enumerate_best_subset_supports(config)

    assert len(supports) == 6
    assert all(len(support) == 2 for support in supports)


def test_best_subset_range_respects_required_and_forbidden_dimensions() -> None:
    config = _range_config(
        comp_idx=(0, 1, 2, 3),
        minimum=1,
        maximum=3,
    )
    config.fixed_features = {0: 0.25, 3: 0.0}

    supports = enumerate_best_subset_supports(config)

    assert supports == [
        (0,),
        (0, 1),
        (0, 2),
        (0, 1, 2),
    ]


def test_best_subset_range_uses_summed_exact_enumeration_limit() -> None:
    config = _range_config(
        comp_idx=(0, 1, 2, 3),
        minimum=1,
        maximum=3,
        best_subset_max_combinations=13,
    )

    with pytest.raises(ValueError, match="would evaluate 14 supports"):
        enumerate_best_subset_supports(config)


def test_best_subset_range_selects_cardinality_by_acquisition_value() -> None:
    bounds = torch.tensor([[0.0] * 4, [1.0] * 4])
    comp_idx = (0, 1, 2, 3)
    config = _range_config(
        comp_idx=comp_idx,
        minimum=1,
        maximum=3,
    )
    acqf = _support_table_acq(
        {
            (0,): 1.0,
            (1, 3): 50.0,
            (0, 1, 2): 10.0,
        },
        comp_idx,
    )

    candidates, acq_value = optimize_candidates(acqf, bounds, config)

    active = tuple(torch.nonzero(candidates[0] > 1e-8).flatten().tolist())
    assert active == (1, 3)
    assert float(acq_value.item()) == pytest.approx(50.0)


def test_best_subset_range_beam_seeds_each_cardinality_and_compares_them() -> None:
    bounds = torch.tensor([[0.0] * 5, [1.0] * 5])
    comp_idx = (0, 1, 2, 3, 4)
    seen: list[tuple[int, ...]] = []
    config = OptimizeConfig(
        optimizer=_recording_optimizer(seen),
        optimizer_kwargs={
            "best_subset_strategy": "beam",
            "best_subset_min_k": 1,
            "best_subset_max_k": 3,
            "best_subset_beam_width": 3,
            "best_subset_beam_steps": 1,
            "best_subset_max_evaluations": 8,
        },
        repair_config=CandidateRepairConfig(
            comp_idx=comp_idx,
            k=3,
            score="value",
            support_selection="best_subset",
        ),
    )
    acqf = _support_table_acq(
        {
            (0,): 1.0,
            (0, 1): 5.0,
            (0, 1, 2): 30.0,
        },
        comp_idx,
    )

    candidates, acq_value = optimize_candidates(acqf, bounds, config)

    active = tuple(torch.nonzero(candidates[0] > 1e-8).flatten().tolist())
    assert active == (0, 1, 2)
    assert float(acq_value.item()) == pytest.approx(30.0)
    assert {1, 2, 3}.issubset({len(support) for support in seen})


def test_best_subset_range_beam_budget_must_seed_every_cardinality() -> None:
    bounds = torch.tensor([[0.0] * 5, [1.0] * 5])
    config = OptimizeConfig(
        optimizer=_recording_optimizer([]),
        optimizer_kwargs={
            "best_subset_strategy": "beam",
            "best_subset_min_k": 1,
            "best_subset_max_k": 3,
            "best_subset_max_evaluations": 2,
        },
        repair_config=CandidateRepairConfig(
            comp_idx=(0, 1, 2, 3, 4),
            k=3,
            support_selection="best_subset",
        ),
    )

    with pytest.raises(ValueError, match="at least the number of allowed cardinalities"):
        optimize_candidates(lambda X: X.sum(), bounds, config)


def test_best_subset_range_rejects_invalid_cardinality_bounds() -> None:
    config = _range_config(
        comp_idx=(0, 1, 2),
        minimum=3,
        maximum=2,
    )

    with pytest.raises(ValueError, match="best_subset_max_k"):
        enumerate_best_subset_supports(config)
