from __future__ import annotations

import pytest
import torch

from bochan.api.configs import CandidateRepairConfig, OptimizeConfig
from bochan.api.factory import optimize_candidates as base_optimize_candidates
from bochan.api.optimizer import dispatch as optimizer_dispatch
from bochan.api.support.best_subset import enumerate_best_subset_supports


def _candidate_for_config(bounds: torch.Tensor, config: OptimizeConfig) -> torch.Tensor:
    d = int(bounds.shape[-1])
    candidate = torch.ones(config.q, d, dtype=bounds.dtype, device=bounds.device)
    for index, value in (config.fixed_features or {}).items():
        candidate[:, int(index)] = float(value)
    return candidate


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


def _support_table_acq(table: dict[tuple[int, ...], float], comp_idx: tuple[int, ...]):
    def acq(X: torch.Tensor) -> torch.Tensor:
        active = tuple(
            index
            for index in comp_idx
            if bool((X[..., index].abs() > 1e-8).any().item())
        )
        return X.new_tensor(table.get(active, -100.0))

    return acq


def test_best_subset_chooses_support_by_optimized_acquisition_value() -> None:
    bounds = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    comp_idx = (0, 1, 2, 3)
    acqf = _support_table_acq(
        {
            (0, 1): 1.0,
            (0, 2): 2.0,
            (0, 3): 3.0,
            (1, 2): 4.0,
            (1, 3): 20.0,
            (2, 3): 5.0,
        },
        comp_idx,
    )
    config = OptimizeConfig(
        q=1,
        optimizer=_callable_optimizer,
        repair_config=CandidateRepairConfig(
            comp_idx=comp_idx,
            k=2,
            support_selection="best_subset",
        ),
    )

    candidates, acq_value = base_optimize_candidates(acqf, bounds, config)

    assert torch.equal(candidates, torch.tensor([[0.0, 1.0, 0.0, 1.0]]))
    assert float(acq_value.item()) == pytest.approx(20.0)


def test_best_subset_q_batch_uses_one_shared_support() -> None:
    bounds = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    comp_idx = (0, 1, 2)
    acqf = _support_table_acq(
        {
            (0, 1): 1.0,
            (0, 2): 3.0,
            (1, 2): 9.0,
        },
        comp_idx,
    )
    config = OptimizeConfig(
        q=2,
        optimizer=_callable_optimizer,
        repair_config=CandidateRepairConfig(
            comp_idx=comp_idx,
            k=2,
            support_selection="best_subset",
        ),
    )

    candidates, acq_value = base_optimize_candidates(acqf, bounds, config)

    assert candidates.shape == (2, 4)
    assert torch.equal(candidates[:, :3], torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]))
    assert torch.equal(candidates[:, 3], torch.ones(2))
    assert float(acq_value.item()) == pytest.approx(9.0)


def test_best_subset_respects_required_and_forbidden_fixed_sparse_dimensions() -> None:
    config = OptimizeConfig(
        fixed_features={0: 0.25, 2: 0.0},
        repair_config=CandidateRepairConfig(
            comp_idx=[0, 1, 2, 3],
            k=2,
            support_selection="best_subset",
        ),
    )

    supports = enumerate_best_subset_supports(config)

    assert supports == [(0, 1), (0, 3)]


def test_best_subset_enforces_exact_enumeration_limit() -> None:
    config = OptimizeConfig(
        optimizer_kwargs={"best_subset_max_combinations": 2},
        repair_config=CandidateRepairConfig(
            comp_idx=[0, 1, 2, 3],
            k=2,
            support_selection="best_subset",
        ),
    )

    with pytest.raises(ValueError, match="would evaluate 6 supports"):
        enumerate_best_subset_supports(config)


def test_best_subset_requires_return_best_only() -> None:
    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    config = OptimizeConfig(
        return_best_only=False,
        optimizer=_callable_optimizer,
        repair_config=CandidateRepairConfig(
            comp_idx=[0, 1, 2],
            k=2,
            support_selection="best_subset",
        ),
    )

    with pytest.raises(ValueError, match="return_best_only=True"):
        base_optimize_candidates(lambda X: X.sum(), bounds, config)


def test_best_subset_rejects_mixed_assignments_on_sparse_dimensions() -> None:
    config = OptimizeConfig(
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
        repair_config=CandidateRepairConfig(
            comp_idx=[0, 1, 2],
            k=2,
            support_selection="best_subset",
        ),
    )

    with pytest.raises(ValueError, match="fixed_features_list entries on k-sparse"):
        enumerate_best_subset_supports(config)


def test_dispatch_applies_best_subset_before_base_backend() -> None:
    bounds = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    comp_idx = (0, 1, 2, 3)
    acqf = _support_table_acq(
        {
            (0, 1): 2.0,
            (0, 2): 1.0,
            (0, 3): 4.0,
            (1, 2): 3.0,
            (1, 3): 12.0,
            (2, 3): 5.0,
        },
        comp_idx,
    )
    seen_supports: list[tuple[int, ...]] = []

    def backend(*, acqf, bounds, config):
        repair = config.repair_config
        assert repair is not None
        seen_supports.append(tuple(int(index) for index in (repair.comp_idx or ())))
        candidate = _candidate_for_config(bounds, config)
        return candidate, acqf(candidate)

    config = OptimizeConfig(
        q=1,
        optimizer="optimize_acqf",
        repair_config=CandidateRepairConfig(
            comp_idx=comp_idx,
            k=2,
            support_selection="best_subset",
        ),
    )

    candidates, acq_value = optimizer_dispatch.optimize_candidates(
        acqf=acqf,
        bounds=bounds,
        config=config,
        base_optimize_candidates=backend,
    )

    assert len(seen_supports) == 6
    assert (1, 3) in seen_supports
    assert torch.equal(candidates, torch.tensor([[0.0, 1.0, 0.0, 1.0]]))
    assert float(acq_value.item()) == pytest.approx(12.0)
