from __future__ import annotations

import pytest
import torch

from bochan.api import CandidateRepairConfig, OptimizeConfig
from bochan.api.support.best_subset import (
    InfeasibleBestSubsetSupportError,
    optimize_best_subset_candidates,
)


def _config() -> OptimizeConfig:
    return OptimizeConfig(
        repair_config=CandidateRepairConfig(
            comp_idx=[0, 1, 2],
            k=2,
            support_selection="best_subset",
        )
    )


def _candidate(bounds: torch.Tensor, config: OptimizeConfig) -> torch.Tensor:
    candidate = torch.ones(1, bounds.shape[-1], dtype=bounds.dtype)
    for index, value in (config.fixed_features or {}).items():
        candidate[:, int(index)] = float(value)
    return candidate


def test_exact_best_subset_skips_only_explicitly_infeasible_supports() -> None:
    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    seen: list[tuple[int, ...]] = []

    def optimize_one(*, acqf, bounds, config):
        repair = config.repair_config
        assert repair is not None
        support = tuple(int(index) for index in repair.comp_idx or ())
        seen.append(support)
        if support == (0, 1):
            raise InfeasibleBestSubsetSupportError("support-specific infeasibility")
        candidate = _candidate(bounds, config)
        return candidate, acqf(candidate)

    def acqf(candidate: torch.Tensor) -> torch.Tensor:
        active = tuple(
            index
            for index in range(3)
            if bool((candidate[..., index].abs() > 1e-8).any())
        )
        return candidate.new_tensor({(0, 2): 3.0, (1, 2): 7.0}.get(active, -1.0))

    candidates, value = optimize_best_subset_candidates(
        acqf=acqf,
        bounds=bounds,
        config=_config(),
        optimize_one=optimize_one,
    )

    assert seen == [(0, 1), (0, 2), (1, 2)]
    torch.testing.assert_close(candidates, torch.tensor([[0.0, 1.0, 1.0]]))
    assert float(value.item()) == pytest.approx(7.0)


def test_exact_best_subset_fails_when_every_support_is_infeasible() -> None:
    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    def optimize_one(**_kwargs):
        raise InfeasibleBestSubsetSupportError("no feasible point")

    with pytest.raises(
        InfeasibleBestSubsetSupportError,
        match="did not produce any feasible support",
    ):
        optimize_best_subset_candidates(
            acqf=lambda candidate: candidate.sum(),
            bounds=bounds,
            config=_config(),
            optimize_one=optimize_one,
        )


def test_unrelated_optimizer_errors_are_not_swallowed() -> None:
    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    def optimize_one(**_kwargs):
        raise RuntimeError("backend bug")

    with pytest.raises(RuntimeError, match="backend bug"):
        optimize_best_subset_candidates(
            acqf=lambda candidate: candidate.sum(),
            bounds=bounds,
            config=_config(),
            optimize_one=optimize_one,
        )
