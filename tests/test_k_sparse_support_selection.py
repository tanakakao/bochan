from __future__ import annotations

from typing import get_args

import pytest
import torch

from bochan.api.configs.base import SupportSelection as ConfigSupportSelection
from bochan.constraints.k_sparse import make_k_sparse_linear_constraints_repair
from bochan.constraints.support import (
    sample_k_without_replacement,
    select_support_mask,
)


def test_support_selection_config_exposes_best_subset() -> None:
    assert get_args(ConfigSupportSelection) == ("topk", "sample", "best_subset")


def test_topk_support_selection_preserves_existing_score_modes() -> None:
    group = torch.tensor([[-0.8, 0.7, 0.2, -0.1]])

    abs_mask = select_support_mask(
        group,
        k=2,
        score="abs",
        support_selection="topk",
        sample_tau=0.2,
        sample_eps=0.05,
        generator=None,
    )
    value_mask = select_support_mask(
        group,
        k=2,
        score="value",
        support_selection="topk",
        sample_tau=0.2,
        sample_eps=0.05,
        generator=None,
    )

    assert torch.equal(abs_mask, torch.tensor([[True, True, False, False]]))
    assert torch.equal(value_mask, torch.tensor([[False, True, True, False]]))


def test_sample_support_selection_is_reproducible_with_generator() -> None:
    group = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    generator_a = torch.Generator().manual_seed(1234)
    generator_b = torch.Generator().manual_seed(1234)

    mask_a = select_support_mask(
        group,
        k=2,
        score="value",
        support_selection="sample",
        sample_tau=0.2,
        sample_eps=0.05,
        generator=generator_a,
    )
    mask_b = select_support_mask(
        group,
        k=2,
        score="value",
        support_selection="sample",
        sample_tau=0.2,
        sample_eps=0.05,
        generator=generator_b,
    )

    assert torch.equal(mask_a, mask_b)
    assert int(mask_a.sum().item()) == 2


def test_sample_helper_keeps_existing_without_replacement_behavior() -> None:
    scores = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    generator = torch.Generator().manual_seed(7)

    selected = sample_k_without_replacement(
        scores,
        k=3,
        tau=0.2,
        eps=0.05,
        generator=generator,
    )

    assert selected.shape == (1, 3)
    assert selected.unique().numel() == 3


def test_best_subset_is_not_silently_approximated_in_repair_layer() -> None:
    group = torch.tensor([[0.1, 0.9, 0.6]])

    with pytest.raises(NotImplementedError, match="acquisition-aware support search"):
        select_support_mask(
            group,
            k=2,
            score="value",
            support_selection="best_subset",
            sample_tau=0.2,
            sample_eps=0.05,
            generator=None,
        )


def test_k_sparse_repair_uses_shared_support_selector() -> None:
    bounds = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    repair = make_k_sparse_linear_constraints_repair(
        bounds=bounds,
        comp_idx=[0, 1, 2, 3],
        k=2,
        score="value",
        support_selection="topk",
        max_iters=0,
    )

    result = repair(torch.tensor([[0.1, 0.9, 0.6, 0.2]]))

    assert torch.equal(result, torch.tensor([[0.0, 0.9, 0.6, 0.0]]))


def test_k_sparse_repair_rejects_unresolved_best_subset() -> None:
    bounds = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )
    repair = make_k_sparse_linear_constraints_repair(
        bounds=bounds,
        comp_idx=[0, 1, 2],
        k=2,
        support_selection="best_subset",
    )

    with pytest.raises(NotImplementedError, match="resolved before k-sparse repair"):
        repair(torch.tensor([[0.1, 0.9, 0.6]]))
