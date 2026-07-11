from __future__ import annotations

import torch

from bochan.tabular.candidate_outputs import _select_best_candidate_set


def test_select_best_candidate_set_from_deepgp_batch() -> None:
    candidates = torch.arange(10 * 2 * 5, dtype=torch.double).reshape(10, 2, 5)
    acq_value = torch.tensor([0.1, 0.4, 0.2, 0.3, 0.5, 0.9, 0.8, 0.7, 0.6, 0.0], dtype=torch.double)

    selected, selected_value = _select_best_candidate_set(
        candidates,
        acq_value,
        q=2,
        return_best_only=True,
    )

    assert selected.shape == torch.Size([2, 5])
    assert torch.equal(selected, candidates[5])
    assert torch.equal(selected_value, acq_value[5])


def test_select_best_candidate_set_keeps_standard_output() -> None:
    candidates = torch.rand(2, 5, dtype=torch.double)
    acq_value = torch.tensor(0.7, dtype=torch.double)

    selected, selected_value = _select_best_candidate_set(
        candidates,
        acq_value,
        q=2,
        return_best_only=True,
    )

    assert selected is candidates
    assert selected_value is acq_value


def test_select_best_candidate_set_preserves_all_sets_when_requested() -> None:
    candidates = torch.rand(3, 2, 5, dtype=torch.double)
    acq_value = torch.rand(3, dtype=torch.double)

    selected, selected_value = _select_best_candidate_set(
        candidates,
        acq_value,
        q=2,
        return_best_only=False,
    )

    assert selected is candidates
    assert selected_value is acq_value
