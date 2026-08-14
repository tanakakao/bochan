from __future__ import annotations

import torch

from bochan.acquisition.multiclass.active_learning.hetero_multi_output import (
    _align_score_to_weight,
)


def test_aligns_output_first_score_to_batch_output_weight() -> None:
    score = torch.arange(6, dtype=torch.double).reshape(2, 3)
    weight = torch.ones(3, 2, dtype=torch.double)

    aligned = _align_score_to_weight(score, weight)

    torch.testing.assert_close(aligned, score.T)
    assert aligned.shape == weight.shape


def test_averages_leading_sample_axis_when_weight_has_no_sample_axis() -> None:
    score = torch.arange(24, dtype=torch.double).reshape(4, 3, 2)
    weight = torch.ones(3, 2, dtype=torch.double)

    aligned = _align_score_to_weight(score, weight)

    torch.testing.assert_close(aligned, score.mean(dim=0))
    assert aligned.shape == weight.shape
