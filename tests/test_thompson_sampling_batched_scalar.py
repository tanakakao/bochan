from __future__ import annotations

import torch

from bochan.optim.thompson_sampling_adapter import ThompsonScalarizedObjective


class _BatchedScalarObjective:
    _is_mo = False

    def forward(self, samples: torch.Tensor, X=None) -> torch.Tensor:
        del X
        return samples[..., 0]


def test_batched_scalar_objective_preserves_candidate_axis() -> None:
    X = torch.zeros(7, 2, dtype=torch.double)
    samples = torch.rand(3, 4, 7, 1, dtype=torch.double)
    objective = ThompsonScalarizedObjective(
        objective=_BatchedScalarObjective(),
    )

    scores = objective(samples, X=X)

    assert scores.shape == torch.Size([3, 4, 7])
    torch.testing.assert_close(scores, samples[..., 0])
