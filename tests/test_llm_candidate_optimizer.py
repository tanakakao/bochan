from __future__ import annotations

import torch

from bochan.api import OptimizeConfig, optimize_candidates
from bochan.optim import optimize_acqf_llm_candidate_set


class SumAcquisition:
    def __call__(self, X):
        # X shape: batch x q x d. The LLM optimizer evaluates candidates with q=1.
        return X.squeeze(-2).sum(dim=-1)


def test_llm_candidate_optimizer_reranks_explicit_candidate_set():
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    candidate_set = torch.tensor(
        [
            [0.1, 0.1],
            [0.9, 0.8],
            [0.2, 0.7],
        ],
        dtype=torch.double,
    )

    candidates, values = optimize_acqf_llm_candidate_set(
        acq_function=SumAcquisition(),
        bounds=bounds,
        q=2,
        candidate_set=candidate_set,
        n_llm_candidates=3,
    )

    assert candidates.shape == torch.Size([2, 2])
    assert torch.allclose(candidates[0], torch.tensor([0.9, 0.8], dtype=torch.double))
    assert torch.all(values[:-1] >= values[1:])


def test_public_optimize_config_dispatches_llm_candidate_set():
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    config = OptimizeConfig(
        optimizer="llm_candidate_set",
        q=1,
        optimizer_kwargs={
            "candidate_set": [[0.1, 0.2], [0.8, 0.9]],
            "n_llm_candidates": 2,
        },
    )

    candidates, values = optimize_candidates(SumAcquisition(), bounds, config)

    assert candidates.shape == torch.Size([1, 2])
    assert torch.allclose(candidates[0], torch.tensor([0.8, 0.9], dtype=torch.double))
    assert values.shape == torch.Size([1])
